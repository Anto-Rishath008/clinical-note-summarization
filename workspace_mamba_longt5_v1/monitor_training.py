"""
Live Training Monitor - V3  (ROUGE-L Focused)
======================================================
Streams logs/v2_run/train.log to console in real-time.
**Every step** that has a ROUGE-L value is highlighted with trend arrows.
Full ROUGE-L history table shown on startup and periodically.

Usage:
  python monitor_training.py              <- monitors default v2_run
  python monitor_training.py memory32_run <- monitors any named run

Uses binary-mode I/O for file seeking (safe on Windows CRLF / UTF-8).
"""
import sys, os, time, glob, re
from pathlib import Path

# Windows UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent

# Allow overriding run name via CLI, default to v2_run
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "v2_run"

LOG  = ROOT / "logs" / RUN_NAME / "train.log"
ERR  = ROOT / "logs" / RUN_NAME / "train_err.log"
CKPT = ROOT / "checkpoints" / RUN_NAME

# Load max_steps from config if available
MAX_STEPS = 50000
EVAL_STEPS = 1000
config_path = ROOT / "configs" / "full_train.yaml"
if config_path.exists():
    try:
        import yaml
        cfg = yaml.safe_load(open(config_path))
        MAX_STEPS = cfg.get("training", {}).get("max_steps", 50000)
        EVAL_STEPS = cfg.get("training", {}).get("eval_steps", 1000)
    except Exception:
        pass

# ─── Tracking state ─────────────────────────────────────────────────────────
# All known ROUGE-L values: list of (step, rouge_l, rouge_1, rouge_2, val_loss)
rouge_history = []
# Track the number of known evals to detect new ones
last_known_eval_count = 0


def get_current_step_from_log():
    """Parse the latest step number from the log file."""
    try:
        with open(LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read().decode("utf-8", errors="replace")
        for line in reversed(tail.splitlines()):
            if "Step " in line and "/" in line and "Loss:" in line:
                try:
                    part = line.split("Step ")[1].split("/")[0].strip()
                    return int(part)
                except Exception:
                    pass
    except Exception:
        pass
    return None


def get_current_loss_from_log():
    """Parse the latest loss from the log file."""
    try:
        with open(LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read().decode("utf-8", errors="replace")
        for line in reversed(tail.splitlines()):
            m = re.search(r"Loss:\s*([\d.]+)", line)
            if m and "Step " in line:
                return float(m.group(1))
    except Exception:
        pass
    return None


def get_all_eval_rouge():
    """Parse ALL validation ROUGE scores with step numbers from log.
    Returns list of (step, rouge_l, rouge_1, rouge_2, val_loss)."""
    results = []
    try:
        with open(LOG, "rb") as f:
            text = f.read().decode("utf-8", errors="replace")
        lines = text.splitlines()
        current_step = None
        for line in lines:
            m_step = re.search(r"EVALUATION at Step (\d+)", line)
            if m_step:
                current_step = int(m_step.group(1))
            m_val = re.search(
                r"Val Loss:\s*([\d.]+)\s*\|\s*ROUGE-1:\s*([\d.]+)\s*\|\s*ROUGE-2:\s*([\d.]+)\s*\|\s*ROUGE-L:\s*([\d.]+)",
                line,
            )
            if m_val and current_step is not None:
                val_loss = float(m_val.group(1))
                rouge_1 = float(m_val.group(2))
                rouge_2 = float(m_val.group(3))
                rouge_l = float(m_val.group(4))
                results.append((current_step, rouge_l, rouge_1, rouge_2, val_loss))
                current_step = None
    except Exception:
        pass
    return results


def trend_arrow(current, previous):
    """Return a colored trend arrow comparing current to previous ROUGE-L."""
    if previous is None:
        return "  "
    diff = current - previous
    if diff > 0.005:
        return "▲▲"  # strong increase
    elif diff > 0.001:
        return "▲ "  # mild increase
    elif diff > -0.001:
        return "━ "  # flat
    elif diff > -0.005:
        return "▼ "  # mild decrease
    else:
        return "▼▼"  # strong decrease


def bar_chart(value, max_val=0.35, width=25):
    """Create a simple text bar chart for ROUGE-L value."""
    filled = int(min(value / max_val, 1.0) * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    return bar


def format_rouge_table(history, title="ROUGE-L EVALUATION HISTORY", last_n=None):
    """Format a detailed ROUGE-L table with trends and bar charts."""
    if not history:
        return "  No evaluation data yet.\n"

    data = history if last_n is None else history[-last_n:]
    best_rl = max(h[1] for h in history)
    best_step = [h[0] for h in history if h[1] == best_rl][-1]

    lines = []
    lines.append("")
    lines.append(f"  ┌─────────────────────────────────────────────────────────────────────────┐")
    lines.append(f"  │  {title:<71}│")
    lines.append(f"  ├─────────┬──────────┬──────────┬──────────┬──────────┬────┬─────────────┤")
    lines.append(f"  │  Step   │ ROUGE-L  │ ROUGE-1  │ ROUGE-2  │ Val Loss │ Δ  │ Visual      │")
    lines.append(f"  ├─────────┼──────────┼──────────┼──────────┼──────────┼────┼─────────────┤")

    prev_rl = None
    for i, (step, rl, r1, r2, vl) in enumerate(data):
        arrow = trend_arrow(rl, prev_rl)
        bar = bar_chart(rl, max_val=0.35, width=11)
        marker = " ★" if step == best_step else "  "
        diff_str = f"{rl - prev_rl:+.4f}" if prev_rl is not None else "  --  "

        lines.append(
            f"  │ {step:>6} │ {rl:.4f}  │ {r1:.4f}  │ {r2:.4f}  │ {vl:.4f}  │ {arrow} │ {bar} │{marker}"
        )
        prev_rl = rl

    lines.append(f"  └─────────┴──────────┴──────────┴──────────┴──────────┴────┴─────────────┘")

    # Summary line
    current_rl = history[-1][1]
    improving_count = sum(
        1 for i in range(1, len(history)) if history[i][1] > history[i - 1][1]
    )
    total_evals = len(history)
    lines.append(f"  ★ Best: {best_rl:.4f} at step {best_step} | Current: {current_rl:.4f} | "
                 f"Improving: {improving_count}/{total_evals - 1} evals")

    # Consecutive trend (last N)
    if len(history) >= 3:
        recent = [h[1] for h in history[-5:]]
        consec_up = 0
        consec_down = 0
        for j in range(len(recent) - 1, 0, -1):
            if recent[j] > recent[j - 1]:
                consec_up += 1
            else:
                break
        for j in range(len(recent) - 1, 0, -1):
            if recent[j] < recent[j - 1]:
                consec_down += 1
            else:
                break
        if consec_up >= 2:
            lines.append(f"  >>> ROUGE-L RISING for {consec_up} consecutive evals! <<<")
        elif consec_down >= 3:
            lines.append(f"  !!! ROUGE-L FALLING for {consec_down} consecutive evals !!!")

    lines.append("")
    return "\n".join(lines)


def plateau_analysis(eval_history, patience=5):
    """Analyse whether training is in a plateau."""
    if len(eval_history) < 3:
        return {"status": "insufficient_data"}

    steps  = [e[0] for e in eval_history]
    rouges = [e[1] for e in eval_history]
    losses = [e[4] for e in eval_history]

    best_idx = rouges.index(max(rouges))
    best_rouge = rouges[best_idx]
    best_step = steps[best_idx]
    steps_since_best = steps[-1] - best_step

    recent = rouges[-patience:]
    window = rouges[max(0, len(rouges) - 2 * patience) : -patience] or rouges[:-patience]

    recent_avg = sum(recent) / len(recent)
    window_avg = sum(window) / len(window) if window else recent_avg

    recent_losses = losses[-3:]
    loss_diverging = len(recent_losses) >= 3 and all(
        recent_losses[i] < recent_losses[i + 1]
        for i in range(len(recent_losses) - 1)
    )

    if len(recent) >= 3:
        first_half = recent[: len(recent) // 2]
        second_half = recent[len(recent) // 2 :]
        fh_avg = sum(first_half) / len(first_half)
        sh_avg = sum(second_half) / len(second_half)
        trend_delta = sh_avg - fh_avg
    else:
        trend_delta = 0.0

    if loss_diverging and steps_since_best > patience * 1000:
        status = "diverging"
    elif steps_since_best >= patience * 1000:
        status = "plateau"
    elif trend_delta > 0.002:
        status = "improving"
    else:
        status = "plateau"

    return {
        "status": status,
        "best_step": best_step,
        "best_rouge": best_rouge,
        "current_rouge": rouges[-1],
        "steps_since_best": steps_since_best,
        "trend_delta": trend_delta,
        "recent_avg": recent_avg,
        "window_avg": window_avg,
        "loss_diverging": loss_diverging,
        "n_evals": len(eval_history),
    }


def format_plateau_report(info):
    """Format plateau analysis into a readable string."""
    if info["status"] == "insufficient_data":
        return "  [PLATEAU] Not enough evals yet to determine trend."

    status_map = {
        "improving": "IMPROVING  ▲",
        "plateau": "PLATEAU    ━",
        "diverging": "DIVERGING  ▼",
    }
    status_str = status_map.get(info["status"], info["status"])

    lines = [
        "",
        "  ╔══════════════════════════════════════════════════════════╗",
        f"  ║  TREND ANALYSIS  —  Status: {status_str:<27}║",
        "  ╠══════════════════════════════════════════════════════════╣",
        f"  ║  ★ Best ROUGE-L : {info['best_rouge']:.4f}  at step {info['best_step']:<6}               ║",
        f"  ║    Current      : {info['current_rouge']:.4f}  (steps since best: {info['steps_since_best']:<6})   ║",
        f"  ║    Recent avg   : {info['recent_avg']:.4f}  |  Prev window avg: {info['window_avg']:.4f}   ║",
        f"  ║    Trend delta  : {info['trend_delta']:+.4f}  (+ve = ROUGE-L improving)        ║",
        f"  ║    Val loss     : {'RISING (possible overfit!)   ' if info['loss_diverging'] else 'stable/falling (good)    '}        ║",
        "  ╚══════════════════════════════════════════════════════════╝",
        "",
    ]
    return "\n".join(lines)


def format_new_eval_alert(entry, prev_entry, best_rl):
    """Format a big alert when a new ROUGE-L evaluation is detected."""
    step, rl, r1, r2, vl = entry
    lines = []
    lines.append("")
    lines.append("  ╔══════════════════════════════════════════════════════════════╗")
    lines.append(f"  ║   ★★★  NEW ROUGE-L EVALUATION at Step {step:<6}  ★★★          ║")
    lines.append(f"  ╠══════════════════════════════════════════════════════════════╣")
    lines.append(f"  ║   ROUGE-L : {rl:.4f}   {bar_chart(rl, 0.35, 20)}          ║")
    lines.append(f"  ║   ROUGE-1 : {r1:.4f}   ROUGE-2 : {r2:.4f}   Val Loss: {vl:.4f}  ║")

    if prev_entry is not None:
        prev_rl = prev_entry[1]
        diff = rl - prev_rl
        if diff > 0:
            lines.append(f"  ║   Change  : {diff:+.4f}  ▲ INCREASING (was {prev_rl:.4f})                ║")
        elif diff < -0.001:
            lines.append(f"  ║   Change  : {diff:+.4f}  ▼ DECREASING (was {prev_rl:.4f})                ║")
        else:
            lines.append(f"  ║   Change  : {diff:+.4f}  ━ FLAT       (was {prev_rl:.4f})                ║")

    if rl >= best_rl:
        lines.append(f"  ║   >>> NEW ALL-TIME BEST ROUGE-L! <<<                        ║")
    else:
        gap = best_rl - rl
        lines.append(f"  ║   Gap to best: {gap:.4f}  (best={best_rl:.4f})                          ║")

    lines.append(f"  ╚══════════════════════════════════════════════════════════════╝")
    lines.append("")
    return "\n".join(lines)


def count_checkpoints():
    try:
        return len(glob.glob(str(CKPT / "checkpoint_step_*.pt")))
    except Exception:
        return 0


def get_latest_checkpoint():
    try:
        ckpts = sorted(
            glob.glob(str(CKPT / "checkpoint_step_*.pt")),
            key=lambda p: int(re.search(r"checkpoint_step_(\d+)\.pt", p).group(1)),
        )
        return Path(ckpts[-1]).name if ckpts else "none"
    except Exception:
        return "none"


def read_last_n_bytes(path, n=8192):
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - n))
        return f.read().decode("utf-8", errors="replace")


def format_eta(step, max_steps, rate_steps_per_sec):
    """ETA based on actual training rate."""
    if step and rate_steps_per_sec and rate_steps_per_sec > 0:
        remaining = (max_steps - step) / rate_steps_per_sec
        h, rem = divmod(int(remaining), 3600)
        m, _ = divmod(rem, 60)
        return f"{h}h {m:02d}m"
    return "calc..."


def next_eval_eta(step, rate):
    """Time until next ROUGE-L evaluation."""
    if step is None or rate is None or rate <= 0:
        return "?"
    next_eval_step = ((step // EVAL_STEPS) + 1) * EVAL_STEPS
    steps_remaining = next_eval_step - step
    secs = steps_remaining / rate
    m, s = divmod(int(secs), 60)
    return f"~{m}m{s:02d}s (step {next_eval_step})"


# ═══════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════

print(flush=True)
print("=" * 76, flush=True)
print(f"  ROUGE-L FOCUSED TRAINING MONITOR  --  {RUN_NAME}", flush=True)
print(f"  V2 Model: RMSNorm + SwiGLU + BiMamba + CrossChunkAttn + GatedXAttn", flush=True)
print(f"  Data   : MIMIC-IV-BHC full dataset (270,031 samples)", flush=True)
print(f"  Log    : logs/{RUN_NAME}/train.log", flush=True)
print(f"  Steps  : {MAX_STEPS:,} total  |  Eval every: {EVAL_STEPS} steps", flush=True)
print(f"  Ctrl+C : stop monitoring (training keeps running)", flush=True)
print("=" * 76, flush=True)
print(flush=True)

# ── Load initial ROUGE-L history ────────────────────────────────────────────
rouge_history = get_all_eval_rouge()
last_known_eval_count = len(rouge_history)

if rouge_history:
    print(format_rouge_table(rouge_history, "ROUGE-L HISTORY (from log)"), flush=True)
    pa = plateau_analysis(rouge_history)
    print(format_plateau_report(pa), flush=True)
else:
    print("  No evaluation history found in log yet.", flush=True)

# Wait for log file to appear
if not LOG.exists():
    print(f"Waiting for training to start (log: {LOG})...", flush=True)
    waited = 0
    while not LOG.exists():
        time.sleep(2)
        waited += 2
        if waited % 20 == 0:
            print(f"  ... waiting ({waited}s) ...", flush=True)
    print("\nLog appeared! Streaming...\n", flush=True)

# Show tail of existing log
tail_text = read_last_n_bytes(LOG, 4096)
tail_lines = [l for l in tail_text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if l.strip()]
print("--- Last known log entries ---", flush=True)
for line in tail_lines[-10:]:
    print(line, flush=True)
print("\n--- Streaming live output (ROUGE-L highlighted) ---\n", flush=True)

# Set starting position at end of file
with open(LOG, "rb") as _f:
    _f.seek(0, 2)
    last_pos = _f.tell()

last_status_time = time.time()
last_rouge_check_time = time.time()
last_ckpt_count = count_checkpoints()
monitor_start = time.time()
first_step_seen = None
first_step_time = None

# Main loop
try:
    while True:
        # ── Stream new log lines with ROUGE-L highlighting ──────────────
        try:
            current_size = LOG.stat().st_size
            if current_size > last_pos:
                with open(LOG, "rb") as f:
                    f.seek(last_pos)
                    new_bytes = f.read()
                last_pos = current_size
                new_text = new_bytes.decode("utf-8", errors="replace")
                new_text = new_text.replace("\r\n", "\n").replace("\r", "\n")
                for line in new_text.split("\n"):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    # Track step rate for ETA
                    m_step = re.search(r"Step\s+(\d+)/", stripped)
                    if m_step:
                        s = int(m_step.group(1))
                        if first_step_seen is None:
                            first_step_seen = s
                            first_step_time = time.time()

                    # Highlight ROUGE-L lines from evaluation prominently
                    if "Val Loss:" in stripped and "ROUGE-L:" in stripped:
                        print("  " + "=" * 60, flush=True)
                        print("  >>> " + stripped, flush=True)
                        print("  " + "=" * 60, flush=True)
                    elif "NEW BEST" in stripped:
                        print("  ★★★ " + stripped + " ★★★", flush=True)
                    elif "EVALUATION at Step" in stripped:
                        print("\n  ────── " + stripped + " ──────", flush=True)
                    else:
                        print(stripped, flush=True)
        except OSError:
            pass

        # ── Check for new ROUGE-L evaluations ──────────────────────────
        now = time.time()
        if now - last_rouge_check_time >= 5:  # check every 5 seconds
            last_rouge_check_time = now
            current_history = get_all_eval_rouge()
            if len(current_history) > last_known_eval_count:
                # New evaluation(s) detected!
                new_evals = current_history[last_known_eval_count:]
                rouge_history = current_history
                best_rl = max(h[1] for h in rouge_history)

                for idx, entry in enumerate(new_evals):
                    prev = (
                        rouge_history[last_known_eval_count + idx - 1]
                        if (last_known_eval_count + idx) > 0
                        else None
                    )
                    print(format_new_eval_alert(entry, prev, best_rl), flush=True)

                last_known_eval_count = len(rouge_history)

                # Show compact recent table after new eval
                print(
                    format_rouge_table(
                        rouge_history,
                        f"ROUGE-L TRACKER (latest {min(10, len(rouge_history))} evals)",
                        last_n=10,
                    ),
                    flush=True,
                )

        # ── Checkpoint alert ────────────────────────────────────────────
        try:
            ckpt_count = count_checkpoints()
            if ckpt_count > last_ckpt_count:
                latest = get_latest_checkpoint()
                print(
                    f"\n  *** NEW CHECKPOINT: {latest}  (total: {ckpt_count}) ***\n",
                    flush=True,
                )
                last_ckpt_count = ckpt_count
        except Exception:
            pass

        # ── Heartbeat every 30s with ROUGE-L focus ─────────────────────
        if now - last_status_time >= 30:
            step = get_current_step_from_log()
            loss = get_current_loss_from_log()
            elapsed = now - monitor_start

            # Compute rate
            rate = None
            if first_step_seen is not None and step is not None and first_step_time:
                dt = now - first_step_time
                ds = step - first_step_seen
                if dt > 0 and ds > 0:
                    rate = ds / dt

            pct = f"{step / MAX_STEPS * 100:.1f}%" if step else "?%"
            eta = format_eta(step, MAX_STEPS, rate)
            next_eval = next_eval_eta(step, rate)

            # ROUGE-L info
            if rouge_history:
                best_rl = max(h[1] for h in rouge_history)
                best_step = [h[0] for h in rouge_history if h[1] == best_rl][-1]
                current_rl = rouge_history[-1][1]
                rl_str = f"best={best_rl:.4f}@{best_step}  current={current_rl:.4f}"
            else:
                rl_str = "no evals yet"

            loss_str = f"{loss:.1f}" if loss else "?"

            print(
                f"\n  ┌── HEARTBEAT {time.strftime('%H:%M:%S')} ──────────────────────────────────────┐",
                flush=True,
            )
            print(
                f"  │  Step: {step}/{MAX_STEPS} ({pct})  |  Loss: {loss_str}  |  ETA: {eta}",
                flush=True,
            )
            print(
                f"  │  ROUGE-L: {rl_str}",
                flush=True,
            )
            print(
                f"  │  Next eval: {next_eval}  |  Checkpoints: {last_ckpt_count}",
                flush=True,
            )
            print(
                f"  └───────────────────────────────────────────────────────────┘\n",
                flush=True,
            )

            # Full plateau analysis every 3 minutes
            if int(elapsed) % 180 < 30:
                if rouge_history:
                    pa = plateau_analysis(rouge_history)
                    print(format_plateau_report(pa), flush=True)

            last_status_time = now

        time.sleep(1)

except KeyboardInterrupt:
    step = get_current_step_from_log()
    rouge_history = get_all_eval_rouge()
    print("\n" + "=" * 76, flush=True)
    print(f"  Monitor stopped at {time.strftime('%H:%M:%S')}", flush=True)
    print(f"  Training is still running in the background.", flush=True)
    if rouge_history:
        best_rl = max(h[1] for h in rouge_history)
        best_step = [h[0] for h in rouge_history if h[1] == best_rl][-1]
        print(f"  Step: {step}/{MAX_STEPS}  |  Best ROUGE-L: {best_rl:.4f} @ step {best_step}", flush=True)
        print(format_rouge_table(rouge_history, "FINAL ROUGE-L SUMMARY", last_n=15), flush=True)
        pa = plateau_analysis(rouge_history)
        print(format_plateau_report(pa), flush=True)
    print("=" * 76 + "\n", flush=True)
    sys.exit(0)
