"""
Live Training Monitor - V2
===========================
Streams logs/v2_run/train.log to console in real-time.
Shows training steps, loss, ROUGE evals, and checkpoint saves.

Usage:
  python monitor_training.py              <- monitors default v2_run
  python monitor_training.py memory32_run <- monitors any named run

Uses binary-mode I/O for file seeking (safe on Windows CRLF / UTF-8).
"""
import sys, os, time, glob
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
config_path = ROOT / "configs" / "full_train.yaml"
if config_path.exists():
    try:
        import yaml
        cfg = yaml.safe_load(open(config_path))
        MAX_STEPS = cfg.get("training", {}).get("max_steps", 50000)
    except Exception:
        pass


def get_current_step_from_log():
    """Parse the latest step number from the log file."""
    try:
        with open(LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
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


def get_best_rouge():
    """Parse best ROUGE-L from log."""
    try:
        with open(LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            text = f.read().decode("utf-8", errors="replace")
        best = 0.0
        for line in text.splitlines():
            if "ROUGE-L:" in line:
                try:
                    val = float(line.split("ROUGE-L:")[1].split()[0])
                    if val > best:
                        best = val
                except Exception:
                    pass
        return best if best > 0 else None
    except Exception:
        return None


def count_checkpoints():
    try:
        return len(glob.glob(str(CKPT / "checkpoint_step_*.pt")))
    except Exception:
        return 0


def get_latest_checkpoint():
    try:
        ckpts = sorted(glob.glob(str(CKPT / "checkpoint_step_*.pt")))
        return Path(ckpts[-1]).name if ckpts else "none"
    except Exception:
        return "none"


def read_last_n_bytes(path, n=8192):
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - n))
        return f.read().decode("utf-8", errors="replace")


def format_eta(step, elapsed_sec):
    if step and step > 0 and elapsed_sec > 10:
        rate = step / elapsed_sec
        remaining = (MAX_STEPS - step) / max(rate, 1e-6)
        h, rem = divmod(int(remaining), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m"
    return "calc..."


# Header
print(flush=True)
print("=" * 72, flush=True)
print(f"  LIVE TRAINING MONITOR  --  {RUN_NAME}", flush=True)
print(f"  V2 Model: RMSNorm + SwiGLU + BiMamba + CrossChunkAttn + GatedXAttn", flush=True)
print(f"  Data   : MIMIC-IV-BHC full dataset (270,031 samples)", flush=True)
print(f"  Log    : logs/{RUN_NAME}/train.log", flush=True)
print(f"  Steps  : {MAX_STEPS:,} total", flush=True)
print(f"  Ctrl+C : stop monitoring (training keeps running)", flush=True)
print("=" * 72, flush=True)
print(flush=True)

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
tail_text = read_last_n_bytes(LOG, 8192)
tail_lines = [l for l in tail_text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if l.strip()]
print("--- Last known log entries ---", flush=True)
for line in tail_lines[-20:]:
    print(line, flush=True)
print("\n--- Streaming live output ---\n", flush=True)

# Set starting position at end of file
with open(LOG, "rb") as _f:
    _f.seek(0, 2)
    last_pos = _f.tell()

last_status_time = time.time()
last_ckpt_count  = count_checkpoints()
monitor_start    = time.time()

# Main loop
try:
    while True:
        # Stream new log lines
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
                    if line.strip():
                        print(line, flush=True)
        except OSError:
            pass

        # Checkpoint alert
        try:
            ckpt_count = count_checkpoints()
            if ckpt_count > last_ckpt_count:
                latest = get_latest_checkpoint()
                print(f"\n  *** NEW CHECKPOINT: {latest}  (total: {ckpt_count}) ***\n", flush=True)
                last_ckpt_count = ckpt_count
        except Exception:
            pass

        # Heartbeat every 60s
        now = time.time()
        if now - last_status_time >= 60:
            step      = get_current_step_from_log()
            best_rl   = get_best_rouge()
            elapsed   = now - monitor_start
            pct       = f"{step/MAX_STEPS*100:.1f}%" if step else "?%"
            rouge_str = f"{best_rl:.4f}" if best_rl else "?"
            eta       = format_eta(step, elapsed)
            print(
                f"  [heartbeat {time.strftime('%H:%M:%S')}] "
                f"step={step}/{MAX_STEPS} ({pct}) | "
                f"bestROUGE-L={rouge_str} | "
                f"ckpts={last_ckpt_count} | latest={get_latest_checkpoint()} | ETA={eta}",
                flush=True,
            )
            last_status_time = now

        time.sleep(1)

except KeyboardInterrupt:
    step    = get_current_step_from_log()
    best_rl = get_best_rouge()
    print("\n" + "=" * 72, flush=True)
    print(f"  Monitor stopped.  step={step}/{MAX_STEPS}  bestROUGE-L={best_rl}", flush=True)
    print(f"  Training is still running in background.", flush=True)
    print("=" * 72 + "\n", flush=True)
    sys.exit(0)
