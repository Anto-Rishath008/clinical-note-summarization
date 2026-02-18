"""
Live Training Monitor
=====================
Streams logs/memory32_run/train.log to console in real-time.
Shows training steps, loss, ROUGE-L evals, and checkpoint saves.
Run: python monitor_training.py

Uses binary-mode I/O for file seeking (safe on Windows CRLF / UTF-8).
"""
import sys, os, time, glob, signal
from pathlib import Path

# ── Windows UTF-8 output ──────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
LOG  = ROOT / "logs" / "memory32_run" / "train.log"
ERR  = ROOT / "logs" / "memory32_run" / "train_err.log"
CKPT = ROOT / "checkpoints" / "memory32_run"

# ── helpers ───────────────────────────────────────────────────────────────────
def get_current_step_from_tqdm():
    try:
        with open(ERR, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            # read last 2 KB — enough to find the most recent tqdm line
            f.seek(max(0, size - 2048))
            tail = f.read().decode("utf-8", errors="replace")
        for line in reversed(tail.splitlines()):
            if "Step ," in line:
                part = line.split("Step ,")[1].split("/")[0].strip()
                return int(part)
    except Exception:
        pass
    return None

def count_checkpoints():
    return len(glob.glob(str(CKPT / "checkpoint_step_*.pt")))

def get_latest_checkpoint():
    ckpts = sorted(glob.glob(str(CKPT / "checkpoint_step_*.pt")))
    return Path(ckpts[-1]).name if ckpts else "none"

def read_last_n_bytes(path, n=8192):
    """Read the last n bytes of a file in binary mode, decode as UTF-8."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - n))
        return f.read().decode("utf-8", errors="replace")

# ── header ────────────────────────────────────────────────────────────────────
print(flush=True)
print("=" * 72, flush=True)
print("  LIVE TRAINING MONITOR  --  memory32_run  (exp_memory32.yaml)", flush=True)
print("  Config : n_memory_tokens=32 | max_steps=50000 | lr=5e-5", flush=True)
print("  Data   : MIMIC-IV-BHC full dataset (270,031 samples)", flush=True)
print("  Log    : logs/memory32_run/train.log", flush=True)
print("  Press Ctrl+C to stop monitoring (training keeps running)", flush=True)
print("=" * 72, flush=True)
print(flush=True)

if not LOG.exists():
    print(f"ERROR: log file not found: {LOG}", flush=True)
    sys.exit(1)

# ── show last 15 lines for context ───────────────────────────────────────────
tail_text = read_last_n_bytes(LOG, 8192)
tail_lines = [l for l in tail_text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if l.strip()]
print("--- Last known log entries ---", flush=True)
for line in tail_lines[-15:]:
    print(line, flush=True)
print("", flush=True)
print("--- Now streaming live (new lines appear every ~7 min at log_steps=50) ---", flush=True)
print(flush=True)

# ── binary-safe file position ─────────────────────────────────────────────────
with open(LOG, "rb") as _f:
    _f.seek(0, 2)
    last_pos = _f.tell()          # byte offset — safe for binary seeks

last_status_time = time.time()
last_ckpt_count  = count_checkpoints()

# ── main loop ─────────────────────────────────────────────────────────────────
try:
    while True:
        # --- read new bytes since last_pos ---
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
            pass   # log file briefly locked by writer — retry next cycle

        # --- new checkpoint alert ---
        try:
            ckpt_count = count_checkpoints()
            if ckpt_count > last_ckpt_count:
                latest = get_latest_checkpoint()
                print(f"\n  *** NEW CHECKPOINT: {latest}  (total saved: {ckpt_count}) ***\n", flush=True)
                last_ckpt_count = ckpt_count
        except Exception:
            pass

        # --- periodic heartbeat every 60 s ---
        now = time.time()
        if now - last_status_time >= 60:
            step = get_current_step_from_tqdm()
            step_str = f"{step}" if step else "?"
            pct = f"{step/50000*100:.1f}" if step else "?"
            print(
                f"  [heartbeat {time.strftime('%H:%M:%S')}] "
                f"step={step_str}/50000 ({pct}%) | "
                f"ckpts={last_ckpt_count} | latest={get_latest_checkpoint()}",
                flush=True,
            )
            last_status_time = now

        time.sleep(1)

except KeyboardInterrupt:
    print("\n  [monitor stopped by user — training PID keeps running]\n", flush=True)
    sys.exit(0)
except Exception as e:
    print(f"\n  [monitor crashed: {e}]\n", flush=True)
    raise
