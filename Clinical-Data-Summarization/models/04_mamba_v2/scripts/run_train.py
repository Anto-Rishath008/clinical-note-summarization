"""
Resilient training runner with auto-restart on crash.
Keeps restarting from the latest checkpoint until training completes (150K steps).
"""
import subprocess, sys, os, time, datetime

MAX_RETRIES = 50          # max consecutive restart attempts
COOLDOWN_SECS = 15        # wait between restarts to let GPU memory free
MAX_STEPS = 150000        # stop restarting once this step is reached

python = sys.executable
wd = os.path.dirname(os.path.abspath(__file__))
config = "configs/full_train.yaml"
ckpt_dir = "checkpoints/v2_run"

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def get_latest_step():
    """Read the latest checkpoint step number from disk."""
    import glob, re
    pattern = os.path.join(wd, ckpt_dir, "checkpoint_step_*.pt")
    files = glob.glob(pattern)
    if not files:
        return 0
    steps = []
    for f in files:
        m = re.search(r'checkpoint_step_(\d+)\.pt$', f)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else 0

def run_training():
    """Run one training session, return (exit_code, last_step)."""
    cmd = [python, "-u", "src/train.py", "--config", config, "--resume", ckpt_dir]
    log(f"CMD: {' '.join(cmd)}")
    
    stdout_log = os.path.join(wd, "train_stdout.log")
    stderr_log = os.path.join(wd, "train_stderr.log")
    
    with open(stdout_log, "a") as out, open(stderr_log, "a") as err:
        out.write(f"\n{'='*60}\n[RESTART] {datetime.datetime.now()}\n{'='*60}\n")
        err.write(f"\n{'='*60}\n[RESTART] {datetime.datetime.now()}\n{'='*60}\n")
        r = subprocess.run(cmd, cwd=wd, stdout=out, stderr=err)
    
    # Check for errors in stderr
    with open(stderr_log) as f:
        lines = f.readlines()
    errors = [l for l in lines[-80:] if "Training:" not in l and l.strip() 
              and "RESTART" not in l and "===" not in l]
    
    return r.returncode, errors

# ---- Main auto-restart loop ----
log("=" * 60)
log("RESILIENT TRAINING RUNNER")
log(f"Auto-restart enabled (max {MAX_RETRIES} retries, {COOLDOWN_SECS}s cooldown)")
log(f"Target: {MAX_STEPS} steps")
log("=" * 60)

attempt = 0
while attempt < MAX_RETRIES:
    step_before = get_latest_step()
    log(f"Attempt {attempt + 1}/{MAX_RETRIES} | Latest checkpoint: step {step_before}")
    
    if step_before >= MAX_STEPS:
        log(f"Training already reached {step_before} >= {MAX_STEPS}. Done!")
        break
    
    start = time.time()
    exit_code, errors = run_training()
    elapsed = time.time() - start
    step_after = get_latest_step()
    
    log(f"Training exited with code {exit_code} after {elapsed/60:.1f} min")
    log(f"Steps: {step_before} -> {step_after}")
    
    if step_after >= MAX_STEPS:
        log(f"Training completed! Reached step {step_after}.")
        break
    
    if exit_code == 0:
        log("Training exited cleanly (code 0). Done!")
        break
    
    # Training crashed — show errors if any
    if errors:
        log("=== RECENT ERRORS ===")
        for e in errors[-20:]:
            log(f"  {e.rstrip()}")
    else:
        log("No Python traceback found (process killed externally?)")
    
    # Check if progress was made
    if step_after == step_before and elapsed < 120:
        log("WARNING: No progress and quick exit — possible startup error")
        attempt += 3  # penalize to avoid infinite fast crash loops
    
    attempt += 1
    log(f"Cooling down {COOLDOWN_SECS}s before restart...")
    time.sleep(COOLDOWN_SECS)

if attempt >= MAX_RETRIES:
    log(f"GAVE UP after {MAX_RETRIES} attempts. Last step: {get_latest_step()}")
else:
    log(f"SUCCESS! Final step: {get_latest_step()}")
