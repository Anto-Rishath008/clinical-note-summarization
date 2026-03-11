"""Simple training launcher with full output capture."""
import subprocess, sys, os

os.chdir(r"c:\Users\antor\OneDrive\Desktop\3rd year\SEMESTER-6\NLP\Project\Codes\workspace_mamba_longt5_v1")

cmd = [
    r"C:\Users\antor\AppData\Local\Programs\Python\Python312\python.exe",
    "-u", "src/train.py",
    "--config", "configs/full_train.yaml",
    "--resume", "checkpoints/v2_run"
]

print(f"Launching: {' '.join(cmd)}", flush=True)
proc = subprocess.Popen(
    cmd,
    stdout=sys.stdout,
    stderr=sys.stderr,
    bufsize=0
)
print(f"PID: {proc.pid}", flush=True)
proc.wait()
print(f"Exit code: {proc.returncode}", flush=True)
