@echo off
cd /d "c:\Users\antor\OneDrive\Desktop\3rd year\SEMESTER-6\NLP\Project\Codes\workspace_mamba_longt5_v1"
echo Starting training at %date% %time% > train_status.txt
echo PID check: >> train_status.txt
"C:\Users\antor\AppData\Local\Programs\Python\Python312\python.exe" -u src/train.py --config configs/full_train.yaml --resume checkpoints/v2_run >> train_restart_out.log 2>> train_restart_err.log
echo Training exited with code %errorlevel% at %date% %time% >> train_status.txt
