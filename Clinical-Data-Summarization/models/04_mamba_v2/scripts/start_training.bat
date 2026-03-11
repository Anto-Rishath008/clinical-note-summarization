@echo off
cd /d "c:\Users\antor\OneDrive\Desktop\3rd year\SEMESTER-6\NLP\Project\Codes\workspace_mamba_longt5_v1"
"c:\Users\antor\OneDrive\Desktop\3rd year\SEMESTER-6\NLP\Project\Codes\.conda\python.exe" -u src/train.py --config configs/full_train.yaml --resume checkpoints/v2_run/best_model.pt
pause
