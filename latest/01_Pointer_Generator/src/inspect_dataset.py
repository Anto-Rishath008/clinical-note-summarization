import pandas as pd

csv_path = r'mimic-iv-ext-bhc-labeled-clinical-notes-dataset-for-hospital-course-summarization-1.2.0\mimic-iv-ext-bhc-labeled-clinical-notes-dataset-for-hospital-course-summarization-1.2.0\mimic-iv-bhc.csv'

print("Loading dataset info...")
df = pd.read_csv(csv_path, nrows=3)

print(f"\nColumns: {df.columns.tolist()}")
print(f"Sample shapes:")
print(f"  Row 0 - input: {len(df.iloc[0]['input'])} chars, target: {len(df.iloc[0]['target'])} chars")
print(f"  input_tokens: {df.iloc[0]['input_tokens']}, target_tokens: {df.iloc[0]['target_tokens']}")

print(f"\n=== SAMPLE INPUT (first 300 chars) ===")
print(df.iloc[0]['input'][:300])

print(f"\n=== SAMPLE TARGET (first 300 chars) ===")
print(df.iloc[0]['target'][:300])

# Get total count
print(f"\n=== DATASET SIZE ===")
import subprocess
result = subprocess.run(['powershell', '-Command', f'(Get-Content "{csv_path}" | Measure-Object -Line).Lines'], 
                       capture_output=True, text=True)
print(f"Total rows (including header): {result.stdout.strip()}")
