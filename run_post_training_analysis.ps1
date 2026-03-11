# Post-Training Analysis Runner
# Run this script after training completes to evaluate model and generate visualizations

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "POST-TRAINING ANALYSIS" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

# Set working directory
Set-Location "c:\Users\antor\OneDrive\Desktop\3rd year\SEMESTER-6\NLP\Project\Codes"

# Set Python path
$env:PYTHONPATH = "."

# Check if training is still running
$pythonProcess = Get-Process | Where-Object {$_.ProcessName -eq "python"}
if ($pythonProcess) {
    Write-Host "WARNING: Python training process is still running!" -ForegroundColor Yellow
    Write-Host "Process ID: $($pythonProcess.Id)" -ForegroundColor Yellow
    $response = Read-Host "Do you want to wait for it to finish? (y/n)"
    
    if ($response -eq 'y') {
        Write-Host "`nWaiting for training to complete..." -ForegroundColor Yellow
        Wait-Process -Id $pythonProcess.Id
        Write-Host "Training process completed!`n" -ForegroundColor Green
    }
}

# Check if best model exists
$bestModelPath = "artifacts\checkpoints\full_training_restart\best_model.pt"
if (-not (Test-Path $bestModelPath)) {
    Write-Host "ERROR: Best model not found at $bestModelPath" -ForegroundColor Red
    Write-Host "Make sure training has completed and saved checkpoints." -ForegroundColor Red
    exit 1
}

Write-Host "✓ Best model found: $bestModelPath`n" -ForegroundColor Green

# Create results directory with timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resultsDir = "results\analysis_$timestamp"

Write-Host "Running comprehensive analysis..." -ForegroundColor Cyan
Write-Host "Results will be saved to: $resultsDir`n" -ForegroundColor Cyan

# Run analysis script
python scripts/post_training_analysis.py `
    --checkpoint "$bestModelPath" `
    --config "configs/rtx4070_8gb.yaml" `
    --tokenizer "artifacts/tokenizer/spm.model" `
    --tokenized_dir "data/tokenized" `
    --output_dir "$resultsDir"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n============================================" -ForegroundColor Green
    Write-Host "ANALYSIS COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "============================================`n" -ForegroundColor Green
    
    Write-Host "Generated files:" -ForegroundColor Cyan
    Get-ChildItem $resultsDir | ForEach-Object {
        Write-Host "  - $($_.Name)" -ForegroundColor White
    }
    
    Write-Host "`nOpening results directory..." -ForegroundColor Cyan
    Start-Process $resultsDir
    
    # Display final ROUGE scores
    Write-Host "`n============================================" -ForegroundColor Cyan
    Write-Host "FINAL RESULTS SUMMARY" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    
    if (Test-Path "$resultsDir\validation_results.csv") {
        $valResults = Import-Csv "$resultsDir\validation_results.csv"
        $avgRouge1 = ($valResults | Measure-Object -Property rouge1 -Average).Average
        $avgRouge2 = ($valResults | Measure-Object -Property rouge2 -Average).Average
        $avgRougeL = ($valResults | Measure-Object -Property rougeL -Average).Average
        
        Write-Host "`nValidation Set Performance:" -ForegroundColor Yellow
        Write-Host "  ROUGE-1: $([math]::Round($avgRouge1, 4))" -ForegroundColor White
        Write-Host "  ROUGE-2: $([math]::Round($avgRouge2, 4))" -ForegroundColor White
        Write-Host "  ROUGE-L: $([math]::Round($avgRougeL, 4))" -ForegroundColor White
    }
    
    if (Test-Path "$resultsDir\test_results.csv") {
        $testResults = Import-Csv "$resultsDir\test_results.csv"
        $avgRouge1 = ($testResults | Measure-Object -Property rouge1 -Average).Average
        $avgRouge2 = ($testResults | Measure-Object -Property rouge2 -Average).Average
        $avgRougeL = ($testResults | Measure-Object -Property rougeL -Average).Average
        
        Write-Host "`nTest Set Performance:" -ForegroundColor Yellow
        Write-Host "  ROUGE-1: $([math]::Round($avgRouge1, 4))" -ForegroundColor White
        Write-Host "  ROUGE-2: $([math]::Round($avgRouge2, 4))" -ForegroundColor White
        Write-Host "  ROUGE-L: $([math]::Round($avgRougeL, 4))" -ForegroundColor White
    }
    
    Write-Host "`n============================================`n" -ForegroundColor Cyan
    
} else {
    Write-Host "`nERROR: Analysis failed!" -ForegroundColor Red
    Write-Host "Check the error messages above for details." -ForegroundColor Red
}
