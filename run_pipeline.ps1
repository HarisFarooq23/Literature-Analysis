# Simple one-shot runner for beginners.
# Usage (from this folder in PowerShell):
#   .\run_pipeline.ps1

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host ""
Write-Host "=== llm_extract simple pipeline ===" -ForegroundColor Cyan
Write-Host "Working folder: $PWD"
Write-Host ""

if (-not (Test-Path ".\input_pdfs")) {
    New-Item -ItemType Directory -Path ".\input_pdfs" | Out-Null
}

$pdfCount = (Get-ChildItem -Path ".\input_pdfs" -Filter "*.pdf" -ErrorAction SilentlyContinue).Count
if ($pdfCount -eq 0) {
    Write-Host "No PDF files found in .\input_pdfs" -ForegroundColor Yellow
    Write-Host "1) Copy your .pdf files into: $PWD\input_pdfs"
    Write-Host "2) Run this script again: .\run_pipeline.ps1"
    exit 1
}

Write-Host "Found $pdfCount PDF(s) in .\input_pdfs"
Write-Host "Starting: parse PDFs + LLM extraction..."
Write-Host ""

llm-extract from-pdfs --config config.yaml

Write-Host ""
Write-Host "Done. Open your results here:" -ForegroundColor Green
Write-Host "  $PWD\extraction_output\extracted_dataset.csv"
Write-Host "  $PWD\extraction_output\extracted_dataset.parquet"
Write-Host ""
