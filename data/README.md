# Data Directory

## Overview
This directory contains datasets for clinical note summarization.

## Setup Instructions

**IMPORTANT**: Due to privacy regulations (HIPAA/PHI), the actual clinical datasets are NOT included in this repository.

### Option 1: Download MIMIC-III Clinical Notes
1. Request access: https://mimic.mit.edu/
2. Download clinical notes dataset
3. Place files in: \data/raw/\

### Option 2: Download PMC Patients Dataset
1. Access: https://github.com/zhao-zy15/PMC-Patients
2. Download and extract to: \data/raw/\

### Expected Structure
\\\
data/
├── README.md (this file)
├── raw/           # Original datasets (not tracked)
├── processed/     # Preprocessed data (not tracked)
├── tokenized/     # Tokenized data (not tracked)
└── sample_data.json  # Synthetic examples (tracked)
\\\

## Sample Data
A small synthetic sample is provided in \sample_data.json\ for testing purposes only.
