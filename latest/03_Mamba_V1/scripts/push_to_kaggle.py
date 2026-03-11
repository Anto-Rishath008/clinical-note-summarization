#!/usr/bin/env python3
"""
Kaggle Push & Run Script
=========================

Push notebook to Kaggle and start GPU training from VS Code.

Usage:
    python push_to_kaggle.py --username YOUR_KAGGLE_USERNAME

Prerequisites:
1. Install kaggle: pip install kaggle
2. Get API token from https://www.kaggle.com/settings (Account -> API -> Create New Token)
3. Save kaggle.json to ~/.kaggle/kaggle.json (or C:\\Users\\YOU\\.kaggle\\kaggle.json on Windows)
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path


def setup_kaggle_credentials():
    """Check if Kaggle credentials are set up"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("=" * 60)
        print("KAGGLE SETUP REQUIRED")
        print("=" * 60)
        print("\n1. Go to: https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New Token'")
        print("4. Download kaggle.json")
        print(f"5. Save it to: {kaggle_json}")
        print("\nAlternatively, set environment variables:")
        print("  KAGGLE_USERNAME=your_username")
        print("  KAGGLE_KEY=your_api_key")
        print("=" * 60)
        return False
    
    # Check permissions on Unix
    if os.name != 'nt':
        os.chmod(kaggle_json, 0o600)
    
    return True


def update_metadata(username: str, notebook_dir: Path):
    """Update kernel-metadata.json with username"""
    metadata_path = notebook_dir / "kernel-metadata.json"
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Update ID with username
    metadata['id'] = f"{username}/mamba-transformer-clinical-summarization"
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Updated metadata with username: {username}")
    return metadata


def push_notebook(notebook_dir: Path):
    """Push notebook to Kaggle"""
    print("\n" + "=" * 60)
    print("PUSHING NOTEBOOK TO KAGGLE")
    print("=" * 60)
    
    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(notebook_dir)],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0


def check_status(kernel_id: str):
    """Check kernel execution status"""
    print("\n" + "=" * 60)
    print("CHECKING KERNEL STATUS")
    print("=" * 60)
    
    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel_id],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description='Push notebook to Kaggle')
    parser.add_argument('--username', type=str, required=True, 
                        help='Your Kaggle username')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check status, do not push')
    args = parser.parse_args()
    
    # Get notebook directory
    script_dir = Path(__file__).parent
    notebook_dir = script_dir / "notebooks"
    
    if not notebook_dir.exists():
        notebook_dir = script_dir.parent / "notebooks"
    
    if not notebook_dir.exists():
        print(f"ERROR: Notebook directory not found: {notebook_dir}")
        sys.exit(1)
    
    # Check credentials
    if not setup_kaggle_credentials():
        # Check environment variables
        if not os.environ.get('KAGGLE_USERNAME') or not os.environ.get('KAGGLE_KEY'):
            sys.exit(1)
    
    kernel_id = f"{args.username}/mamba-transformer-clinical-summarization"
    
    if args.check_only:
        check_status(kernel_id)
        return
    
    # Update metadata
    update_metadata(args.username, notebook_dir)
    
    # Push notebook
    if push_notebook(notebook_dir):
        print("\n✓ Notebook pushed successfully!")
        print(f"\nView at: https://www.kaggle.com/code/{kernel_id}")
        print("\nTo check status, run:")
        print(f"  kaggle kernels status {kernel_id}")
        print("\nOr run this script with --check-only:")
        print(f"  python {Path(__file__).name} --username {args.username} --check-only")
    else:
        print("\n✗ Failed to push notebook")
        sys.exit(1)


if __name__ == '__main__':
    main()
