#!/usr/bin/env python3
"""
Simple validation script to test basic setup
"""

import os
import sys
from pathlib import Path

def validate_setup():
    """Validate the basic project setup"""
    print("🔍 Validating project setup...")
    
    # Check required files
    required_files = [
        "agent.py",
        "test_parsers.py", 
        "requirements.txt",
        ".env.example",
        "data/icici/icici sample.pdf",
        "data/icici/result.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
    
    # Check directories
    if not os.path.exists("custom_parsers"):
        print("❌ custom_parsers directory missing")
        return False
    else:
        print("✅ custom_parsers directory exists")
    
    # Check data files
    import pandas as pd
    try:
        df = pd.read_csv("data/icici/result.csv")
        expected_columns = ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
        if list(df.columns) == expected_columns:
            print(f"✅ CSV schema correct: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print(f"❌ CSV schema mismatch. Expected: {expected_columns}, Got: {list(df.columns)}")
            return False
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False
    
    print("🎉 Project setup validation complete!")
    return True

if __name__ == "__main__":
    success = validate_setup()
    sys.exit(0 if success else 1)
