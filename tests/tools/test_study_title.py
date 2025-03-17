#!/usr/bin/env python

import os
import sys
from dotenv import load_dotenv
from Bio import Entrez

# Add the SRAgent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the function to test
from SRAgent.tools.study_info import get_study_title_from_accession

# Load environment variables
load_dotenv()

def test_get_study_title():
    """Test the get_study_title_from_accession function"""
    print("\n=== Testing get_study_title_from_accession ===")
    
    # Set email for Entrez
    Entrez.email = os.getenv("EMAIL", "your.email@example.com")
    
    # Test with different accessions
    test_accessions = [
        "SRP557106",
        "PRJNA1210001",
        "GSE63525",
        "PRJNA192983"
    ]
    
    for acc in test_accessions:
        print(f"\n--- Getting title for: {acc} ---")
        result = get_study_title_from_accession(acc)
        print(result)

if __name__ == "__main__":
    print("Starting study title test...")
    test_get_study_title()
    print("\nTest completed!") 