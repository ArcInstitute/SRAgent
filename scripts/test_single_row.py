#!/usr/bin/env python
"""
Test script to process a single row from the sample CSV.
"""

import os
import sys
import asyncio
import pandas as pd
from Bio import Entrez

# Add the parent directory to the path so we can import from SRAgent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SRAgent.agents.publications import create_publications_agent_stream, configure_logging
from scripts.process_publications_df import find_publication_for_study

async def main():
    """Main function to run the test."""
    # Configure logging
    configure_logging()
    
    # Set email and API key for Entrez
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
    # Read the sample CSV
    df = pd.read_csv("scripts/sample_accessions.csv")
    
    # Get the first row
    row = df.iloc[3]  # Using the ERP156277 row which we know has a publication
    
    print(f"Processing row: {row.to_dict()}")
    
    # Find publication for the row
    result = await find_publication_for_study(row)
    
    # Print the result
    print("\nResult:")
    for key, value in result.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(main()) 