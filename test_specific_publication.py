#!/usr/bin/env python

import os
import sys
import asyncio
import re
import logging
from typing import Tuple, Optional
from dotenv import load_dotenv
from Bio import Entrez
from langchain_core.messages import HumanMessage

# Add the SRAgent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the function to test
from SRAgent.agents.publications import create_publications_agent_stream, configure_logging

# Load environment variables
load_dotenv()

# Configure logging to suppress specific messages
configure_logging()

def extract_pmid_pmcid(result: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract PMID and PMCID from the agent's response.
    
    Args:
        result: The agent's response as a string.
        
    Returns:
        A tuple of (pmid, pmcid) extracted from the response, or (None, None) if not found.
    """
    pmid = None
    pmcid = None
    
    # Look for PMID in the response
    pmid_patterns = [
        r"PMID:?\s*(\d+)",
        r"PMID\s+(\d+)",
        r"PMID[:\s]*(\d+)",
        r"PubMed ID:?\s*(\d+)",
        r"PubMed\s+ID:?\s*(\d+)",
        r"\*\*PMID:\*\*\s*(\d+)",  # For markdown formatted output
        r"\*\*PMID\*\*:?\s*(\d+)",  # For markdown formatted output
        r"- \*\*PMID:\*\*\s*(\d+)",  # For markdown list items
        r"- \*\*PMID\*\*:?\s*(\d+)",  # For markdown list items
    ]
    
    for pattern in pmid_patterns:
        match = re.search(pattern, result, re.IGNORECASE)
        if match:
            pmid = match.group(1)
            break
    
    # Look for PMCID in the response
    pmcid_patterns = [
        r"PMCID:?\s*(PMC\d+)",
        r"PMCID\s+(PMC\d+)",
        r"PMCID[:\s]*(PMC\d+)",
        r"PMC:?\s*(\d+)",
        r"PMC\s+ID:?\s*(PMC\d+)",
        r"PMC\s+ID:?\s*(\d+)",
        r"\*\*PMCID:\*\*\s*(PMC\d+)",  # For markdown formatted output
        r"\*\*PMCID\*\*:?\s*(PMC\d+)",  # For markdown formatted output
        r"- \*\*PMCID:\*\*\s*(PMC\d+)",  # For markdown list items
        r"- \*\*PMCID\*\*:?\s*(PMC\d+)",  # For markdown list items
    ]
    
    for pattern in pmcid_patterns:
        match = re.search(pattern, result, re.IGNORECASE)
        if match:
            pmcid = match.group(1)
            # Add PMC prefix if it's just a number
            if not pmcid.startswith("PMC"):
                pmcid = f"PMC{pmcid}"
            break
    
    # If we still haven't found the PMID, try a more general approach
    if pmid is None:
        # Look for any number that appears after "PMID" in any format
        general_pmid_pattern = r"PMID.*?(\d+)"
        match = re.search(general_pmid_pattern, result, re.IGNORECASE)
        if match:
            pmid = match.group(1)
    
    # If we still haven't found the PMCID, try a more general approach
    if pmcid is None:
        # Look for PMC followed by numbers
        general_pmcid_pattern = r"PMC.*?(\d+)"
        match = re.search(general_pmcid_pattern, result, re.IGNORECASE)
        if match:
            pmcid = f"PMC{match.group(1)}"
    
    return pmid, pmcid

async def test_specific_publication():
    """Test the publications agent with both accession numbers at the same time"""
    print("\n=== Testing publications agent with both accessions at once ===")
    
    # Configure logging
    configure_logging()
    
    # Set email and API key for Entrez
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
    # Test with both accession numbers at once
    accession1 = "SRP270870"
    accession2 = "PRJNA644744"
    test_accessions = f"{accession1} and {accession2}"
    expected_pmid = "36602862"
    expected_pmcid = "PMC10014110"
    
    print(f"\n--- Finding publications for: {test_accessions} ---")
    input_message = {"messages": [HumanMessage(content=f"Find publications for {test_accessions}. These accessions are linked to the same publication.")]}
    
    try:
        # Run the agent
        start_time = asyncio.get_event_loop().time()
        result = await create_publications_agent_stream(input_message)
        end_time = asyncio.get_event_loop().time()
        
        # Print the result
        print(f"Agent response:\n{result}")
        
        # Extract PMID and PMCID from the result
        pmid, pmcid = extract_pmid_pmcid(result)
        
        # Check if the results match the expected values
        pmid_correct = pmid == expected_pmid
        pmcid_correct = pmcid == expected_pmcid
        
        # Print the results
        print(f"\nExtracted PMID: {pmid}, Expected: {expected_pmid}, Correct: {pmid_correct}")
        print(f"Extracted PMCID: {pmcid}, Expected: {expected_pmcid}, Correct: {pmcid_correct}")
        print(f"Overall success: {pmid_correct and pmcid_correct}")
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Configure logging
    configure_logging()
    
    print("Starting specific publication test...")
    asyncio.run(test_specific_publication())
    print("\nTest completed!") 