#!/usr/bin/env python

import os
import sys
import asyncio
import json
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

async def test_specific_accessions():
    """Test the publications agent with specific accessions"""
    print("\n=== Testing publications agent with specific accessions ===")
    
    # Configure logging
    configure_logging()
    
    # Set email and API key for Entrez
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
    # Test cases
    test_cases = [
        {
            "name": "SRP270870_PRJNA644744",
            "accessions": "SRP270870 and PRJNA644744",
            "expected_pmid": "36602862",
            "expected_pmcid": "PMC10014110"
        },
        {
            "name": "ERP149679_PRJEB64504_E-MTAB-8142",
            "accessions": "ERP149679, PRJEB64504, and E-MTAB-8142",
            "expected_pmid": "33479125",
            "expected_pmcid": "PMC7611557"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- Finding publications for: {test_case['accessions']} ---")
        input_message = {"messages": [HumanMessage(content=f"Find publications for {test_case['accessions']}. These accessions are linked to the same publication.")]}
        
        try:
            # Run the agent
            start_time = asyncio.get_event_loop().time()
            result = await create_publications_agent_stream(input_message)
            end_time = asyncio.get_event_loop().time()
            
            # Print the result
            print(f"Agent response:\n{json.dumps(result, indent=2)}")
            
            # Get PMID and PMCID from the result
            pmid = result.get("pmid")
            pmcid = result.get("pmcid")
            
            # Check if the results match the expected values
            pmid_correct = pmid == test_case["expected_pmid"]
            pmcid_correct = pmcid == test_case["expected_pmcid"]
            
            # Print the results
            print(f"\nFound PMID: {pmid}, Expected: {test_case['expected_pmid']}, Correct: {pmid_correct}")
            print(f"Found PMCID: {pmcid}, Expected: {test_case['expected_pmcid']}, Correct: {pmcid_correct}")
            print(f"Overall success: {pmid_correct and pmcid_correct}")
            print(f"Execution time: {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Configure logging
    configure_logging()
    
    print("Starting specific accessions test...")
    asyncio.run(test_specific_accessions())
    print("\nTest completed!") 