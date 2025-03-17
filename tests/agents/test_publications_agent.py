#!/usr/bin/env python

import os
import sys
import asyncio
from dotenv import load_dotenv
from Bio import Entrez

# Add the SRAgent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the function to test
from SRAgent.agents.publications import create_publications_agent_stream
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

async def test_publications_agent():
    """Test the publications agent with different accessions"""
    print("\n=== Testing publications agent ===")
    
    # Set email and API key for Entrez
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
    # Test with different accessions
    test_accessions = [
        "PRJNA1210001",  # The one that was failing before
        "GSE63525",      # One that works well
    ]
    
    for acc in test_accessions:
        print(f"\n--- Finding publications for: {acc} ---")
        input_message = {"messages": [HumanMessage(content=f"Find publications for {acc}")]}
        result = await create_publications_agent_stream(input_message, summarize_steps=False)
        print(result)

if __name__ == "__main__":
    print("Starting publications agent test...")
    asyncio.run(test_publications_agent())
    print("\nTest completed!") 