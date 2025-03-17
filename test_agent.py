import os
import asyncio
import json
import logging
from langchain_core.messages import HumanMessage
from Bio import Entrez
from SRAgent.agents.publications import create_publications_agent_stream

# Set up logging
logging.basicConfig(level=logging.INFO)

async def main():
    # Set email and API key for Entrez
    Entrez.email = os.getenv("EMAIL", "test@example.com")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
    print("Starting test...")
    
    # Create input message
    accessions = "ERP156277 and PRJEB71477 and E-MTAB-13085"
    input_message = {"messages": [{"role": "user", "content": f"Find publications for {accessions}. These accessions are linked to the same publication."}]}
    
    print(f"Input message: {input_message}")
    
    try:
        # Run the agent
        print("Running agent...")
        result = await create_publications_agent_stream(input_message)
        
        # Print the result
        print("\nRaw result:")
        print(json.dumps(result, indent=2))
        
        # Print specific fields
        print("\nExtracted fields:")
        print(f"PMID: {result.get('pmid')}")
        print(f"PMCID: {result.get('pmcid')}")
        print(f"Title: {result.get('title')}")
        print(f"Source: {result.get('source')}")
        print(f"Multiple publications: {result.get('multiple_publications')}")
        
        # Test PMCID to PMID conversion directly
        if result.get('pmcid') and not result.get('pmid'):
            print("\nTesting PMCID to PMID conversion directly:")
            from SRAgent.tools.pmid import pmid_from_pmcid
            pmcid = result.get('pmcid')
            print(f"PMCID: {pmcid}")
            pmid = pmid_from_pmcid(pmcid)
            print(f"PMID from conversion: {pmid}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 