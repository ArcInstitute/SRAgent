#!/usr/bin/env python

import os
import sys
import json
import logging
from dotenv import load_dotenv

# Add the SRAgent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions to test
from SRAgent.tools.google_search import (
    google_search,
    shorten_results,
    find_publication_id_with_retry,
    google_search_tool
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_google_search():
    """Test the basic Google search function"""
    print("\n=== Testing google_search function ===")
    
    # Check if API keys are set
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    cse_id = os.getenv("GOOGLE_SEARCH_CSE_ID")
    
    if not api_key or not cse_id:
        print("WARNING: Google Search API key or CSE ID not found in environment variables.")
        print("API Key:", "Set" if api_key else "Not set")
        print("CSE ID:", "Set" if cse_id else "Not set")
    else:
        # Print partial keys for debugging (safely)
        print(f"API Key: {api_key[:4]}...{api_key[-4:]} (length: {len(api_key)})")
        print(f"CSE ID: {cse_id[:4]}...{cse_id[-4:]} (length: {len(cse_id)})")
    
    # Try different search terms
    search_terms = [
        "SRP557106",  # Original accession
        "GSE63525",   # GEO accession
        "bioinformatics"  # General term that should work
    ]
    
    for term in search_terms:
        print(f"\nSearching for: {term}")
        try:
            results = google_search(term, num=2)
            if isinstance(results, str) and results.startswith("ERROR"):
                print(f"Error: {results}")
            else:
                print(f"Search successful! Found {len(results)} results")
                # Print just the titles to keep output manageable
                for i, result in enumerate(results):
                    print(f"Result {i+1}: {result.get('title', 'No title')}")
        except Exception as e:
            print(f"Exception during search: {e}")

def test_find_publication_id_with_retry():
    """Test the retry function for finding publication IDs"""
    print("\n=== Testing find_publication_id_with_retry function ===")
    results = find_publication_id_with_retry("GSE63525", num_results=2)
    if isinstance(results, str) and results.startswith("ERROR"):
        print(f"Error: {results}")
    else:
        print(f"Search successful! Found {len(results)} results")
        # Print the shortened results
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result.get('title', 'No title')}")
            print(f"Link: {result.get('link', 'No link')}")

def test_google_search_tool():
    """Test the Google search tool function using the invoke method"""
    print("\n=== Testing google_search_tool function ===")
    try:
        # Use the invoke method instead of calling directly to avoid deprecation warning
        results = google_search_tool.invoke("GSE63525")
        print(results)
    except Exception as e:
        print(f"Error testing google_search_tool with invoke: {e}")
        print("Trying alternative method...")
        try:
            # Try using the underlying function directly
            from SRAgent.tools.google_search import find_publication_id_with_retry
            results = find_publication_id_with_retry("GSE63525", num_results=2)
            if isinstance(results, str) and results.startswith("ERROR"):
                print(f"Error: {results}")
            else:
                print(f"Search successful using direct function call! Found {len(results)} results")
                # Print the shortened results
                for i, result in enumerate(results):
                    print(f"Result {i+1}: {result.get('title', 'No title')}")
                    print(f"Link: {result.get('link', 'No link')}")
        except Exception as e2:
            print(f"Error with alternative method: {e2}")

if __name__ == "__main__":
    print("Starting Google search tests...")
    
    # Run the tests
    test_google_search()
    test_find_publication_id_with_retry()
    test_google_search_tool()
    
    print("\nTests completed!")
    
    # Print information about the oauth2client warning
    print("\nNote about 'file_cache is only supported with oauth2client<4.0.0' warning:")
    print("This is a known warning from the Google API client library.")
    print("It doesn't affect functionality but indicates that the library is using")
    print("a newer version of oauth2client that doesn't support file caching.")
    print("You can ignore this warning or downgrade oauth2client if needed.") 