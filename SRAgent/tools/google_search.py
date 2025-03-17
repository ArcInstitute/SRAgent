# import
## batteries
import os
import sys
import time
import logging
from typing import Annotated, List, Dict, Optional, Any
## 3rd party
from langchain_core.tools import tool
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API keys from environment
api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
cse_id = os.getenv("GOOGLE_SEARCH_CSE_ID")

def google_search(search_term, num=4, api_key=api_key, cse_id=cse_id, **kwargs):
    """Perform a Google search using the Custom Search API. Return top 4 results."""
    if not api_key or not cse_id:
        return f"ERROR: Google Search API key or CSE ID not found in environment variables."
    
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=search_term, cx=cse_id, num=num, **kwargs).execute()
        if int(res["searchInformation"]["totalResults"]) == 0:
            return []
        return res.get("items", [])
    except HttpError as e:
        logger.error(f"Google search API error: {e}")
        return f"ERROR: Google search API error: {e}"
    except Exception as e:
        logger.error(f"Error performing Google search: {e}")
        return f"ERROR: Error performing Google search: {e}"

def shorten_results(results: list):
    """Shorten the output from Google search API to only include the most relevant fields"""
    if isinstance(results, str) and results.startswith("ERROR"):
        return results
    
    pubmed_pagemap_keep = [
        "citation_publication_date",
        "citation_title",
        "citation_author_institution",
        "og:site_name",
        "citation_publisher",
        "citation_journal_title",
        "og:description",
        "citation_journal_abbrev",
        "og:title",
        "citation_author",
        "title",  # biorxiv-specific
    ]

    shortened_results = []
    for r in results:
        first_part = {k: r.get(k) for k in ["title", "link", "snippet"]}
        pagemap_part = {}
        pagemap = r.get("pagemap")
        if pagemap:
            metatags = pagemap.get("metatags")
            if metatags:
                pagemap_part = {k: v for k, v in metatags[0].items() if k in pubmed_pagemap_keep}
                # rename keys to avoid conflicts
                for col in list(set(pagemap_part.keys()) & set(first_part.keys())):
                    pagemap_part[f"pagemap_{col}"] = pagemap_part.pop(col)

        shortened_results.append({**first_part, **pagemap_part})

    return shortened_results

def find_publication_id_with_retry(
    query: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    authors: Optional[str] = None,
    institution: Optional[str] = None,
    year: Optional[str] = None,
    accession: Optional[str] = None,
    retries=3,
    backoff_factor=2,
    num_results: int = 4,
):
    """Retry the search for a publication ID using Google search."""
    for attempt in range(retries):
        try:
            # Call your function to get results
            results = google_search(query, num=num_results)
            if isinstance(results, str) and results.startswith("ERROR"):
                logger.error(f"Error in Google search: {results}")
                time.sleep(backoff_factor * (2**attempt))
                continue
                
            # Shorten results for readability
            shortened_results = shorten_results(results)
            
            # Return the results
            return shortened_results
        except HttpError as e:
            if e.resp.status == 429:  # HTTP 429 means "Too Many Requests"
                logger.error(f"Rate limit exceeded. Retrying in {backoff_factor} seconds...")
                time.sleep(backoff_factor * (2**attempt))
            else:
                logger.error(f"HTTP error: {e}")
                return f"ERROR: HTTP error: {e}"
        except Exception as e:
            logger.error(f"Error in find_publication_id_with_retry: {e}")
            return f"ERROR: Error in find_publication_id_with_retry: {e}"
    
    return f"ERROR: Failed to retrieve publication ID after {retries} attempts"

@tool
def google_search_tool(
    query: Annotated[str, "Search query for Google"],
    num_results: Annotated[int, "Number of results to return"] = 4,
) -> Annotated[str, "Google search results"]:
    """
    Search Google for information. Use this when you need to find publications
    by searching for accession numbers or study titles.
    
    For accession numbers, always put them in quotes to ensure exact matches.
    For example: "SRP557106" or "PRJNA1210001"
    """
    try:
        # Check if Google Search API key and CSE ID are available
        if not api_key or not cse_id:
            # Fall back to a mock implementation
            return f"Google Search API key or CSE ID not found in environment variables.\n" \
                   f"This is a mock Google search for: {query}\n" \
                   f"In a real implementation, this would use the Google Search API.\n" \
                   f"Please continue with the workflow using Entrez tools."
        
        # Perform the search with retry
        results = find_publication_id_with_retry(query, num_results=num_results)
        
        # Check if there was an error
        if isinstance(results, str) and results.startswith("ERROR"):
            return results
        
        # Format the results for readability
        formatted_results = "Google Search Results:\n\n"
        for i, result in enumerate(results):
            formatted_results += f"Result {i+1}:\n"
            formatted_results += f"Title: {result.get('title', 'N/A')}\n"
            formatted_results += f"Link: {result.get('link', 'N/A')}\n"
            formatted_results += f"Snippet: {result.get('snippet', 'N/A')}\n"
            
            # Add publication-specific information if available
            if "citation_publication_date" in result:
                formatted_results += f"Publication Date: {result.get('citation_publication_date', 'N/A')}\n"
            if "citation_author" in result:
                formatted_results += f"Authors: {result.get('citation_author', 'N/A')}\n"
            if "citation_author_institution" in result:
                formatted_results += f"Institution: {result.get('citation_author_institution', 'N/A')}\n"
            if "citation_journal_title" in result:
                formatted_results += f"Journal: {result.get('citation_journal_title', 'N/A')}\n"
            
            formatted_results += "\n"
        
        return formatted_results
    except Exception as e:
        logger.error(f"Error in google_search_tool: {e}")
        return f"Error performing Google search: {str(e)}" 