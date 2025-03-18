# import
## batteries
import os
import sys
import re
import logging
import requests
from typing import Annotated, Dict, Any, Optional, List
## 3rd party
from langchain_core.tools import tool
from bs4 import BeautifulSoup
## package
from SRAgent.tools.utils import determine_database

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool
def get_pmid_from_geo(
    geo_accession: Annotated[str, "GEO accession number (e.g., GSE159812)"]
) -> Annotated[str, "PMID directly extracted from the GEO page"]:
    """
    Extract the PMID directly from a GEO page by parsing the HTML.
    This is the most reliable way to get the correct PMID for a GEO dataset.
    
    Args:
        geo_accession: The GEO accession number (GSE, GDS, etc.)
        
    Returns:
        The PMID as a string if found, or an error message if not found.
    """
    try:
        # Verify this is a GEO accession
        if not geo_accession.startswith(('GSE', 'GDS', 'GSM')):
            return f"Error: {geo_accession} does not appear to be a valid GEO accession number. Valid accessions start with GSE, GDS, or GSM."
        
        # Construct the URL
        url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={geo_accession}"
        
        # Make the request
        logger.info(f"Fetching GEO page for {geo_accession}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for the citation block
        # In GEO pages, the citation is usually in a row with the label "Citation(s)"
        citation_row = soup.find(string=re.compile(r'Citation', re.IGNORECASE))
        if citation_row and citation_row.parent and citation_row.parent.parent:
            citation_tr = citation_row.parent.parent
            
            # Find the PMID which is usually in a span with class "pubmed_id"
            pmid_span = citation_tr.find('span', class_='pubmed_id')
            if pmid_span and pmid_span.has_attr('id'):
                pmid = pmid_span['id']
                logger.info(f"Found PMID {pmid} for {geo_accession}")
                return pmid
            
            # Alternative: look for a link to PubMed
            pubmed_link = citation_tr.find('a', href=re.compile(r'/pubmed/'))
            if pubmed_link:
                # Extract the PMID from the link text or href
                pmid_match = re.search(r'/pubmed/(\d+)', pubmed_link['href'])
                if pmid_match:
                    pmid = pmid_match.group(1)
                    logger.info(f"Found PMID {pmid} for {geo_accession} from link")
                    return pmid
                elif pubmed_link.string and pubmed_link.string.isdigit():
                    pmid = pubmed_link.string
                    logger.info(f"Found PMID {pmid} for {geo_accession} from link text")
                    return pmid
        
        # If we get here, we didn't find a citation
        logger.info(f"No citation found for {geo_accession}")
        return f"No PMID found for {geo_accession} on its GEO page."
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return f"Error accessing GEO page: {str(e)}"
    except Exception as e:
        logger.error(f"Error extracting PMID from GEO page: {e}")
        return f"Error extracting PMID: {str(e)}"

# For testing
if __name__ == "__main__":
    # Test with a known GEO accession
    test_accession = "GSE159812"
    result = get_pmid_from_geo(test_accession)
    print(result) 