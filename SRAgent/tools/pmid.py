# import
## batteries
import os
import time
import logging
from typing import Annotated, Optional
## 3rd party
from Bio import Entrez
from langchain_core.tools import tool
## package
from SRAgent.tools.utils import set_entrez_access

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool
def pmcid_from_pmid(
    pmid: Annotated[str, "PMID of the publication"],
    retries: Annotated[int, "Number of retries"] = 5,
    backoff_factor: Annotated[int, "Backoff factor for retries"] = 1,
) -> Annotated[Optional[str], "PMCID of the publication if available"]:
    """
    Get the PMCID of a paper using its PMID using Entrez.
    Returns None if no PMCID is found or if there's an error.
    """
    set_entrez_access()
    params = {"dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "xml"}

    for attempt in range(retries):
        try:
            handle = Entrez.elink(**params)
            records = Entrez.read(handle)
            handle.close()

            if records:
                linksets = records[0].get("LinkSetDb", [])
                if linksets:
                    links = linksets[0].get("Link", [])
                    if links:
                        if len(links) > 1:
                            logger.warning(f"Multiple PMCID links found for PMID {pmid}")
                            # there is no basis to choose one here, so return None
                            return None
                        pmcid = links[0]["Id"]
                        return f"PMC{pmcid}"
                return "No PMCID found for this PMID"

        except Exception as e:
            if "429" in str(e):  # Check if error code 429 is in the exception message
                wait_time = backoff_factor * (2**attempt)
                logger.error(
                    f"Rate limit exceeded for PMID {pmid}. Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Request failed for PMID {pmid}: {e}")
                return f"Error retrieving PMCID: {str(e)}"

    return "Failed to retrieve PMCID after multiple attempts"

@tool
def pmid_from_pmcid(
    pmcid: Annotated[str, "PMCID of the publication"],
    retries: Annotated[int, "Number of retries"] = 5,
    backoff_factor: Annotated[int, "Backoff factor for retries"] = 1,
) -> Annotated[Optional[str], "PMID of the publication if available"]:
    """
    Get the PMID of a paper using its PMCID using Entrez.
    Returns None if no PMID is found or if there's an error.
    """
    set_entrez_access()
    
    # Remove PMC prefix if present
    if pmcid.startswith("PMC"):
        pmcid = pmcid.removeprefix("PMC")

    params = {"dbfrom": "pmc", "db": "pubmed", "id": pmcid, "retmode": "xml"}

    for attempt in range(retries):
        try:
            handle = Entrez.elink(**params)
            records = Entrez.read(handle)
            handle.close()

            if records:
                linksets = records[0].get("LinkSetDb", [])
                if linksets:
                    links = linksets[0].get("Link", [])
                    if links:
                        if len(links) > 1:
                            logger.warning(f"Multiple PMID links found for PMCID {pmcid}")
                            # there is no basis to choose one here, so return None
                            return None
                        pmid = links[0]["Id"]
                        return pmid
                return "No PMID found for this PMCID"

        except Exception as e:
            if "429" in str(e):  # Check if error code 429 is in the exception message
                wait_time = backoff_factor * (2**attempt)
                logger.error(
                    f"Rate limit exceeded for PMCID {pmcid}. Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Request failed for PMCID {pmcid}: {e}")
                return f"Error retrieving PMID: {str(e)}"

    return "Failed to retrieve PMID after multiple attempts"

@tool
def get_publication_details(
    pmid: Annotated[str, "PMID of the publication"],
) -> Annotated[dict, "Publication details"]:
    """
    Get detailed information about a publication using its PMID.
    Returns a dictionary with publication details.
    """
    set_entrez_access()
    
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        
        if not records or "PubmedArticle" not in records:
            return {"error": "No publication details found for this PMID"}
        
        article = records["PubmedArticle"][0]
        
        # Extract basic information
        medline_citation = article.get("MedlineCitation", {})
        article_data = medline_citation.get("Article", {})
        
        # Get title
        title = article_data.get("ArticleTitle", "No title available")
        
        # Get journal information
        journal = article_data.get("Journal", {})
        journal_title = journal.get("Title", "No journal title available")
        
        # Get publication date
        pub_date = journal.get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year", "Unknown")
        month = pub_date.get("Month", "Unknown")
        day = pub_date.get("Day", "Unknown")
        publication_date = f"{year}-{month}-{day}" if "Unknown" not in [year, month, day] else year
        
        # Get authors
        author_list = article_data.get("AuthorList", [])
        authors = []
        for author in author_list:
            if "LastName" in author and "ForeName" in author:
                authors.append(f"{author['LastName']} {author['ForeName']}")
            elif "LastName" in author:
                authors.append(author["LastName"])
            elif "CollectiveName" in author:
                authors.append(author["CollectiveName"])
        
        # Get abstract
        abstract_text = article_data.get("Abstract", {}).get("AbstractText", [])
        abstract = " ".join([str(text) for text in abstract_text]) if abstract_text else "No abstract available"
        
        # Get DOI
        article_id_list = article.get("PubmedData", {}).get("ArticleIdList", [])
        doi = next((id_value for id_value in article_id_list if id_value.attributes.get("IdType") == "doi"), "No DOI available")
        
        # Get PMCID
        pmcid = pmcid_from_pmid(pmid)
        if pmcid and pmcid.startswith("Error") or pmcid == "No PMCID found for this PMID" or pmcid == "Failed to retrieve PMCID after multiple attempts":
            pmcid = "No PMCID available"
        
        # Compile results
        publication_details = {
            "pmid": pmid,
            "pmcid": pmcid,
            "doi": str(doi),
            "title": title,
            "journal": journal_title,
            "publication_date": publication_date,
            "authors": authors,
            "abstract": abstract
        }
        
        return publication_details
    
    except Exception as e:
        logger.error(f"Error retrieving publication details for PMID {pmid}: {e}")
        return {"error": f"Error retrieving publication details: {str(e)}"} 