import os
import re
import random
import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple
from Bio import Entrez

# Base URLs for Entrez API
ELINK_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
EFETCH_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESUMMARY_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


async def fetch_url(
    session: aiohttp.ClientSession,
    url: str,
    params: Dict[str, Any],
    base_params: Optional[Dict[str, Any]] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    max_retries: int = 5,
) -> Optional[str]:
    """
    Helper function to fetch and parse from a URL with parameters.
    Implements retry logic with exponential backoff, attempting up to max_retries times.

    Args:
        session: The HTTP session to use for requests
        url: The URL to fetch from
        params: Parameters to include in the request
        base_params: Base parameters to include in all requests
        semaphore: Semaphore to limit concurrent connections
        max_retries: Maximum number of retry attempts

    Returns:
        The response text if successful, None if all retries failed
    """
    max_retries = 8
    retry_count = 0
    base_delay = 1.0  # Starting delay in seconds

    async with semaphore:
        while retry_count < max_retries:
            try:
                merged_params = {**base_params, **params}
                async with session.get(url, params=merged_params) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise Exception(f"HTTP error {response.status}: {text}")
                    return await response.text()
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    # If we've exhausted all retries, return None
                    return None

                # Calculate backoff delay with exponential increase and some jitter
                delay = (
                    base_delay
                    * (2 ** (retry_count - 1))
                    * (0.9 + 0.2 * random.random())
                )
                print(
                    f"Request to {url} failed (attempt {retry_count}/{max_retries}): {str(e)}. Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)


async def direct_sra_fetch(
    session: aiohttp.ClientSession,
    sra_id: str,
    base_params: Optional[Dict[str, Any]] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> List[str]:
    """
    Get SRX/ERX accession directly from SRA using efetch.
    This approach treats the ID as an SRA internal ID.

    Args:
        session: The HTTP session to use for requests
        sra_id: The SRA ID to fetch
        base_params: Base parameters to include in all requests
        semaphore: Semaphore to limit concurrent connections

    Returns:
        List of SRX/ERX accessions found
    """
    params = {"db": "sra", "id": sra_id, "retmode": "xml"}

    try:
        xml_text = await fetch_url(
            session, EFETCH_BASE_URL, params, base_params, semaphore
        )
        if not xml_text:
            return []
        root = ET.fromstring(xml_text)

        accessions = []
        # Look for experiment accessions in various places
        for exp in root.findall(".//EXPERIMENT"):
            accession = exp.get("accession")
            if accession and (
                accession.startswith("SRX") or accession.startswith("ERX")
            ):
                accessions.append(accession)

        # Look for experiment references
        for exp_ref in root.findall(".//EXPERIMENT_REF"):
            accession = exp_ref.get("accession")
            if accession and (
                accession.startswith("SRX") or accession.startswith("ERX")
            ):
                accessions.append(accession)

        return accessions
    except Exception as e:
        print(f"Error in direct SRA fetch for ID {sra_id}: {str(e)}")
        return []


async def try_sra_summary(
    session: aiohttp.ClientSession,
    sra_id: str,
    base_params: Optional[Dict[str, Any]] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> List[str]:
    """
    Try using esummary to get the SRX/ERX accession.
    Sometimes works when efetch doesn't.

    Args:
        session: The HTTP session to use for requests
        sra_id: The SRA ID to fetch
        base_params: Base parameters to include in all requests
        semaphore: Semaphore to limit concurrent connections

    Returns:
        List of SRX/ERX accessions found
    """
    params = {"db": "sra", "id": sra_id, "retmode": "xml"}

    try:
        xml_text = await fetch_url(
            session, ESUMMARY_BASE_URL, params, base_params, semaphore
        )
        if not xml_text:
            return []
        root = ET.fromstring(xml_text)

        accessions = []

        # Process DocSum elements which contain experiment info
        for doc_sum in root.findall(".//DocSum"):
            # Look for ExpAcc or Experiment field
            for item in doc_sum.findall("./Item"):
                if item.get("Name") == "ExpAcc" and item.text:
                    acc = item.text
                    if acc.startswith("SRX") or acc.startswith("ERX"):
                        accessions.append(acc)

                # Sometimes the experiment info is in ExpXml
                elif item.get("Name") == "ExpXml" and item.text:
                    # Try to extract SRX/ERX from the XML text
                    exp_matches = re.findall(r'accession="(SRX\d+|ERX\d+)"', item.text)
                    accessions.extend(exp_matches)

        return accessions
    except Exception as e:
        print(f"Error in SRA summary for ID {sra_id}: {str(e)}")
        return []


async def get_sra_links(
    session: aiohttp.ClientSession,
    entrez_id: str,
    base_params: Optional[Dict[str, Any]] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    from_db: str = "nucleotide",
) -> List[str]:
    """
    Get SRA links for a given Entrez ID.

    Args:
        session: The HTTP session to use for requests
        entrez_id: The Entrez ID to find links for
        base_params: Base parameters to include in all requests
        semaphore: Semaphore to limit concurrent connections
        from_db: Source database to search from, default is "nucleotide"

    Returns:
        List of SRA links found
    """
    params = {"dbfrom": from_db, "db": "sra", "id": entrez_id}

    xml_text = await fetch_url(session, ELINK_BASE_URL, params, base_params, semaphore)
    if not xml_text:
        return []

    root = ET.fromstring(xml_text)

    sra_links = []
    for linkset in root.findall(".//LinkSetDb"):
        if linkset.find("DbTo") is not None and linkset.find("DbTo").text == "sra":
            for link in linkset.findall("Link/Id"):
                sra_links.append(link.text)

    return sra_links


async def process_id(
    session: aiohttp.ClientSession,
    input_id: str,
    base_params: Optional[Dict[str, Any]] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> Tuple[str, Optional[List[str]]]:
    """
    Process a single ID to get SRX/ERX accessions.

    Args:
        session: The HTTP session to use for requests
        input_id: The ID to process
        base_params: Base parameters to include in all requests
        semaphore: Semaphore to limit concurrent connections

    Returns:
        Tuple of (input_id, accessions) where accessions is a list of SRX/ERX accessions or None if none found
    """
    try:
        # First, try direct SRA fetch (treating it as an SRA internal ID)
        accessions = await direct_sra_fetch(session, input_id, base_params, semaphore)

        if accessions:
            return input_id, accessions

        # If that fails, try esummary
        accessions = await try_sra_summary(session, input_id, base_params, semaphore)

        if accessions:
            return input_id, accessions

        # If direct methods fail, try the original approach (treating as Entrez ID)
        # Try different source databases
        databases = ["sra", "nucleotide", "gene", "bioproject", "biosample"]

        for db in databases:
            sra_links = await get_sra_links(
                session, input_id, base_params, semaphore, db
            )

            if sra_links:
                # For each SRA link, try to get SRX/ERX accessions
                all_accessions = []
                for sra_id in sra_links:
                    direct_results = await direct_sra_fetch(
                        session, sra_id, base_params, semaphore
                    )
                    if direct_results:
                        all_accessions.extend(direct_results)

                if all_accessions:
                    return input_id, all_accessions

        # If all methods fail, return None
        return input_id, None

    except Exception as e:
        print(f"Error processing ID {input_id}: {str(e)}")
        return input_id, None


async def entrez_id_to_srx(sra_id: str, max_concurrent: int = 5) -> List[str]:
    """
    Asynchronously convert an SRA internal ID or Entrez ID to SRX/ERX accessions.

    Args:
        sra_id: Single SRA internal ID or Entrez ID to convert
        max_concurrent: Maximum number of concurrent requests to NCBI

    Returns:
        List of corresponding SRX/ERX accessions.
        If no accessions are found, returns an empty list
    """
    # Common parameters for all requests
    base_params = {"tool": "python_async_script"}

    # Create a semaphore to limit concurrent connections
    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        # Process a single ID and return its accessions directly
        _, accessions = await process_id(session, sra_id, base_params, semaphore)

        # Return unique accessions only
        if accessions:
            return list(
                dict.fromkeys(accessions)
            )  # Preserves order while removing duplicates
        return []


# Example usage
async def main():
    from dotenv import load_dotenv

    load_dotenv(override=True)

    Entrez.email = os.getenv("EMAIL")

    # Test individual IDs
    test_ids = ["30604662", "21712488", "28707082"]

    for test_id in test_ids:
        accessions = await entrez_id_to_srx(test_id)

        if accessions:
            print(f"ID {test_id} corresponds to: {', '.join(accessions)}")
        else:
            print(f"No SRX/ERX accession found for ID {test_id}")


if __name__ == "__main__":
    asyncio.run(main())
