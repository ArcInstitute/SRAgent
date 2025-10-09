# import
## batteries
from __future__ import annotations
import os
import json
from typing import Annotated

## 3rd party
import requests
from langchain_core.tools import tool


# ============================================================================
# Helper Functions (not langchain tools)
# ============================================================================


def _get_core_info(doi: str) -> dict | None:
    """
    Get paper information from CORE API.

    Args:
        doi: DOI of the paper

    Returns:
        Dictionary with work info or None if not found/error
    """
    base_url = "https://api.core.ac.uk/v3"

    try:
        api_key = os.environ["CORE_API_KEY"]
    except KeyError:
        return "Error: CORE_API_KEY is not set"

    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"q": f"doi:{doi}", "limit": 1}

    try:
        response = requests.get(
            f"{base_url}/search/works", headers=headers, params=params, timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if data.get("results") and len(data["results"]) > 0:
            work = data["results"][0]
            return {
                "work_id": work.get("id"),
                "title": work.get("title"),
                "doi": work.get("doi"),
                "download_url": work.get("downloadUrl"),
                "full_text_available": work.get("fullText") is not None,
            }
    except Exception:
        pass

    return None


def _get_unpaywall_info(doi: str, email: str | None = None) -> dict | None:
    """
    Get paper information from Unpaywall API.

    Args:
        doi: DOI of the paper
        email: Email for Unpaywall API (optional, uses EMAIL env var if not provided)

    Returns:
        Dictionary with OA location info or None if not found/error
    """
    if email is None:
        email = os.environ.get("EMAIL")
        if not email:
            return None

    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("is_oa") and data.get("best_oa_location"):
            return {
                "is_oa": True,
                "pdf_url": data["best_oa_location"].get("url_for_pdf"),
                "version": data["best_oa_location"].get("version"),
                "host_type": data["best_oa_location"].get("host_type"),
            }
    except Exception:
        pass

    return None


def _download_from_preprint_server(doi: str, output_path: str) -> dict:
    """
    Download a preprint from arXiv, bioRxiv, or medRxiv.

    Args:
        doi: DOI of the preprint
        output_path: Path to save the PDF

    Returns:
        Dictionary with success status and message
    """
    # Try to import cloudscraper for Cloudflare bypass
    try:
        import cloudscraper

        use_cloudscraper = True
    except ImportError:
        use_cloudscraper = False

    # Detect source from DOI
    if doi.startswith("10.48550/arXiv.") or doi.startswith("10.48550/arxiv."):
        # arXiv paper
        arxiv_id = doi.replace("10.48550/arXiv.", "").replace("10.48550/arxiv.", "")
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        source = "arXiv"
        use_cloudscraper = False

    elif doi.startswith("10.1101/"):
        # bioRxiv or medRxiv - use API to get metadata
        try:
            # Try bioRxiv API first
            api_url = f"https://api.biorxiv.org/details/biorxiv/{doi}/na/json"
            api_response = requests.get(api_url, timeout=10)

            if api_response.status_code == 200:
                data = api_response.json()
                if data.get("collection") and len(data["collection"]) > 0:
                    paper = data["collection"][0]
                    version = paper.get("version", "1")
                    pdf_url = (
                        f"https://www.biorxiv.org/content/{doi}v{version}.full.pdf"
                    )
                    source = "bioRxiv"
                else:
                    # Try medRxiv API
                    api_url = f"https://api.biorxiv.org/details/medrxiv/{doi}/na/json"
                    api_response = requests.get(api_url, timeout=10)

                    if api_response.status_code == 200:
                        data = api_response.json()
                        if data.get("collection") and len(data["collection"]) > 0:
                            paper = data["collection"][0]
                            version = paper.get("version", "1")
                            pdf_url = f"https://www.medrxiv.org/content/{doi}v{version}.full.pdf"
                            source = "medRxiv"
                        else:
                            return {
                                "success": False,
                                "message": "Not found in bioRxiv or medRxiv API",
                            }
                    else:
                        return {
                            "success": False,
                            "message": f"medRxiv API returned status {api_response.status_code}",
                        }
            else:
                # Try medRxiv API
                api_url = f"https://api.biorxiv.org/details/medrxiv/{doi}/na/json"
                api_response = requests.get(api_url, timeout=10)

                if api_response.status_code == 200:
                    data = api_response.json()
                    if data.get("collection") and len(data["collection"]) > 0:
                        paper = data["collection"][0]
                        version = paper.get("version", "1")
                        pdf_url = (
                            f"https://www.medrxiv.org/content/{doi}v{version}.full.pdf"
                        )
                        source = "medRxiv"
                    else:
                        return {"success": False, "message": "Not found in medRxiv API"}
                else:
                    return {
                        "success": False,
                        "message": f"bioRxiv/medRxiv APIs returned status {api_response.status_code}",
                    }

        except Exception as e:
            return {"success": False, "message": f"API query failed: {e}"}
    else:
        return {"success": False, "message": "Not a recognized preprint DOI format"}

    # Download the PDF
    try:
        if use_cloudscraper and source in ["bioRxiv", "medRxiv"]:
            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "desktop": True}
            )
            pdf_response = scraper.get(pdf_url, timeout=30)
        else:
            pdf_response = requests.get(pdf_url, timeout=30)

        pdf_response.raise_for_status()

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, "wb") as f:
            f.write(pdf_response.content)

        return {
            "success": True,
            "message": f"Downloaded from {source}",
            "source": source,
        }

    except Exception as e:
        if not use_cloudscraper and source in ["bioRxiv", "medRxiv"]:
            return {
                "success": False,
                "message": f"Download failed: {e}. NOTE: Install cloudscraper for bioRxiv/medRxiv: pip install cloudscraper",
            }
        return {"success": False, "message": f"Download failed: {e}"}


# ============================================================================
# Langchain Tool
# ============================================================================


@tool
def download_paper_by_doi(
    doi: Annotated[
        str,
        "DOI of the paper (e.g., '10.1038/nature12373' or '10.48550/arXiv.2301.12345')",
    ],
    output_path: Annotated[
        str | None, "Path to save the PDF file. If None, the DOI is used"
    ] = None,
    email: Annotated[
        str | None,
        "Email for Unpaywall API (optional, uses EMAIL env var if not provided)",
    ] = None,
) -> Annotated[str, "Status message indicating success or failure of the download"]:
    """
    Download a paper or preprint using its DOI.

    This function tries multiple sources in order:
    1. For preprint DOIs (arXiv, bioRxiv, medRxiv): Try preprint server first
    2. Try CORE API
    3. Try Unpaywall API

    Supported DOI formats:
    - arXiv: 10.48550/arXiv.{arxiv_id}
    - bioRxiv/medRxiv: 10.1101/{date_code}
    - Any other DOI format: tries CORE and Unpaywall

    Args:
        doi: DOI of the paper/preprint
        output_path: Path to save the PDF file. If None, the DOI is used
        email: Email for Unpaywall API (optional, uses EMAIL env var)

    Returns:
        Success message or detailed error message listing all sources attempted
    """
    errors = []

    # set output path
    if output_path is None or output_path == "":
        output_path = f"{doi.replace('/', '_')}.pdf"
    # ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 1: Try preprint server if it's a preprint DOI
    is_preprint = (
        doi.startswith("10.48550/arXiv.")
        or doi.startswith("10.48550/arxiv.")
        or doi.startswith("10.1101/")
    )

    if is_preprint:
        result = _download_from_preprint_server(doi, output_path)
        if result["success"]:
            return f"Successfully downloaded from {result['source']} to {output_path}"
        errors.append(f"Preprint server: {result['message']}")

    # Step 2: Try CORE
    core_info = _get_core_info(doi)
    if core_info and core_info.get("download_url"):
        try:
            pdf_response = requests.get(core_info["download_url"], timeout=30)
            pdf_response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(pdf_response.content)

            return f"Successfully downloaded from CORE to {output_path}"
        except Exception as e:
            errors.append(f"CORE: Download failed - {e}")
    else:
        if os.environ.get("CORE_API_KEY"):
            errors.append("CORE: No download URL available")
        else:
            errors.append("CORE: Skipped (no API key)")

    # Step 3: Try Unpaywall
    unpaywall_info = _get_unpaywall_info(doi, email)
    if unpaywall_info and unpaywall_info.get("pdf_url"):
        try:
            pdf_response = requests.get(unpaywall_info["pdf_url"], timeout=30)
            pdf_response.raise_for_status()

            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_path, "wb") as f:
                f.write(pdf_response.content)

            version_info = f" ({unpaywall_info.get('version', 'unknown version')})"
            return (
                f"Successfully downloaded from Unpaywall{version_info} to {output_path}"
            )
        except Exception as e:
            errors.append(f"Unpaywall: Download failed - {e}")
    else:
        if email or os.environ.get("EMAIL"):
            errors.append("Unpaywall: No open access PDF available")
        else:
            errors.append("Unpaywall: Skipped (no email configured)")

    # All sources failed
    error_summary = "\n".join([f"  - {err}" for err in errors])
    return f"ERROR: Failed to download {doi} from all sources:\n{error_summary}"


if __name__ == "__main__":
    # python -m SRAgent.tools.papers
    from dotenv import load_dotenv

    load_dotenv()

    # Test regular paper (should try CORE then Unpaywall)
    print("=" * 60)
    print("Test 1: Regular paper DOI")
    print("=" * 60)
    doi = "10.1136/BMJOPEN-2023-079350"
    result = download_paper_by_doi.invoke({"doi": doi, "output_path": "tmp/paper1.pdf"})
    print(result)
    print()

    # Test arXiv (should download from arXiv directly)
    print("=" * 60)
    print("Test 2: arXiv preprint")
    print("=" * 60)
    arxiv_doi = "10.48550/arXiv.2301.12345"
    result = download_paper_by_doi.invoke(
        {"doi": arxiv_doi, "output_path": "tmp/arxiv_paper.pdf"}
    )
    print(result)
    print()

    # Test bioRxiv (should try bioRxiv, then CORE, then Unpaywall)
    print("=" * 60)
    print("Test 3: bioRxiv preprint")
    print("=" * 60)
    biorxiv_doi = "10.1101/2025.02.27.640494"
    result = download_paper_by_doi.invoke(
        {"doi": biorxiv_doi, "output_path": "tmp/biorxiv_paper.pdf"}
    )
    print(result)
    print()

    # Test Unpaywall (use a DOI known to be in Unpaywall)
    print("=" * 60)
    print("Test 4: Paper from Unpaywall")
    print("=" * 60)
    unpaywall_doi = "10.1371/journal.pone.0000308"
    result = download_paper_by_doi.invoke(
        {"doi": unpaywall_doi, "output_path": "tmp/unpaywall_paper.pdf"}
    )
    print(result)
