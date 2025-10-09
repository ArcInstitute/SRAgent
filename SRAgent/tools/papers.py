# import
## batteries
from __future__ import annotations
import os
import json
from typing import Annotated

## 3rd party
import requests
from langchain_core.tools import tool


@tool
def get_work_by_doi(
    doi: Annotated[str, "DOI of the paper (e.g., '10.1038/nature12373')"],
    api_key: Annotated[
        str | None, "CORE API key (optional, uses CORE_API_KEY env var if not provided)"
    ] = None,
) -> Annotated[
    str,
    "JSON string with work information including work_id, title, doi, download_url, and full_text_available",
]:
    """
    Search CORE for a paper by DOI and return work information.
    """
    base_url = "https://api.core.ac.uk/v3"

    if api_key is None:
        api_key = os.environ.get("CORE_API_KEY")
        if not api_key:
            return "ERROR: CORE_API_KEY environment variable not set and no api_key provided."

    headers = {"Authorization": f"Bearer {api_key}"}

    # Search using DOI field
    params = {"q": f"doi:{doi}", "limit": 1}

    try:
        response = requests.get(
            f"{base_url}/search/works", headers=headers, params=params
        )
        response.raise_for_status()
    except Exception as e:
        return f"ERROR: Failed to search CORE: {e}"

    data = response.json()

    if data.get("results") and len(data["results"]) > 0:
        work = data["results"][0]
        result = {
            "work_id": work.get("id"),
            "title": work.get("title"),
            "doi": work.get("doi"),
            "download_url": work.get("downloadUrl"),
            "full_text_available": work.get("fullText") is not None,
        }

        return json.dumps(result, indent=2)

    return f"Paper with DOI {doi} not found in CORE"


@tool
def download_paper_by_doi(
    doi: Annotated[str, "DOI of the paper (e.g., '10.1038/nature12373')"],
    output_path: Annotated[str, "Path to save the PDF file"] = "paper.pdf",
    api_key: Annotated[
        str | None, "CORE API key (optional, uses CORE_API_KEY env var if not provided)"
    ] = None,
) -> Annotated[str, "Status message indicating success or failure of the download"]:
    """
    Download a paper from CORE using its DOI.
    """
    # Step 1: Search by DOI to get work_id
    work_info_str = get_work_by_doi.invoke({"doi": doi, "api_key": api_key})

    if work_info_str.startswith("ERROR:") or work_info_str.startswith("Paper with DOI"):
        return work_info_str

    try:
        work_info = json.loads(work_info_str)
    except Exception as e:
        return f"ERROR: Failed to parse work info: {e}"

    download_url = work_info.get("download_url")

    if not download_url:
        return f"No download URL available for {doi}"

    # Step 2: Download the PDF
    try:
        pdf_response = requests.get(download_url)
        pdf_response.raise_for_status()

        # check that output directory exists
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        # write to output path
        with open(output_path, "wb") as f:
            f.write(pdf_response.content)

        return f"Successfully downloaded {doi} to {output_path}"

    except Exception as e:
        return f"ERROR: Failed to download PDF: {e}"


@tool
def download_preprint_by_doi(
    doi: Annotated[
        str,
        "DOI of the preprint (e.g., '10.48550/arXiv.2301.12345' for arXiv, '10.1101/2020.03.15.20030213' for bioRxiv/medRxiv)",
    ],
    output_path: Annotated[str, "Path to save the PDF file"] = "preprint.pdf",
) -> Annotated[str, "Status message indicating success or failure of the download"]:
    """
    Download a preprint from arXiv, bioRxiv, or medRxiv using its DOI.

    This function automatically detects the source (arXiv, bioRxiv, or medRxiv) based on the DOI
    and downloads the PDF from the appropriate repository.

    Supported DOI formats:
    - arXiv: 10.48550/arXiv.{arxiv_id}
    - bioRxiv/medRxiv: 10.1101/{date_code}

    Note: bioRxiv/medRxiv have Cloudflare protection (as of May 2025). This function attempts
    to use cloudscraper if available, otherwise falls back to standard requests.
    """
    # Try to import cloudscraper for Cloudflare bypass
    try:
        import cloudscraper

        use_cloudscraper = True
    except ImportError:
        use_cloudscraper = False

    # Detect source from DOI
    if doi.startswith("10.48550/arXiv.") or doi.startswith("10.48550/arxiv."):
        # arXiv paper - doesn't need cloudscraper
        arxiv_id = doi.replace("10.48550/arXiv.", "").replace("10.48550/arxiv.", "")
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        source = "arXiv"
        use_cloudscraper = False  # arXiv doesn't need it

    elif doi.startswith("10.1101/"):
        # bioRxiv or medRxiv - use API to get metadata first
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
                            return f"ERROR: Paper with DOI {doi} not found in bioRxiv or medRxiv API"
                    else:
                        return f"ERROR: Failed to query medRxiv API (status: {api_response.status_code})"
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
                        return f"ERROR: Paper with DOI {doi} not found in medRxiv API"
                else:
                    return f"ERROR: Failed to query bioRxiv/medRxiv APIs (status: {api_response.status_code})"

        except Exception as e:
            return f"ERROR: Failed to query bioRxiv/medRxiv API: {e}"
    else:
        return f"ERROR: Unsupported DOI format. This tool supports arXiv (10.48550/arXiv.*) and bioRxiv/medRxiv (10.1101/*) DOIs."

    # Download the PDF
    try:
        if use_cloudscraper and source in ["bioRxiv", "medRxiv"]:
            # Use cloudscraper for Cloudflare-protected sites
            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "desktop": True}
            )
            pdf_response = scraper.get(pdf_url, timeout=30)
        else:
            # Use standard requests for arXiv or if cloudscraper not available
            pdf_response = requests.get(pdf_url, timeout=30)

        pdf_response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(pdf_response.content)

        scraper_info = (
            " (using cloudscraper)"
            if use_cloudscraper and source in ["bioRxiv", "medRxiv"]
            else ""
        )
        return f"Successfully downloaded {source} paper {doi} to {output_path}{scraper_info}"

    except Exception as e:
        if not use_cloudscraper and source in ["bioRxiv", "medRxiv"]:
            return f"ERROR: Failed to download PDF from {source}: {e}\n\nNOTE: bioRxiv/medRxiv require the 'cloudscraper' library to bypass Cloudflare protection. Install it with: pip install cloudscraper"
        return f"ERROR: Failed to download PDF from {source}: {e}"


if __name__ == "__main__":
    # python -m SRAgent.tools.papers
    from dotenv import load_dotenv

    load_dotenv()

    doi = "10.1136/BMJOPEN-2023-079350"

    # # get work by doi
    # result = get_work_by_doi.invoke({"doi": doi})
    # print("Work info:")
    # print(result)
    # print()

    # # download paper by doi
    # print("Downloading paper from CORE...")
    # result = download_paper_by_doi.invoke({"doi": doi})
    # print(result)
    # print()

    # # Test preprint downloads
    # print("\nTesting preprint downloads:")

    # # Test arXiv
    # arxiv_doi = "10.48550/arXiv.2301.12345"
    # print(f"\nDownloading arXiv paper {arxiv_doi}...")
    # result = download_preprint_by_doi.invoke(
    #     {"doi": arxiv_doi, "output_path": "tmp/arxiv_paper.pdf"}
    # )
    # print(result)

    # Test bioRxiv
    # biorxiv_doi = "10.1101/2025.08.08.669291"
    biorxiv_doi = "10.1101/2025.02.27.640494"
    print(f"\nDownloading bioRxiv paper {biorxiv_doi}...")
    result = download_preprint_by_doi.invoke(
        {"doi": biorxiv_doi, "output_path": "tmp/biorxiv_paper.pdf"}
    )
    print(result)
