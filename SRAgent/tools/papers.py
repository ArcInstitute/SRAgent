# import
## batteries
from __future__ import annotations
import os
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
        import json

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

    # Parse the JSON response
    import json

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

        with open(output_path, "wb") as f:
            f.write(pdf_response.content)

        return f"Successfully downloaded {doi} to {output_path}"

    except Exception as e:
        return f"ERROR: Failed to download PDF: {e}"


if __name__ == "__main__":
    # python -m SRAgent.tools.papers
    from dotenv import load_dotenv

    load_dotenv()

    doi = "10.1136/BMJOPEN-2023-079350"

    # get work by doi
    result = get_work_by_doi.invoke({"doi": doi})
    print("Work info:")
    print(result)
    print()

    # download paper by doi
    print("Downloading paper...")
    result = download_paper_by_doi.invoke({"doi": doi})
    print(result)
