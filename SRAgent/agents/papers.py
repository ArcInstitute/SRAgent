# import
## batteries
from __future__ import annotations
import os
import re
import asyncio
from pathlib import Path
from typing import Annotated, Any, Callable

## 3rd party
from Bio import Entrez
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.prebuilt import create_react_agent

## package
from SRAgent.agents.utils import set_model
from SRAgent.agents.esearch import create_esearch_agent
from SRAgent.agents.esummary import create_esummary_agent
from SRAgent.agents.efetch import create_efetch_agent
from SRAgent.agents.elink import create_elink_agent
from SRAgent.tools.papers import download_paper_by_doi


# ============================================================================
# Structured Output Models
# ============================================================================


class PublicationDOI(BaseModel):
    """A publication with its PubMed ID and DOI."""

    pubmed_id: str = Field(description="PubMed ID (PMID)")
    doi: str | None = Field(description="DOI of the publication, or None if not found")


class PublicationsResult(BaseModel):
    """Result of finding publications for an SRA accession."""

    accession: str = Field(description="SRA accession that was queried")
    publications: list[PublicationDOI] = Field(
        description="List of publications with their PubMed IDs and DOIs"
    )


# ============================================================================
# Helper Functions
# ============================================================================


async def _find_publications_for_accession(
    accession: str,
    elink_agent: Callable,
    esearch_agent: Callable,
    config: RunnableConfig,
) -> list[str]:
    """
    Find PubMed publications associated with an SRA accession.

    Args:
        accession: SRA accession (SRX or SRP)
        elink_agent: elink agent tool
        esearch_agent: esearch agent tool
        config: Runnable config

    Returns:
        List of PubMed Entrez IDs
    """
    # Step 1: Convert accession to Entrez ID if needed
    entrez_id = None

    # Try to find the Entrez ID for the accession
    message = f"Find the Entrez ID for {accession} in the SRA database"
    result = await esearch_agent.ainvoke({"message": message}, config=config)

    # Extract Entrez ID from response
    content = result["messages"][-1].content
    id_match = re.search(r"\b\d{6,}\b", content)
    if id_match:
        entrez_id = id_match.group()

    if not entrez_id:
        return []

    # Step 2: Use elink to find associated PubMed IDs
    message = f"Use elink to find PubMed publications associated with SRA Entrez ID {entrez_id}"
    result = await elink_agent.ainvoke({"message": message}, config=config)

    # Extract PubMed IDs from response
    content = result["messages"][-1].content
    pubmed_ids = re.findall(r"\b\d{7,8}\b", content)

    return list(set(pubmed_ids))  # Remove duplicates


async def _extract_dois_from_pubmed(
    pubmed_ids: list[str],
    efetch_agent: Callable,
    esummary_agent: Callable,
    config: RunnableConfig,
) -> dict[str, str | None]:
    """
    Extract DOIs from PubMed Entrez IDs.

    Args:
        pubmed_ids: List of PubMed Entrez IDs
        efetch_agent: efetch agent tool
        esummary_agent: esummary agent tool
        config: Runnable config

    Returns:
        Dictionary mapping {pubmed_id: doi} (doi is None if not found)
    """
    dois: dict[str, str | None] = {}

    for pubmed_id in pubmed_ids:
        doi = None

        # Primary approach: Use efetch to get full record with DOI
        try:
            message = f"Use efetch to retrieve the DOI for PubMed ID {pubmed_id} from the pubmed database in XML format"
            result = await efetch_agent.ainvoke({"message": message}, config=config)
            content = result["messages"][-1].content

            # Extract DOI from response - look for DOI pattern
            doi_match = re.search(r"10\.\d{4,}/[^\s\]]+", content)
            if doi_match:
                doi = doi_match.group().rstrip(".,;)")
        except Exception:
            pass

        # Fallback: Use esummary
        if not doi:
            try:
                message = f"Use esummary to get the DOI for PubMed ID {pubmed_id}"
                result = await esummary_agent.ainvoke(
                    {"message": message}, config=config
                )
                content = result["messages"][-1].content

                # Extract DOI from response
                doi_match = re.search(r"10\.\d{4,}/[^\s\]]+", content)
                if doi_match:
                    doi = doi_match.group().rstrip(".,;)")
            except Exception:
                pass

        dois[pubmed_id] = doi

    return dois


async def _download_papers_batch(
    dois: dict[str, str | None],
    output_dir: str,
    api_key: str | None = None,
    email: str | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Download papers for a batch of DOIs.

    Args:
        dois: Dictionary mapping {pubmed_id: doi}
        output_dir: Base directory to save papers
        api_key: CORE API key (optional)
        email: Email for Unpaywall (optional)

    Returns:
        Dictionary with download status for each DOI
    """
    results: dict[str, dict[str, Any]] = {}

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for pubmed_id, doi in dois.items():
        if not doi:
            results[pubmed_id] = {
                "status": "skipped",
                "doi": None,
                "path": None,
                "error": "No DOI found",
            }
            continue

        # Sanitize DOI for filename (replace / with _)
        safe_doi = doi.replace("/", "_")
        output_path = os.path.join(output_dir, f"{safe_doi}.pdf")

        # Download the paper
        try:
            result_msg = download_paper_by_doi(
                doi=doi,
                output_path=output_path,
                api_key=api_key,
                email=email,
            )

            if result_msg.startswith("Successfully"):
                results[pubmed_id] = {
                    "status": "success",
                    "doi": doi,
                    "path": output_path,
                    "error": None,
                }
            else:
                results[pubmed_id] = {
                    "status": "failed",
                    "doi": doi,
                    "path": None,
                    "error": result_msg,
                }
        except Exception as e:
            results[pubmed_id] = {
                "status": "failed",
                "doi": doi,
                "path": None,
                "error": str(e),
            }

    return results


# ============================================================================
# Main Agent
# ============================================================================


def create_papers_agent(
    model_name: str | None = None,
    return_tool: bool = True,
) -> Callable:
    """
    Create an agent that finds DOIs for publications associated with SRA accessions.

    Args:
        model_name: Override model name from settings
        return_tool: If True, return as tool; if False, return agent

    Returns:
        Configured agent instance or tool that returns a list of DOIs
    """
    # Create model
    model = set_model(model_name=model_name, agent_name="papers")

    # Set tools - sub-agents for querying NCBI
    tools = [
        create_esearch_agent(),
        create_esummary_agent(),
        create_efetch_agent(),
        create_elink_agent(),
    ]

    # State modifier
    state_mod = "\n".join(
        [
            "# Role and Purpose",
            " - You are an expert bioinformatician helping to find publications associated with SRA accessions",
            " - Your goal is to find PubMed publications and their DOIs for a given SRA accession",
            " - You have access to four Entrez sub-agents: esearch, elink, efetch, and esummary",
            "# Sub-Agent Capabilities",
            " - esearch_agent: Searches Entrez databases (e.g., sra, gds, pubmed) to find Entrez IDs from accessions or search terms",
            "   * Use to convert SRA accessions to Entrez IDs",
            "   * Example: 'Find the Entrez ID for SRX4967527 in the sra database'",
            " - elink_agent: Links related entries between Entrez databases using Entrez IDs",
            "   * Use to find PubMed publications linked to SRA Entrez IDs",
            "   * Example: 'Use elink with dbfrom=sra, db=pubmed, and id=<entrez_id>'",
            "   * Returns Entrez IDs (not accessions); requires Entrez IDs as input",
            " - efetch_agent: Fetches full records from Entrez databases",
            "   * Best for extracting DOIs from PubMed records (use rettype=xml for detailed records)",
            "   * Example: 'Use efetch to retrieve the DOI for PubMed ID <pmid> from the pubmed database in XML format'",
            "   * If unsure of database, can use which_entrez_databases tool",
            " - esummary_agent: Fetches summaries from Entrez databases",
            "   * Use as fallback if efetch doesn't return DOI",
            "   * Example: 'Use esummary to get the DOI for PubMed ID <pmid>'",
            "   * If unsure of database, can use which_entrez_databases tool",
            "# Workflow",
            " 1. Convert the SRA accession to an Entrez ID using esearch_agent (database=sra)",
            " 2. Find associated PubMed Entrez IDs using elink_agent (dbfrom=sra, db=pubmed, id=<sra_entrez_id>)",
            " 3. For each PubMed Entrez ID:",
            "    a. Try efetch_agent first (database=pubmed, rettype=xml) to extract DOI from full record",
            "    b. If DOI not found, try esummary_agent as fallback",
            " 4. Return structured result with accession and list of publications (PubMed ID + DOI)",
            "# Strategy",
            " - If esearch doesn't find the accession, return empty publications list",
            " - If elink returns no PubMed links, return empty publications list",
            " - Not all PubMed records have DOIs - set doi=None for those without DOIs",
            " - Be thorough: always try efetch (XML format) first, then esummary as fallback for DOI extraction",
        ]
    )

    # Create agent with response_format for structured output
    agent = create_react_agent(
        model=model, tools=tools, prompt=state_mod, response_format=PublicationsResult
    )

    # Return agent if not wrapping as tool
    if not return_tool:
        return agent

    # Wrap as tool
    @tool
    async def invoke_papers_agent(
        accession: Annotated[str, "SRA accession to find publications for"],
        config: RunnableConfig,
    ) -> Annotated[list[str], "List of DOIs found"]:
        """
        Find DOIs for publications associated with an SRA accession.
        Returns a list of DOI strings.
        """
        message = f"Find all publications and their DOIs for SRA accession {accession}"
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=message)]}, config=config
        )

        # The structured response is in the 'structured_response' key
        if "structured_response" in result:
            pubs_result = result["structured_response"]
            if isinstance(pubs_result, PublicationsResult):
                # Extract DOIs (skip None values)
                return [
                    pub.doi for pub in pubs_result.publications if pub.doi is not None
                ]
            elif isinstance(pubs_result, dict):
                # Handle dict format
                publications = pubs_result.get("publications", [])
                dois = []
                for pub in publications:
                    if isinstance(pub, dict):
                        doi = pub.get("doi")
                        if doi:
                            dois.append(doi)
                    elif hasattr(pub, "doi") and pub.doi:
                        dois.append(pub.doi)
                return dois

        # Fallback: return empty list
        return []

    return invoke_papers_agent


# ============================================================================
# Main execution function for CLI
# ============================================================================


async def process_accession(
    accession: str,
    output_base_dir: str = "papers",
    api_key: str | None = None,
    email: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Process a single SRA accession: find publications, extract DOIs, and download papers.

    Args:
        accession: SRA accession (SRX or SRP)
        output_base_dir: Base directory for saving papers
        api_key: CORE API key (optional)
        email: Email for Unpaywall (optional)
        config: LangChain config

    Returns:
        Dictionary with processing results
    """
    if config is None:
        config = {}

    # Step 1: Use papers agent to find DOIs
    papers_agent = create_papers_agent(return_tool=False)
    message = f"Find all publications and their DOIs for SRA accession {accession}"
    result = await papers_agent.ainvoke(
        {"messages": [HumanMessage(content=message)]}, config=config
    )

    # Extract publications from structured response
    pubmed_ids = []
    dois = {}
    if "structured_response" in result:
        pubs_result = result["structured_response"]
        if isinstance(pubs_result, PublicationsResult):
            # Extract PubMed IDs and DOIs
            pubmed_ids = [pub.pubmed_id for pub in pubs_result.publications]
            dois = {pub.pubmed_id: pub.doi for pub in pubs_result.publications}
        elif isinstance(pubs_result, dict):
            # Handle dict format
            publications = pubs_result.get("publications", [])
            for pub in publications:
                if isinstance(pub, dict):
                    pmid = pub.get("pubmed_id")
                    doi = pub.get("doi")
                    if pmid:
                        pubmed_ids.append(pmid)
                        dois[pmid] = doi
                elif hasattr(pub, "pubmed_id"):
                    pubmed_ids.append(pub.pubmed_id)
                    dois[pub.pubmed_id] = pub.doi

    if not pubmed_ids:
        return {
            "accession": accession,
            "pubmed_ids": [],
            "dois": {},
            "downloads": {},
            "summary": "No publications found",
        }

    # Filter out None DOIs for downloading
    valid_dois = {k: v for k, v in dois.items() if v is not None}

    if not valid_dois:
        return {
            "accession": accession,
            "pubmed_ids": pubmed_ids,
            "dois": dois,
            "downloads": {},
            "summary": f"Found {len(pubmed_ids)} publication(s) but no DOIs available",
        }

    # Step 2: Download papers
    output_dir = os.path.join(output_base_dir, accession)
    downloads = await _download_papers_batch(valid_dois, output_dir, api_key, email)

    # Create summary
    num_success = sum(1 for d in downloads.values() if d["status"] == "success")

    summary = f"Found {len(pubmed_ids)} publication(s), {len(valid_dois)} DOI(s), downloaded {num_success}/{len(valid_dois)}"

    return {
        "accession": accession,
        "pubmed_ids": pubmed_ids,
        "dois": dois,
        "downloads": downloads,
        "summary": summary,
    }


if __name__ == "__main__":
    # python -m SRAgent.agents.papers
    from dotenv import load_dotenv

    load_dotenv()

    Entrez.email = os.getenv("EMAIL1")
    Entrez.api_key = os.getenv("NCBI_API_KEY1")

    async def main():
        # Test with a known SRX accession
        # accession = "SRX4967527"
        # accession = "PRJNA831566"  # Replogle 2022
        accession = "PRJNA615032"  # SARS-CoV-2 Transcriptomics
        result = await process_accession(
            accession, output_base_dir="papers", email=os.getenv("EMAIL1")
        )

        print(f"\nResults for {accession}:")
        print(f"  PubMed IDs: {result['pubmed_ids']}")
        print(f"  DOIs: {result['dois']}")
        print(f"  Summary: {result['summary']}")
        print(f"\nDownload results:")
        for pmid, info in result["downloads"].items():
            print(
                f"  PMID {pmid}: {info['status']} - {info.get('path') or info.get('error')}"
            )

    asyncio.run(main())
