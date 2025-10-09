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
    Create an agent that finds and downloads papers associated with SRA accessions.

    Args:
        model_name: Override model name from settings
        return_tool: If True, return as tool; if False, return agent

    Returns:
        Configured agent instance or tool
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
            " - You are an expert bioinformatician helping to find and download scientific papers",
            " - Your goal is to find publications associated with SRA accessions and download them",
            " - You have access to Entrez tools to query NCBI databases and link records",
            "# Workflow",
            " 1. Convert the SRA accession to an Entrez ID (use esearch agent with sra database)",
            " 2. Find associated PubMed publications (use elink agent: dbfrom=sra, db=pubmed)",
            " 3. Extract DOI from each PubMed record (use efetch agent with rettype=xml, then esummary as fallback)",
            " 4. Report the PubMed IDs and their DOIs",
            "# Strategy",
            " - If esearch doesn't find the accession, the accession may be invalid",
            " - If elink returns no PubMed links, there are no associated publications",
            " - Not all PubMed records have DOIs - report which ones are missing DOIs",
            " - Be thorough: try both efetch (XML) and esummary to find DOIs",
            "# Response Format",
            " - List each PubMed ID with its DOI (or 'No DOI found')",
            " - Be concise and use bullet points",
            " - Example:",
            "   - PMID 12345678: 10.1038/nature12345",
            "   - PMID 23456789: No DOI found",
        ]
    )

    # Create agent
    agent = create_react_agent(model=model, tools=tools, prompt=state_mod)

    # Return agent if not wrapping as tool
    if not return_tool:
        return agent

    # Wrap as tool
    @tool
    async def invoke_papers_agent(
        message: Annotated[str, "Message to the papers agent"],
        config: RunnableConfig,
    ) -> Annotated[dict, "Response from the papers agent"]:
        """
        Invoke the papers agent to find publications and DOIs for SRA accessions.
        """
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=message)]}, config=config
        )
        return {
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="papers_agent")
            ]
        }

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

    # Create sub-agents
    elink_agent = create_elink_agent()
    esearch_agent = create_esearch_agent()
    efetch_agent = create_efetch_agent()
    esummary_agent = create_esummary_agent()

    # Step 1: Find publications
    pubmed_ids = await _find_publications_for_accession(
        accession, elink_agent, esearch_agent, config
    )

    if not pubmed_ids:
        return {
            "accession": accession,
            "pubmed_ids": [],
            "dois": {},
            "downloads": {},
            "summary": "No publications found",
        }

    # Step 2: Extract DOIs
    dois = await _extract_dois_from_pubmed(
        pubmed_ids, efetch_agent, esummary_agent, config
    )

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

    # Step 3: Download papers
    output_dir = os.path.join(output_base_dir, accession)
    downloads = await _download_papers_batch(valid_dois, output_dir, api_key, email)

    # Create summary
    num_success = sum(1 for d in downloads.values() if d["status"] == "success")
    # num_failed = sum(1 for d in downloads.values() if d["status"] == "failed")

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
        accession = "PRJNA831566"  # Replogle 2022
        result = await process_accession(
            accession, output_base_dir="tmp", email=os.getenv("EMAIL1")
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
