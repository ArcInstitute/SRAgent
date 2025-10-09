# import
## batteries
from __future__ import annotations
import os
import sys
import asyncio
import argparse
from typing import Any

## 3rd party
import pandas as pd
from Bio import Entrez
from rich.console import Console
from rich.table import Table

## package
from SRAgent.cli.utils import CustomFormatter
from SRAgent.agents.papers import process_accession
from SRAgent.tools.utils import set_entrez_access
from SRAgent.workflows.graph_utils import handle_write_graph_option


# functions
def papers_parser(subparsers) -> None:
    """
    Create parser for the papers agent CLI command.
    """
    help = "Papers Agent: Find and download manuscripts associated with SRA accessions"
    desc = """
    The Papers Agent finds scientific publications associated with SRA accessions
    (experiments or studies) and downloads the full-text manuscripts.
    
    Workflow:
      1. Links SRA accession → PubMed publications (via NCBI Entrez elink)
      2. Extracts DOI for each publication (via efetch and esummary)
      3. Downloads papers using multiple sources (CORE, Europe PMC, Unpaywall, preprint servers)
    
    Input formats:
      - Single accession: SRX4967527 or SRP012345
      - CSV file: Must have 'accession' column (header required)
    
    Examples:
      # SRA experiments
        SRAgent papers SRX4967527
      # SRA projects
        SRAgent papers PRJNA615032
      # Multiple accessions
        SRAgent papers accessions.csv
      # Set output directory
        SRAgent papers SRX4967527 --output-dir my_papers
    """
    sub_parser = subparsers.add_parser(
        "papers", help=help, description=desc, formatter_class=CustomFormatter
    )
    sub_parser.set_defaults(func=papers_main)

    # Required arguments
    sub_parser.add_argument(
        "accession_input",
        type=str,
        help='Single SRA accession (SRX*/SRP*) or path to CSV file with "accession" column',
    )

    # Optional arguments
    sub_parser.add_argument(
        "--output-dir",
        type=str,
        default="papers",
        help="Base directory for saving papers (default: papers/)",
    )
    sub_parser.add_argument(
        "--core-api-key",
        type=str,
        default=None,
        help="CORE API key (optional, uses CORE_API_KEY env var if not provided)",
    )
    sub_parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="Email for Unpaywall API (optional, uses EMAIL env var if not provided)",
    )
    sub_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent accession processing tasks (default: 5)",
    )
    sub_parser.add_argument(
        "--recursion-limit",
        type=int,
        default=100,
        help="Maximum recursion limit (default: 100)",
    )
    sub_parser.add_argument(
        "--write-graph",
        type=str,
        metavar="FILE",
        default=None,
        help="Write the workflow graph to a file and exit (supports .png, .svg, .pdf, .mermaid formats)",
    )


def _parse_accession_input(accession_input: str) -> list[str]:
    """
    Parse accession input - either single accession or CSV file.

    Args:
        accession_input: Single accession or path to CSV

    Returns:
        List of accessions
    """
    # Check if it's a file
    if os.path.isfile(accession_input):
        # Read CSV
        try:
            df = pd.read_csv(accession_input)
            df.columns = df.columns.str.lower()
            if "accession" not in df.columns:
                print("ERROR: CSV must have 'accession' column", file=sys.stderr)
                sys.exit(1)
            return df["accession"].tolist()
        except Exception as e:
            print(f"ERROR: Failed to read CSV: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Single accession
        # Validate format (SRX or SRP)
        if not (
            accession_input.startswith("SRX")
            or accession_input.startswith("SRP")
            or accession_input.startswith("ERX")
            or accession_input.startswith("ERP")
        ):
            print(
                f"WARNING: Accession '{accession_input}' doesn't match expected format (SRX*, SRP*, ERX*, ERP*)",
                file=sys.stderr,
            )
        return [accession_input]


async def _process_accessions_batch(
    accessions: list[str],
    output_dir: str,
    core_api_key: str | None,
    email: str | None,
    max_concurrency: int,
    recursion_limit: int,
) -> list[dict[str, Any]]:
    """
    Process multiple accessions with concurrency limit.

    Args:
        accessions: List of SRA accessions
        output_dir: Base output directory
        core_api_key: CORE API key
        email: Email for Unpaywall
        max_concurrency: Max concurrent tasks
        recursion_limit: Recursion limit

    Returns:
        List of results for each accession
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_with_semaphore(accession: str) -> dict[str, Any]:
        async with semaphore:
            config = {"recursion_limit": recursion_limit}
            return await process_accession(
                accession,
                output_base_dir=output_dir,
                api_key=core_api_key,
                email=email,
                config=config,
            )

    # Process all accessions concurrently (with limit)
    tasks = [process_with_semaphore(acc) for acc in accessions]
    results = await asyncio.gather(*tasks)

    return results


def _display_results_table(results: list[dict[str, Any]]) -> None:
    """
    Display results in a formatted table.

    Args:
        results: List of results from process_accession
    """
    console = Console()

    # Create summary table
    table = Table(
        title="📄 Papers Download Summary",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Accession", style="cyan")
    table.add_column("Publications", justify="right", style="green")
    table.add_column("DOIs", justify="right", style="yellow")
    table.add_column("Downloaded", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")

    for result in results:
        num_pubs = len(result["pubmed_ids"])
        num_dois = sum(1 for doi in result["dois"].values() if doi is not None)
        num_downloaded = sum(
            1 for d in result["downloads"].values() if d["status"] == "success"
        )
        num_failed = sum(
            1
            for d in result["downloads"].values()
            if d["status"] == "failed"  # type: ignore
        )

        table.add_row(
            result["accession"],
            str(num_pubs),
            str(num_dois),
            str(num_downloaded),
            str(num_failed),
        )

    console.print(table)

    # Print detailed results
    console.print("\n[bold]Detailed Results:[/bold]\n")
    for result in results:
        console.print(f"[cyan]{result['accession']}[/cyan]: {result['summary']}")

        # Show downloaded papers
        if result["downloads"]:
            console.print("  Downloaded papers:")
            for pmid, info in result["downloads"].items():
                if info["status"] == "success":
                    console.print(f"    ✓ PMID {pmid} ({info['doi']}): {info['path']}")
                elif info["status"] == "failed":
                    console.print(
                        f"    ✗ PMID {pmid} ({info['doi']}): {info['error'][:100]}"
                    )
                else:
                    console.print(f"    - PMID {pmid}: {info['error']}")
        console.print()


def papers_main(args: argparse.Namespace) -> None:
    """
    Main function for the papers agent CLI.

    Args:
        args: Parsed command-line arguments
    """
    # Set Entrez credentials
    set_entrez_access()

    # Handle write-graph option
    if args.write_graph:
        from SRAgent.agents.papers import create_papers_agent

        handle_write_graph_option(create_papers_agent, args.write_graph)
        return

    # Parse accession input
    accessions = _parse_accession_input(args.accession_input)

    print(f"Processing {len(accessions)} accession(s)...")
    print(f"Output directory: {args.output_dir}")
    print()

    # Process accessions
    results = asyncio.run(
        _process_accessions_batch(
            accessions=accessions,
            output_dir=args.output_dir,
            core_api_key=args.core_api_key,
            email=args.email,
            max_concurrency=args.max_concurrency,
            recursion_limit=args.recursion_limit,
        )
    )

    # Display results
    _display_results_table(results)


# main
if __name__ == "__main__":
    pass
