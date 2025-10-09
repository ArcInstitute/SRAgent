# import
## batteries
from __future__ import annotations
import os
import sys
import asyncio
import argparse
from pathlib import Path
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
        "--accession-column",
        type=str,
        default="accession",
        help="Name of the column containing the SRA accessions, if providing a CSV (default: accession)",
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


def _parse_accession_input(
    accession_input: str, accession_column: str
) -> tuple[list[str], pd.DataFrame | None]:
    """
    Parse accession input - either single accession or CSV file.

    Args:
        accession_input: Single accession or path to CSV
        accession_column: Name of the column containing the SRA accessions

    Returns:
        Tuple of (list of accessions, original dataframe if CSV else None)
    """
    # Check if it's a file
    if os.path.isfile(accession_input):
        # Read CSV
        try:
            df = pd.read_csv(accession_input)
            if accession_column not in df.columns:
                print(
                    f"ERROR: CSV must have '{accession_column}' column",
                    file=sys.stderr,
                )
                sys.exit(1)
            accessions = df[accession_column].dropna().astype(str).tolist()
            if len(accessions) < len(df):
                missing = len(df) - len(accessions)
                print(
                    f"WARNING: Skipped {missing} row(s) with missing accession values",
                    file=sys.stderr,
                )
            return accessions, df
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
        return [accession_input], None


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


def _write_results_csv(
    original_df: pd.DataFrame,
    results: list[dict[str, Any]],
    output_dir: str,
    accession_column: str,
    output_filename: str,
) -> Path:
    """
    Merge processing results back into the original dataframe and write to disk.

    Args:
        original_df: Dataframe loaded from the input CSV
        results: Output from processing each accession
        output_dir: Directory where the updated CSV should be written
        accession_column: Name of the accession column to join on
        output_filename: Filename to use for the saved CSV

    Returns:
        Path to the written CSV file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # create dataframe of found papers info
    rows: list[dict[str, Any]] = []
    for result in results:
        accession = result["accession"]
        dois = result.get("dois", {})
        downloads = result.get("downloads", {})

        if not dois:
            rows.append(
                {
                    accession_column: accession,
                    "pubmed_id": None,
                    "doi": None,
                    "download_path": None,
                }
            )
            continue

        for pubmed_id, doi in dois.items():
            download_info = downloads.get(pubmed_id, {})
            rows.append(
                {
                    accession_column: accession,
                    "pubmed_id": pubmed_id,
                    "doi": doi,
                    "download_path": download_info.get("path"),
                }
            )

    results_df = (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(
            columns=[accession_column, "pubmed_id", "doi", "download_path"]
        )
    )

    # merge with original dataframe
    merged_df = original_df.merge(
        results_df,
        how="left",
        left_on=accession_column,
        right_on=accession_column,
    )

    # write to csv
    output_file = output_path / output_filename
    merged_df.to_csv(output_file, index=False)
    return output_file


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
    accessions, input_df = _parse_accession_input(
        args.accession_input, args.accession_column
    )

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

    # If input came from CSV, write updated results
    if input_df is not None:
        output_filename = Path(args.accession_input).name
        updated_csv = _write_results_csv(
            original_df=input_df,
            results=results,
            output_dir=args.output_dir,
            accession_column=args.accession_column,
            output_filename=output_filename,
        )
        print(f"Updated CSV written to: {updated_csv}")


# main
if __name__ == "__main__":
    pass
