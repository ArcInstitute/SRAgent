"""sragent-mcp: MCP server for NCBI SRA / GEO dataset discovery and metadata.

Exposes deterministic Entrez primitives as MCP tools. The host LLM (Claude
Desktop, Claude Code, Cursor, etc.) calls these to find datasets, resolve
accessions, and fetch raw metadata. Structured extraction (e.g. populating
the `RNASeqMetadata` schema from the raw XML) is the host's job — keeps the
MCP fast, deterministic, and free of nested LLM calls.

Tools:
  * find_rnaseq_datasets   — Entrez search with optional modality narrowing
  * entrez_id_to_srx       — Entrez ID → SRX/ERX/DRX accession(s)
  * get_srx_runs           — SRX → SRR/ERR/DRR run accessions
  * get_srx_metadata_text  — raw SRA experiment XML for client-side extraction
  * get_rnaseq_metadata_schema — JSON schema for the metadata extraction target

Transports:
  * stdio (default)         — for Claude Desktop / local MCP clients
  * streamable-http         — for remote MCP / Claude.ai connectors
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from .entrez import (
    entrez_id_to_srx as _entrez_id_to_srx,
    get_srx_metadata_text as _get_srx_metadata_text,
    get_srx_runs as _get_srx_runs,
)
from .schema import RNASeqMetadata, schema_json
from .search import DEFAULT_QUERY_TERMS, build_query, esearch_ids


load_dotenv(override=False)

mcp = FastMCP("sragent")


@mcp.tool()
def find_rnaseq_datasets(
    query_terms: Optional[List[str]] = None,
    organisms: Optional[List[str]] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    max_ids: int = 25,
    database: str = "sra",
    modality: Optional[str] = None,
    extra_exclusions: Optional[List[str]] = None,
) -> dict:
    """Search NCBI SRA / GEO for RNA-seq datasets and return Entrez IDs.

    Args:
        query_terms: keyword list ORed together. Defaults to RNA-seq /
            transcriptome / mRNA-seq.
        organisms: organism names. Common names ('human', 'mouse') and
            scientific names ('Homo sapiens') both work; unrecognized
            inputs pass through.
        min_date / max_date: optional date bounds. Accept YYYY/MM/DD,
            YYYY-MM-DD, YYYY-MM, or just YYYY.
        max_ids: maximum Entrez IDs to return (1-200).
        database: 'sra' (default), 'gds' (GEO), or alias 'geo'.
        modality: optional filter — 'bulk_rna_seq' adds an
            `"rna seq"[Strategy]` filter and excludes single-cell markers;
            'scrna_seq' tightens to RNA-seq strategy without excluding;
            None (default) applies no modality narrowing.
        extra_exclusions: additional NCBI-syntax `NOT` clauses to add.

    Returns the resolved query string + Entrez IDs.
    """
    query = build_query(
        query_terms=query_terms, organisms=organisms,
        min_date=min_date, max_date=max_date,
        modality=modality, extra_exclusions=extra_exclusions,
    )
    ids = esearch_ids(
        query_terms=query_terms, organisms=organisms,
        min_date=min_date, max_date=max_date,
        max_ids=max_ids, database=database,
        modality=modality, extra_exclusions=extra_exclusions,
    )
    return {
        "entrez_ids": ids,
        "count": len(ids),
        "database": database,
        "modality": modality,
        "query": query,
        "default_terms_used": query_terms is None,
    }


@mcp.tool()
def entrez_id_to_srx(entrez_id: str, database: str = "sra") -> dict:
    """Resolve a single Entrez ID to its SRX/ERX/DRX experiment accession(s)."""
    accessions = _entrez_id_to_srx(entrez_id, database=database)
    return {"entrez_id": entrez_id, "database": database, "srx_accessions": accessions}


@mcp.tool()
def get_srx_runs(srx_accession: str) -> dict:
    """List SRR/ERR/DRR run accessions for an SRX/ERX/DRX experiment.
    Accession matching is case-insensitive."""
    runs = _get_srx_runs(srx_accession)
    return {"srx_accession": srx_accession, "srr_accessions": runs, "count": len(runs)}


@mcp.tool()
def get_srx_metadata_text(srx_or_entrez_id: str, database: str = "sra") -> dict:
    """Return raw SRA experiment XML (or GEO esummary blob) as text.

    Intended to be fed to the calling LLM for structured extraction guided
    by `get_rnaseq_metadata_schema`. Truncated at ~200 KB to bound context.
    """
    text = _get_srx_metadata_text(srx_or_entrez_id, database=database)
    truncated = False
    if len(text) > 200_000:
        text = text[:200_000]; truncated = True
    return {"id": srx_or_entrez_id, "database": database, "text": text, "truncated": truncated}


@mcp.tool()
def get_rnaseq_metadata_schema(modality: Optional[str] = None) -> dict:
    """Return the JSON schema for `RNASeqMetadata`.

    Use this schema as the structured-output target when extracting metadata
    from `get_srx_metadata_text` results. Fields cover modality (bulk vs
    scRNA-seq), library strategy/selection/strandedness, instrument, organism,
    tissue/disease/perturbation/cell-line, and SRR run accessions.

    The optional `modality` parameter is informational; the schema is unified.
    """
    return {
        "schema": schema_json(modality),
        "default_search_terms": DEFAULT_QUERY_TERMS,
        "modality": modality,
        "notes": (
            "Set modality_specific fields (e.g. scrna_lib_prep, cell_prep) "
            "to 'not_applicable' when the dataset is bulk RNA-seq. Use 'unsure' "
            "rather than guessing when evidence is ambiguous."
        ),
    }


@mcp.resource("schema://rnaseq-metadata")
def _schema_resource() -> str:
    """RNA-seq metadata JSON schema as a resource."""
    return json.dumps(schema_json(), indent=2)


def main() -> None:
    """Console-script entry point."""
    parser = argparse.ArgumentParser(
        prog="sragent-mcp",
        description="MCP server for NCBI SRA / GEO dataset discovery.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default=os.environ.get("MCP_TRANSPORT", "stdio"),
        help="Transport to use (default: stdio).",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MCP_HOST", "127.0.0.1"),
        help="Host to bind for HTTP transports (use 0.0.0.0 to accept "
             "external connections through a tunnel).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MCP_PORT", "8000")),
        help="Port to bind for HTTP transports (default: 8000).",
    )
    args = parser.parse_args()

    if args.transport in ("streamable-http", "sse"):
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        endpoint = (
            mcp.settings.streamable_http_path
            if args.transport == "streamable-http"
            else mcp.settings.sse_path
        )
        print(f"sragent-mcp listening on http://{args.host}:{args.port}{endpoint}",
              file=sys.stderr)

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
