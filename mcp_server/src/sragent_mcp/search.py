"""Entrez search helpers for the SRAgent MCP server.

Wraps `Bio.Entrez.esearch` with a flexible query builder. Optional modality
filters add bulk-RNA-seq or scRNA-seq specific narrowing; by default no
modality filter is applied.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Iterable, List, Optional
from urllib.error import HTTPError

from Bio import Entrez

from .normalize import normalize_database, normalize_date, normalize_organisms


# Default search keywords — generic RNA-seq.
DEFAULT_QUERY_TERMS: List[str] = [
    "RNA-seq",
    "RNA sequencing",
    "transcriptome sequencing",
    "mRNA-seq",
]

# Modality-specific NOT-clauses for narrowing.
_MODALITY_FILTERS = {
    "bulk_rna_seq": (
        # Tighten to RNA-seq strategy + exclude common scRNA-seq markers.
        [' AND "rna seq"[Strategy]'],
        [
            '"single cell"[Title]',
            '"single-cell"[Title]',
            '"scRNA-seq"[Title]',
            '"snRNA-seq"[Title]',
            '"10X Genomics"[Title]',
            '"10x Genomics"[All Fields]',
            '"Smart-seq"[All Fields]',
            '"Drop-seq"[All Fields]',
            '"CEL-seq"[All Fields]',
        ],
    ),
    "scrna_seq": (
        [' AND "rna seq"[Strategy]'],
        [],  # no exclusions; intentionally permissive
    ),
}


def _ensure_entrez_credentials() -> None:
    """Set Entrez.email / api_key from env if available. NCBI strongly
    encourages an email but does not strictly require it."""
    if not getattr(Entrez, "email", None):
        Entrez.email = os.getenv("ENTREZ_EMAIL", os.getenv("EMAIL", "anonymous@example.com"))
    if not getattr(Entrez, "api_key", None):
        api_key = os.getenv("NCBI_API_KEY")
        if api_key:
            Entrez.api_key = api_key


def build_query(
    query_terms: Optional[Iterable[str]] = None,
    organisms: Optional[Iterable[str]] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    modality: Optional[str] = None,
    extra_filters: Optional[Iterable[str]] = None,
    extra_exclusions: Optional[Iterable[str]] = None,
) -> str:
    """Build an Entrez `term` string.

    Args:
        query_terms: keywords ORed together. Defaults to `DEFAULT_QUERY_TERMS`.
        organisms: organism names (common or scientific). ANDed via `[Organism]`.
        min_date / max_date: date bounds. Accept various formats; normalized
            to NCBI's `YYYY/MM/DD`. `max_date` defaults to end-of-period for
            partial dates so a year-only `2024` covers the whole year.
        modality: optional filter mode. One of `bulk_rna_seq`, `scrna_seq`,
            or `None` (no modality narrowing).
        extra_filters: additional `AND` clauses appended verbatim.
        extra_exclusions: additional `NOT (...)` clauses ORed together with
            the modality exclusions.
    """
    terms = list(query_terms) if query_terms else DEFAULT_QUERY_TERMS
    quoted = " OR ".join(f'"{t}"' for t in terms)
    q = f"({quoted})"

    if min_date and max_date:
        q += f" AND ({normalize_date(min_date, end_of_period=False)}:" \
             f"{normalize_date(max_date, end_of_period=True)}[PDAT])"

    if organisms:
        sci, _ = normalize_organisms([o for o in organisms if o])
        if sci:
            q += " AND (" + " OR ".join(f'"{n}"[Organism]' for n in sci) + ")"

    # Universal SRA filters for RNA-seq dataset discovery
    q += ' AND "public"[Access]'
    q += ' AND "has data"[Properties]'
    q += ' AND "platform illumina"[Filter]'
    q += ' AND "sra bioproject"[Filter]'

    exclusions: List[str] = []
    if modality:
        if modality not in _MODALITY_FILTERS:
            raise ValueError(
                f"Unknown modality {modality!r}; valid: {sorted(_MODALITY_FILTERS)}"
            )
        adds, excls = _MODALITY_FILTERS[modality]
        for a in adds:
            q += a
        exclusions.extend(excls)

    if extra_filters:
        for f in extra_filters:
            q += f if f.startswith(" AND ") else f" AND {f}"

    if extra_exclusions:
        exclusions.extend(extra_exclusions)

    if exclusions:
        q += " NOT (" + " OR ".join(exclusions) + ")"
    return q


def esearch_ids(
    query_terms: Optional[Iterable[str]] = None,
    organisms: Optional[Iterable[str]] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    max_ids: int = 25,
    database: str = "sra",
    modality: Optional[str] = None,
    extra_exclusions: Optional[Iterable[str]] = None,
    *,
    max_retries: int = 3,
    base_delay: float = 3.0,
    verbose: bool = False,
) -> List[str]:
    """Run an Entrez `esearch` and return up to `max_ids` Entrez IDs.

    Pages through results in batches until `max_ids` is reached or the result
    set is exhausted. Retries on HTTP 429 with exponential backoff.
    """
    _ensure_entrez_credentials()
    database = normalize_database(database)

    term = build_query(
        query_terms=query_terms,
        organisms=organisms,
        min_date=min_date,
        max_date=max_date,
        modality=modality,
        extra_exclusions=extra_exclusions,
    )

    ids: List[str] = []
    seen: set[str] = set()
    retstart = 0
    retmax = min(1000, max(1, max_ids))

    while True:
        for attempt in range(max_retries):
            try:
                handle = Entrez.esearch(
                    db=database, term=term, retstart=retstart, retmax=retmax,
                    sort="pub+date",
                )
                results = Entrez.read(handle)
                handle.close()
                for x in results.get("IdList", []):
                    if x not in seen:
                        seen.add(x); ids.append(x)
                retstart += retmax
                time.sleep(0.34)
                break
            except HTTPError as e:
                if e.code == 429 and attempt < max_retries - 1:
                    time.sleep(base_delay * 2 ** attempt)
                    continue
                if verbose:
                    print(f"esearch error: {e}", file=sys.stderr)
                return ids[:max_ids]
            except Exception as e:
                if verbose:
                    print(f"esearch error: {e}", file=sys.stderr)
                return ids[:max_ids]
        else:
            break

        if len(ids) >= max_ids:
            break
        try:
            total = int(results["Count"])
        except (KeyError, ValueError, NameError):
            break
        if retstart >= total:
            break

    return ids[:max_ids]
