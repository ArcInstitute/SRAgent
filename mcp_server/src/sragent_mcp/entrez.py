"""Entrez fetch helpers.

Deterministic, no LLM in the loop. Functions return plain dicts/lists/strings
so they marshal cleanly through MCP.
"""
from __future__ import annotations

import re
import time
from typing import List
from urllib.error import HTTPError

from Bio import Entrez

from .normalize import (
    is_srx_accession,
    normalize_accession,
    normalize_database,
    normalize_entrez_id,
)
from .search import _ensure_entrez_credentials


# Body-scan regexes — case-insensitive so we pick up accessions whether NCBI
# uses upper or mixed case in surrounding text.
SRX_RE = re.compile(r"\b(SRX|ERX|DRX)\d{4,}\b", re.IGNORECASE)
SRR_RE = re.compile(r"\b(SRR|ERR|DRR)\d{4,}\b", re.IGNORECASE)


def _efetch_text(
    db: str, id_or_ids: str | List[str], rettype: str = "full",
    retmode: str = "xml", *, max_retries: int = 3, base_delay: float = 2.0,
) -> str:
    _ensure_entrez_credentials()
    id_arg = ",".join(map(str, id_or_ids)) if isinstance(id_or_ids, list) else str(id_or_ids)
    for attempt in range(max_retries):
        try:
            h = Entrez.efetch(db=db, id=id_arg, rettype=rettype, retmode=retmode)
            data = h.read(); h.close()
            if isinstance(data, bytes):
                data = data.decode("utf-8", errors="replace")
            return data
        except HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                time.sleep(base_delay * 2 ** attempt)
                continue
            raise
    raise RuntimeError("efetch exhausted retries")


def _esummary_text(
    db: str, id_or_ids: str | List[str], *, max_retries: int = 3, base_delay: float = 2.0,
) -> str:
    _ensure_entrez_credentials()
    id_arg = ",".join(map(str, id_or_ids)) if isinstance(id_or_ids, list) else str(id_or_ids)
    for attempt in range(max_retries):
        try:
            h = Entrez.esummary(db=db, id=id_arg)
            data = h.read(); h.close()
            if isinstance(data, bytes):
                data = data.decode("utf-8", errors="replace")
            return data
        except HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                time.sleep(base_delay * 2 ** attempt)
                continue
            raise
    raise RuntimeError("esummary exhausted retries")


def entrez_id_to_srx(entrez_id: str, database: str = "sra") -> List[str]:
    """Resolve an Entrez ID to its SRX/ERX/DRX experiment accession(s)."""
    eid = normalize_entrez_id(entrez_id)
    db = normalize_database(database)
    if db == "sra":
        text = _efetch_text("sra", eid, rettype="full", retmode="xml")
    else:  # gds
        text = _esummary_text("gds", eid)
    seen, out = set(), []
    for m in SRX_RE.finditer(text):
        acc = m.group(0).upper()
        if acc not in seen:
            seen.add(acc); out.append(acc)
    return out


def get_srx_runs(srx_accession: str) -> List[str]:
    """Return SRR/ERR/DRR run accessions for a given SRX/ERX/DRX experiment."""
    srx = normalize_accession(srx_accession)
    if not is_srx_accession(srx):
        raise ValueError(f"Expected SRX/ERX/DRX accession; got {srx_accession!r}")

    _ensure_entrez_credentials()
    h = Entrez.esearch(db="sra", term=f"{srx}[Accession]")
    res = Entrez.read(h); h.close()
    ids = res.get("IdList", [])
    if not ids:
        return []
    text = _efetch_text("sra", ids[0], rettype="full", retmode="xml")
    seen, out = set(), []
    for m in SRR_RE.finditer(text):
        acc = m.group(0).upper()
        if acc not in seen:
            seen.add(acc); out.append(acc)
    return out


def get_srx_metadata_text(srx_or_entrez_id: str, database: str = "sra") -> str:
    """Return raw SRA experiment XML (or GEO esummary blob) for the given
    SRX accession or Entrez ID."""
    db = normalize_database(database)
    s = (srx_or_entrez_id or "").strip()
    if db == "sra":
        if is_srx_accession(s):
            acc = normalize_accession(s)
            _ensure_entrez_credentials()
            h = Entrez.esearch(db="sra", term=f"{acc}[Accession]")
            res = Entrez.read(h); h.close()
            ids = res.get("IdList", [])
            if not ids:
                return f"<error>No SRA record found for {acc}</error>"
            return _efetch_text("sra", ids[0], rettype="full", retmode="xml")
        eid = normalize_entrez_id(s)
        return _efetch_text("sra", eid, rettype="full", retmode="xml")
    eid = normalize_entrez_id(s)
    return _esummary_text("gds", eid)
