"""Input normalization for the SRAgent MCP server.

Centralized so the host LLM doesn't have to massage inputs into NCBI-canonical
form. Kept deterministic — no NCBI calls, no fuzzy matching — so behavior is
predictable and testable.

Covers:
  * Date     (YYYY-MM-DD / YYYY/MM / YYYY → YYYY/MM/DD)
  * Database (sra/gds + aliases like 'geo')
  * Accession (case-insensitive SRX/ERX/DRX/SRR/ERR/DRR + whitespace)
  * Entrez ID (int → str)
  * Organism (common-name shortcuts → scientific names; pass-through OK)

Organism handling is permissive on purpose — anything that isn't in the
shortcut table is passed through as-is so users can search any species
NCBI knows about. SRAgent's own `OrganismEnum` is reproduced here at the
common-name layer so the same shortcuts work.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple


# ---- Organism ----

# Common name (lowercased, stripped) -> NCBI scientific name. Mirrors the
# species in `SRAgent.organisms.OrganismEnum` plus a few synonyms users type.
COMMON_NAME_TO_SCIENTIFIC: dict[str, str] = {
    # mammals
    "human": "Homo sapiens",
    "humans": "Homo sapiens",
    "h. sapiens": "Homo sapiens",
    "h sapiens": "Homo sapiens",
    "homo sapiens": "Homo sapiens",
    "mouse": "Mus musculus",
    "mus musculus": "Mus musculus",
    "rat": "Rattus norvegicus",
    "rattus norvegicus": "Rattus norvegicus",
    "macaque": "Macaca mulatta",
    "rhesus": "Macaca mulatta",
    "marmoset": "Callithrix jacchus",
    "horse": "Equus caballus",
    "dog": "Canis lupus",
    "bovine": "Bos taurus",
    "cow": "Bos taurus",
    "cattle": "Bos taurus",
    "sheep": "Ovis aries",
    "pig": "Sus scrofa",
    "swine": "Sus scrofa",
    "rabbit": "Oryctolagus cuniculus",
    "naked mole rat": "Heterocephalus glaber",
    "chimpanzee": "Pan troglodytes",
    "gorilla": "Gorilla gorilla",
    "cat": "Felis catus",
    "bonobo": "Pan paniscus",
    "green monkey": "Chlorocebus aethiops",
    "goat": "Capra hircus",
    "alpaca": "Vicugna pacos",
    "platypus": "Ornithorhynchus anatinus",
    "ferret": "Mustela putorius",
    # birds
    "chicken": "Gallus gallus",
    "duck": "Anas platyrhynchos",
    # amphibians / fish / inverts / plants (subset of SRAgent's set)
    "frog": "Xenopus tropicalis",
    "axolotl": "Ambystoma mexicanum",
    "zebrafish": "Danio rerio",
    "salmon": "Salmo salar",
    "fruit fly": "Drosophila melanogaster",
    "drosophila": "Drosophila melanogaster",
    "roundworm": "Caenorhabditis elegans",
    "c. elegans": "Caenorhabditis elegans",
    "mosquito": "Anopheles gambiae",
    "thale cress": "Arabidopsis thaliana",
    "arabidopsis": "Arabidopsis thaliana",
    "rice": "Oryza sativa",
    "tomato": "Solanum lycopersicum",
    "corn": "Zea mays",
    "maize": "Zea mays",
}

# Heuristic: input that already looks like a Genus species pair (capitalised
# first token, two whitespace-separated words) is treated as a scientific
# name and passed through.
_SCIENTIFIC_NAME_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[a-z]+){1,2}$")


def normalize_organism(name: str) -> Tuple[str, bool]:
    """Normalize an organism name.

    Returns (scientific_name, was_recognized). When `was_recognized` is
    False, the name is passed through as-is so the caller can still attempt
    a search; callers may surface the unrecognized inputs as a warning.
    """
    if not isinstance(name, str):
        raise TypeError(f"organism must be a string, got {type(name).__name__}")
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("organism cannot be empty")
    key = cleaned.lower()
    if key in COMMON_NAME_TO_SCIENTIFIC:
        return COMMON_NAME_TO_SCIENTIFIC[key], True
    if _SCIENTIFIC_NAME_RE.match(cleaned):
        return cleaned, True
    return cleaned, False


def normalize_organisms(names: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Normalize a list of organism names. Returns (scientific_names, unrecognized)."""
    out: List[str] = []
    unrecognized: List[str] = []
    for n in names:
        sci, ok = normalize_organism(n)
        out.append(sci)
        if not ok:
            unrecognized.append(n)
    return out, unrecognized


# ---- Date ----

_DATE_PARTS_RE = re.compile(r"^\s*(\d{4})(?:[-/.](\d{1,2}))?(?:[-/.](\d{1,2}))?\s*$")


def normalize_date(s: str, *, end_of_period: bool = False) -> str:
    """Normalize various date strings to NCBI's `YYYY/MM/DD` form.

    Accepts `2024/01/15`, `2024-01-15`, `2024.01.15`, `2024-1-5`,
    `2024-03` (-> 2024/03/01 or 2024/03/31 if end_of_period),
    `2024` (-> 2024/01/01 or 2024/12/31 if end_of_period).
    """
    if not isinstance(s, str):
        raise TypeError(f"date must be a string, got {type(s).__name__}")
    m = _DATE_PARTS_RE.match(s)
    if not m:
        raise ValueError(f"Could not parse date: {s!r}")
    year = int(m.group(1))
    month_s, day_s = m.group(2), m.group(3)
    month = (12 if end_of_period else 1) if month_s is None else int(month_s)
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month in {s!r}: {month}")
    if day_s is None:
        day = _last_day_of_month(year, month) if end_of_period else 1
    else:
        day = int(day_s)
        if not 1 <= day <= 31:
            raise ValueError(f"Invalid day in {s!r}: {day}")
    return f"{year:04d}/{month:02d}/{day:02d}"


def _last_day_of_month(year: int, month: int) -> int:
    import datetime as _dt
    if month == 12:
        ny, nm = year + 1, 1
    else:
        ny, nm = year, month + 1
    return (_dt.date(ny, nm, 1) - _dt.timedelta(days=1)).day


# ---- Database ----

_DATABASE_ALIASES: dict[str, str] = {
    "sra": "sra",
    "gds": "gds",
    "geo": "gds",
    "gse": "gds",
    "gpl": "gds",
}


def normalize_database(db: Optional[str]) -> str:
    """Normalize a database name to canonical 'sra' or 'gds'. Defaults to 'sra'."""
    if db is None or db == "":
        return "sra"
    if not isinstance(db, str):
        raise TypeError(f"database must be a string, got {type(db).__name__}")
    key = db.strip().lower()
    if key in _DATABASE_ALIASES:
        return _DATABASE_ALIASES[key]
    raise ValueError(f"database must be 'sra' or 'gds' (or alias 'geo'); got {db!r}")


# ---- Accession ----

_SRX_FULL_RE = re.compile(r"^(SRX|ERX|DRX)\d{4,}$", re.IGNORECASE)
_SRR_FULL_RE = re.compile(r"^(SRR|ERR|DRR)\d{4,}$", re.IGNORECASE)


def normalize_accession(s: str) -> str:
    """Strip + uppercase a single accession string."""
    if not isinstance(s, str):
        raise TypeError(f"accession must be a string, got {type(s).__name__}")
    return s.strip().upper()


def is_srx_accession(s: str) -> bool:
    return bool(_SRX_FULL_RE.match(normalize_accession(s)))


def is_srr_accession(s: str) -> bool:
    return bool(_SRR_FULL_RE.match(normalize_accession(s)))


# ---- Entrez ID ----

_INT_RE = re.compile(r"^\d+$")


def normalize_entrez_id(eid: object) -> str:
    """Coerce an Entrez ID (int or str) to a numeric string."""
    if isinstance(eid, int):
        if eid < 0:
            raise ValueError(f"Entrez ID must be non-negative: {eid}")
        return str(eid)
    if isinstance(eid, str):
        cleaned = eid.strip()
        if not _INT_RE.match(cleaned):
            raise ValueError(f"Entrez ID must be numeric: {eid!r}")
        return cleaned
    raise TypeError(f"Entrez ID must be int or str, got {type(eid).__name__}")
