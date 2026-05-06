"""Tests for sragent-mcp.

Run only the offline tests:
    pytest tests/ -m "not live"
Run everything (requires internet):
    pytest tests/
"""
from __future__ import annotations

import re
import socket
import pytest

from sragent_mcp.entrez import SRR_RE, SRX_RE, entrez_id_to_srx, get_srx_runs
from sragent_mcp.normalize import (
    COMMON_NAME_TO_SCIENTIFIC,
    is_srx_accession,
    normalize_accession,
    normalize_database,
    normalize_date,
    normalize_entrez_id,
    normalize_organism,
)
from sragent_mcp.schema import (
    LibrarySelectionEnum,
    LibraryStrategyEnum,
    Modality,
    RNASeqMetadata,
    StrandednessEnum,
    YesNo,
    schema_json,
)
from sragent_mcp.search import build_query, esearch_ids


def _has_internet(host: str = "eutils.ncbi.nlm.nih.gov") -> bool:
    try:
        socket.create_connection((host, 443), timeout=4)
        return True
    except OSError:
        return False


# ---------------- Query construction ----------------


class TestBuildQuery:
    def test_default_terms(self):
        q = build_query()
        assert '"RNA-seq"' in q
        assert '"transcriptome sequencing"' in q

    def test_organism_normalization(self):
        q = build_query(organisms=["human", "Mus musculus"])
        assert '"Homo sapiens"[Organism]' in q
        assert '"Mus musculus"[Organism]' in q

    def test_unknown_organism_passthrough(self):
        q = build_query(organisms=["Felis catus"])
        assert '"Felis catus"[Organism]' in q

    def test_date_iso_normalization(self):
        q = build_query(min_date="2024-01-15", max_date="2024-06")
        assert "2024/01/15:2024/06/30[PDAT]" in q

    def test_year_only_max_expands_to_eoy(self):
        q = build_query(min_date="2024", max_date="2024")
        assert "2024/01/01:2024/12/31[PDAT]" in q

    def test_no_modality_no_exclusion(self):
        q = build_query()
        assert " NOT (" not in q

    def test_bulk_modality_excludes_single_cell(self):
        q = build_query(modality="bulk_rna_seq")
        assert '"rna seq"[Strategy]' in q
        assert '"single cell"[Title]' in q
        assert '"10x Genomics"[All Fields]' in q
        assert " NOT (" in q

    def test_scrna_modality_no_exclusions(self):
        q = build_query(modality="scrna_seq")
        assert '"rna seq"[Strategy]' in q
        assert " NOT (" not in q

    def test_unknown_modality_raises(self):
        with pytest.raises(ValueError, match="Unknown modality"):
            build_query(modality="bogus")

    def test_extra_exclusions(self):
        q = build_query(extra_exclusions=['"foo"[Title]'])
        assert ' NOT ("foo"[Title])' in q


# ---------------- Normalization ----------------


class TestOrganismNormalization:
    @pytest.mark.parametrize("common, scientific", list(COMMON_NAME_TO_SCIENTIFIC.items()))
    def test_known_aliases(self, common, scientific):
        sci, ok = normalize_organism(common)
        assert sci == scientific and ok

    def test_case_insensitive(self):
        assert normalize_organism("HUMAN")[0] == "Homo sapiens"
        assert normalize_organism("Human")[0] == "Homo sapiens"

    def test_scientific_name_passthrough(self):
        sci, ok = normalize_organism("Felis catus")
        assert sci == "Felis catus" and ok  # passes scientific-name shape

    def test_arbitrary_passes_through_with_unrecognized_flag(self):
        sci, ok = normalize_organism("Made-Up Species")
        assert sci == "Made-Up Species" and not ok

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            normalize_organism("")


class TestDateNormalization:
    @pytest.mark.parametrize("raw, expected, eop", [
        ("2024/01/15", "2024/01/15", False),
        ("2024-01-15", "2024/01/15", False),
        ("2024-1-5", "2024/01/05", False),
        ("2024-03", "2024/03/01", False),
        ("2024-03", "2024/03/31", True),
        ("2024", "2024/01/01", False),
        ("2024", "2024/12/31", True),
        ("2024-02", "2024/02/29", True),  # leap year
        ("2023-02", "2023/02/28", True),
    ])
    def test_normalize_date(self, raw, expected, eop):
        assert normalize_date(raw, end_of_period=eop) == expected

    @pytest.mark.parametrize("raw", ["", "not a date", "2024/13/01"])
    def test_invalid_raises(self, raw):
        with pytest.raises(ValueError):
            normalize_date(raw)


class TestDatabaseNormalization:
    @pytest.mark.parametrize("raw, expected", [
        ("sra", "sra"), ("SRA", "sra"), ("gds", "gds"),
        ("geo", "gds"), ("GEO", "gds"), ("gse", "gds"),
        (None, "sra"), ("", "sra"),
    ])
    def test(self, raw, expected):
        assert normalize_database(raw) == expected

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            normalize_database("pubmed")


class TestAccessionNormalization:
    @pytest.mark.parametrize("raw, expected", [
        ("SRX12345", "SRX12345"), ("srx12345", "SRX12345"),
        ("  SRX12345  ", "SRX12345"),
    ])
    def test_normalize(self, raw, expected):
        assert normalize_accession(raw) == expected

    @pytest.mark.parametrize("raw, valid", [
        ("SRX12345", True), ("srx12345", True),
        ("ERX98765", True), ("DRX1234", True),
        ("AAA12345", False), ("SRX12", False),
    ])
    def test_validation(self, raw, valid):
        assert is_srx_accession(raw) is valid


class TestEntrezIdNormalization:
    @pytest.mark.parametrize("raw, expected", [
        (12345, "12345"), ("12345", "12345"), ("  12345  ", "12345"),
    ])
    def test_normalize(self, raw, expected):
        assert normalize_entrez_id(raw) == expected

    @pytest.mark.parametrize("raw", ["abc", "12.5", "", -1])
    def test_invalid(self, raw):
        with pytest.raises((ValueError, TypeError)):
            normalize_entrez_id(raw)


# ---------------- Schema ----------------


class TestSchema:
    def test_required_fields(self):
        req = set(schema_json()["required"])
        for k in ["is_illumina", "modality", "library_strategy", "library_selection",
                  "strandedness", "organism", "tissue"]:
            assert k in req

    def test_modality_enum(self):
        vals = {e.value for e in Modality}
        assert "bulk_rna_seq" in vals and "scrna_seq" in vals

    def test_library_selection_includes_bulk_methods(self):
        vals = {e.value for e in LibrarySelectionEnum}
        assert "poly-A_selection" in vals and "ribo_depletion" in vals
        assert "not_applicable" in vals  # for non-bulk samples

    def test_round_trip(self):
        m = RNASeqMetadata(
            is_illumina=YesNo.YES, is_paired_end=YesNo.YES,
            modality=Modality.BULK_RNA_SEQ,
            organism="Homo sapiens", instrument_model="NovaSeq",
            read_length="100", library_strategy=LibraryStrategyEnum.RNA_SEQ,
            library_selection=LibrarySelectionEnum.POLY_A,
            strandedness=StrandednessEnum.UNSURE,
            tissue="blood", disease="unknown", perturbation="unknown",
            cell_line="none", sample_condition="control",
        )
        d = m.model_dump()
        assert d["modality"] == "bulk_rna_seq"


# ---------------- Accession regex ----------------


class TestAccessionRegex:
    @pytest.mark.parametrize("text, expected", [
        ("Look at SRX12345", ["SRX12345"]),
        ("ERX98765 and DRX1234", ["ERX98765", "DRX1234"]),
        ("AAA12345", []), ("SRX12", []),
    ])
    def test_srx(self, text, expected):
        assert [m.group(0) for m in re.finditer(SRX_RE, text)] == expected

    @pytest.mark.parametrize("text, expected", [
        ("SRR123456", ["SRR123456"]),
        ("ERR9876543 DRR1234", ["ERR9876543", "DRR1234"]),
    ])
    def test_srr(self, text, expected):
        assert [m.group(0) for m in re.finditer(SRR_RE, text)] == expected


# ---------------- Live NCBI (gated) ----------------


@pytest.mark.live
@pytest.mark.skipif(not _has_internet(), reason="No internet")
class TestLiveNcbi:
    def test_search_human_returns_results(self):
        ids = esearch_ids(organisms=["human"], max_ids=3)
        assert len(ids) > 0
        assert all(i.isdigit() for i in ids)

    def test_bulk_modality_works(self):
        ids = esearch_ids(organisms=["human"], modality="bulk_rna_seq", max_ids=3)
        assert len(ids) >= 1

    def test_entrez_to_srx_round_trip(self):
        ids = esearch_ids(organisms=["human"], max_ids=2)
        assert len(ids) >= 1
        srx = entrez_id_to_srx(ids[0])
        assert all(re.fullmatch(r"(SRX|ERX|DRX)\d{4,}", a) for a in srx)

    def test_srx_to_runs_round_trip(self):
        ids = esearch_ids(organisms=["human"], max_ids=2)
        srx = entrez_id_to_srx(ids[0])
        if srx:
            runs = get_srx_runs(srx[0])
            assert all(re.fullmatch(r"(SRR|ERR|DRR)\d{4,}", r) for r in runs)
