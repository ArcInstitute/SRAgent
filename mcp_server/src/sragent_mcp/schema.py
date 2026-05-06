"""RNA-seq metadata schema for the SRAgent MCP server.

A unified schema that covers both bulk and single-cell RNA-seq fields. The
host LLM populates whichever fields are applicable to the dataset and uses
the explicit `not_applicable` enum value for the rest.

Generic enough to cover most modalities. If you need modality-specific
schemas, derive them with `pydantic`'s `model_copy(update=...)` or use only
a subset of fields.
"""
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class YesNo(str, Enum):
    YES = "yes"
    NO = "no"
    UNSURE = "unsure"


class Modality(str, Enum):
    """High-level dataset modality."""

    BULK_RNA_SEQ = "bulk_rna_seq"
    SCRNA_SEQ = "scrna_seq"
    SNRNA_SEQ = "snrna_seq"
    OTHER = "other"
    UNSURE = "unsure"


class LibrarySelectionEnum(str, Enum):
    """RNA selection / depletion strategy."""

    POLY_A = "poly-A_selection"
    RIBO_DEPLETION = "ribo_depletion"
    TOTAL_RNA = "total_RNA"
    CDNA = "cDNA"
    OTHER = "other"
    UNSURE = "unsure"
    NOT_APPLICABLE = "not_applicable"


class LibraryStrategyEnum(str, Enum):
    """SRA-canonical library strategy."""

    RNA_SEQ = "RNA-Seq"
    MRNA_SEQ = "mRNA-Seq"
    SS_RNA_SEQ = "ssRNA-seq"
    TOTAL_RNA_SEQ = "Total_RNA-Seq"
    OTHER = "other"
    UNSURE = "unsure"


class StrandednessEnum(str, Enum):
    """Strand specificity of the library."""

    STRANDED_FORWARD = "stranded_forward"
    STRANDED_REVERSE = "stranded_reverse"
    UNSTRANDED = "unstranded"
    UNSURE = "unsure"
    NOT_APPLICABLE = "not_applicable"


# Single-cell-specific enums (mirror SRAgent's existing schema so this MCP
# is useful in scRNA-seq workflows too).
class ScRNALibPrepEnum(str, Enum):
    TENX = "10x_Genomics"
    SMART_SEQ = "Smart-seq"
    SMART_SEQ2 = "Smart-seq2"
    SMART_SEQ3 = "Smart-seq3"
    DROP_SEQ = "Drop-seq"
    CEL_SEQ = "CEL-seq"
    CEL_SEQ2 = "CEL-seq2"
    INDROPS = "indrops"
    PARSE = "Parse"
    BD_RHAPSODY = "BD_Rhapsody"
    OTHER = "other"
    NOT_APPLICABLE = "not_applicable"


class CellPrepEnum(str, Enum):
    SINGLE_CELL = "single_cell"
    SINGLE_NUCLEUS = "single_nucleus"
    UNSURE = "unsure"
    NOT_APPLICABLE = "not_applicable"


class RNASeqMetadata(BaseModel):
    """Per-experiment metadata fields a client should extract from SRA."""

    # Universal
    is_illumina: YesNo = Field(description="Is the dataset Illumina sequence data?")
    is_paired_end: YesNo = Field(description="Is the dataset paired-end?")
    modality: Modality = Field(description="Bulk, single-cell, single-nucleus, or other")
    organism: str = Field(description="Organism (scientific name)")
    instrument_model: str = Field(description="Sequencing instrument (e.g. Illumina NovaSeq 6000)")
    read_length: str = Field(description="Approximate read length in bp; 'unknown' if not stated")

    # Library characteristics
    library_strategy: LibraryStrategyEnum = Field(description="Reported library strategy")
    library_selection: LibrarySelectionEnum = Field(description="RNA selection / depletion strategy")
    strandedness: StrandednessEnum = Field(description="Strand specificity of the library")

    # Single-cell-specific (use 'not_applicable' when modality is bulk)
    scrna_lib_prep: ScRNALibPrepEnum = Field(
        default=ScRNALibPrepEnum.NOT_APPLICABLE,
        description="Single-cell library prep kit (10X, Smart-seq, etc.). Use 'not_applicable' for bulk.",
    )
    cell_prep: CellPrepEnum = Field(
        default=CellPrepEnum.NOT_APPLICABLE,
        description="Single-cell vs single-nucleus. Use 'not_applicable' for bulk.",
    )

    # Sample biology
    tissue: str = Field(description="Tissue(s) sequenced")
    disease: str = Field(description="Disease/condition or 'unknown'")
    perturbation: str = Field(description="Treatments / perturbations or 'unknown'")
    cell_line: str = Field(description="Cell line(s) if any; otherwise 'none'")
    sample_condition: str = Field(
        description="Short description of the experimental group / contrast"
    )

    # Run accessions (filled in by the MCP, not the LLM)
    srr_accessions: List[str] = Field(
        default_factory=list,
        description="SRR/ERR/DRR run accessions associated with this experiment",
    )


def schema_json(modality: Optional[str] = None) -> dict:
    """Return the JSON schema for `RNASeqMetadata`.

    The `modality` parameter is currently informational — the schema is
    unified. Future versions may return a slimmed-down schema for the
    requested modality only.
    """
    return RNASeqMetadata.model_json_schema()
