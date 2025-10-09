# import
## batteries
from __future__ import annotations
from typing import Annotated
import sys

## 3rd party
from google.cloud import bigquery
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

## package
from SRAgent.tools.utils import to_json, join_accs


MISSING_CLIENT_MSG = (
    "BigQuery requires Google Cloud credentials. "
    "Enable one of: (1) 'gcloud auth application-default login' and set 'GCP_PROJECT_ID', "
    "or (2) set 'GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json' and 'GCP_PROJECT_ID'. "
    "Without credentials, BigQuery tools cannot run."
)

_WARNED_NO_CLIENT = False


def _warn_no_client_once():
    global _WARNED_NO_CLIENT
    if not _WARNED_NO_CLIENT:
        print(
            "WARNING: BigQuery not used — missing Google Cloud credentials.\n"
            "- Run: gcloud auth application-default login (and set GCP_PROJECT_ID), or\n"
            "- Set: GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json and GCP_PROJECT_ID",
            file=sys.stderr,
        )
        _WARNED_NO_CLIENT = True


# functions
@tool
def get_study_metadata(
    study_accessions: Annotated[
        list[str], "A list of SRA study accession numbers (SRP)"
    ],
    limit: Annotated[int, "The maximum number of records to return"] = 100,
    config: RunnableConfig = None,
) -> Annotated[str, "JSON string of SRA experiment metadata"]:
    """
    Get study-level metadata for a list of SRA study accessions.
    The metadata fields returned:
    - sra_study: SRA study accession (the query accession)
    - bioproject: BioProject accession (parent of study)
    - experiments: Comma-separated list of associated experiment accessions (SRX)
    """
    if config is None or config.get("configurable", {}).get("client") is None:
        _warn_no_client_once()
        return MISSING_CLIENT_MSG
    query = f"""
    WITH distinct_values AS (
        SELECT DISTINCT
            m.sra_study,
            m.bioproject,
            m.experiment
        FROM `nih-sra-datastore.sra.metadata` as m
        WHERE m.sra_study IN ({join_accs(study_accessions)})
        LIMIT {limit}
    )
    SELECT 
        sra_study,
        bioproject,
        STRING_AGG(experiment, ',') as experiments
    FROM distinct_values
    GROUP BY sra_study, bioproject
    """
    client = config["configurable"]["client"]
    return to_json(client.query(query))


# def create_get_experiment_metadata(client):
@tool
def get_experiment_metadata(
    experiment_accessions: Annotated[
        list[str], "A list of SRA experiment accession numbers (SRX)"
    ],
    limit: Annotated[int, "The maximum number of records to return"] = 100,
    config: RunnableConfig = None,
) -> Annotated[str, "JSON string of SRA experiment metadata"]:
    """
    Get experiment-level metadata for a list of SRA experiment accessions.
    The metadata fields returned:
    - experiment: SRA experiment accession (the query accession)
    - sra_study: SRA study accession (parent of experiment)
    - library_name: Library name (e.g., 1, 2, 3)
    - librarylayout: Library layout (e.g., single, paired)
    - libraryselection: Library selection (e.g., random, PCR)
    - librarysource: Library source (e.g., transcriptomic, genomic)
    - platform: Sequencing platform (e.g., Illumina, PacBio)
    - instrument: Sequencing instrument (e.g., HiSeq, NovaSeq)
    - acc: Comma-separated list of associated run accessions (SRR)
    """
    if config is None or config.get("configurable", {}).get("client") is None:
        _warn_no_client_once()
        return MISSING_CLIENT_MSG
    query = f"""
    WITH distinct_values AS (
        SELECT DISTINCT
            m.experiment,
            m.sra_study,
            m.library_name, 
            m.librarylayout,
            m.libraryselection, 
            m.librarysource,
            m.platform,
            m.instrument,
            m.acc
        FROM `nih-sra-datastore.sra.metadata` as m
        WHERE m.experiment IN ({join_accs(experiment_accessions)})
        LIMIT {limit}
    )
    SELECT
        experiment,
        library_name,
        librarylayout,
        libraryselection,
        librarysource,
        platform,
        instrument,
        STRING_AGG(acc, ',') as acc
    FROM distinct_values
    GROUP BY experiment, library_name, librarylayout, libraryselection, librarysource, platform, instrument
    """
    client = config["configurable"]["client"]
    return to_json(client.query(query))


@tool
def get_run_metadata(
    run_accessions: Annotated[list[str], "A list of SRA run accession numbers (SRR)"],
    limit: Annotated[int, "The maximum number of records to return"] = 100,
    config: RunnableConfig = None,
) -> Annotated[str, "JSON string of SRA run metadata"]:
    """
    Get run-level metadata for a list of SRA run accessions.
    The metadata fields returned:
    - acc: SRA run accession (the query accession)
    - experiment: SRA experiment accession (parent of run)
    - biosample: BioSample accession (parent of run)
    - organism: Organism name
    - assay_type: Assay type (e.g., RNA-Seq, ChIP-Seq)
    - mbases: Total bases sequenced (in megabases)
    - avgspotlen: Average spot length (in base pairs)
    - insertsize: Insert size (in base pairs)
    """
    if config is None or config.get("configurable", {}).get("client") is None:
        _warn_no_client_once()
        return MISSING_CLIENT_MSG
    query = f"""
    SELECT 
        m.acc,
        m.experiment,
        m.biosample,
        m.organism,
        m.assay_type,
        m.mbases,
        m.avgspotlen,
        m.insertsize
    FROM `nih-sra-datastore.sra.metadata` as m
    WHERE m.acc IN ({join_accs(run_accessions)})
    LIMIT {limit}
    """
    client = config["configurable"]["client"]
    return to_json(client.query(query))


@tool
def get_study_experiment_run(
    accessions: Annotated[list[str], "A list of SRA study accession numbers"],
    limit: Annotated[int, "The maximum number of records to return"] = 100,
    config: RunnableConfig = None,
) -> Annotated[str, "JSON string of SRA experiment metadata"]:
    """
    Get study, experiment, and run accessions for a list of SRA and/or ENA accessions.
    The accessions can be from any level of the SRA hierarchy: study, experiment, or run.
    The metadata fields returned:
    - study_accession: SRA or ENA study accession (SRP or PRJNA)
    - experiment_accession: SRA or ENA experiment accession (SRX or ERX)
    - run_accession: SRA or ENA run accession (SRR or ERR)
    """
    if config is None or config.get("configurable", {}).get("client") is None:
        _warn_no_client_once()
        return MISSING_CLIENT_MSG
    # get study accessions
    study_acc = [x for x in accessions if x.startswith("SRP") or x.startswith("PRJNA")]
    exp_acc = [x for x in accessions if x.startswith("SRX") or x.startswith("ERX")]
    run_acc = [x for x in accessions if x.startswith("SRR") or x.startswith("ERR")]

    # create WHERE query
    study_query = (
        f"m.sra_study IN ({join_accs(study_acc)})" if len(study_acc) > 0 else None
    )
    exp_query = f"m.experiment IN ({join_accs(exp_acc)})" if len(exp_acc) > 0 else None
    run_query = f"m.acc IN ({join_accs(run_acc)})" if len(run_acc) > 0 else None
    query = " OR ".join(
        [x for x in [study_query, exp_query, run_query] if x is not None]
    )

    # if empty
    if query is None or query == "":
        return "No valid accessions provided."

    # create full query
    query = f"""
    SELECT DISTINCT
        m.sra_study AS study_accession,
        m.experiment AS experiment_accession,
        m.acc AS run_accession
    FROM `nih-sra-datastore.sra.metadata` as m
    WHERE {query}
    LIMIT {limit}
    """

    # return query results
    client = config["configurable"]["client"]
    return to_json(client.query(query))


if __name__ == "__main__":
    # python -m SRAgent.tools.bigquery
    # setup
    from dotenv import load_dotenv

    load_dotenv(override=True)
    config = {"configurable": {"client": bigquery.Client()}}

    # test tools
    # get_study_experiment_run
    input = {"accessions": ["SRP548813", "SRX26939191", "SRR31573627"]}
    # print(get_study_experiment_run.invoke(input, config=config))

    # get_study_metadata
    input = {"study_accessions": ["SRP548813"]}
    # print(get_study_metadata.invoke(input, config=config))

    # get_experiment_metadata
    input = {"experiment_accessions": ["SRX26939191"]}
    # print(get_experiment_metadata.invoke(input, config=config))

    # get_run_metadata
    input = {"run_accessions": ["SRR31573627", "SRR31573628"]}
    # print(get_run_metadata.invoke(input, config=config))
