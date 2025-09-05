# import
## batteries
import os
from typing import List, Set

## 3rd party
import psycopg2
import pandas as pd
from pypika import Query, Table, Criterion
from psycopg2.extensions import connection

## package
from SRAgent.db.utils import execute_query


# functions
def db_find_srx(srx_accessions: List[str], conn: connection) -> List[int]:
    """
    Get SRX records on the database
    Args:
        conn: Connection to the database.
        database: Name of the database to query.
    Returns:
        List of entrez_id values of SRX records that have not been processed.
    """
    srx_metadata = Table("srx_metadata")
    stmt = (
        Query.from_(srx_metadata)
        .select("*")
        .distinct()
        .where(srx_metadata.srx_accession.isin(srx_accessions))
    )
    # return as pandas dataframe
    return pd.read_sql(str(stmt), conn)


def db_get_srx_records(
    conn: connection, column: str = "entrez_id", database: str = "sra"
) -> List[int]:
    """
    Get the entrez_id values of all SRX records in the database.
    Args:
        conn: Connection to the database.
        database: Name of the database to query.
    Returns:
        List of entrez_id values of SRX records that have not been processed.
    """
    srx_metadata = Table("srx_metadata")
    target_column = getattr(srx_metadata, column)
    stmt = (
        Query.from_(srx_metadata)
        .select(target_column)
        .distinct()
        .where(srx_metadata.database == database)
    )

    # Fetch the results and return a list of {target_column} values
    return [row[0] for row in execute_query(stmt, conn)]


def db_get_unprocessed_records(
    conn: connection, database: str = "sra", max_srx: int = 3
) -> pd.DataFrame:
    """
    Get all suitable unprocessed SRX records, limiting by unique srx_accession values.
    Args:
        conn: Connection to the database.
        database: Name of the database to query.
        max_srx: Maximum number of SRX records to return.
    Returns:
        dataframe of unprocessed SRX records.
    """
    srx_metadata = Table("srx_metadata")
    srx_srr = Table("srx_srr")

    subquery = (
        Query.from_(srx_metadata)
        .select(srx_metadata.srx_accession)
        .where(
            Criterion.all(
                [
                    srx_metadata.database == database,
                    srx_metadata.is_illumina == "yes",
                    srx_metadata.is_single_cell == "yes",
                    srx_metadata.is_paired_end == "yes",
                    ~srx_metadata.tech_10x.isin(["other", "not_applicable"]),
                ]
            )
        )
        .distinct()
        .limit(max_srx)
    )

    stmt = (
        Query.from_(srx_metadata)
        .inner_join(srx_srr)
        .on(srx_metadata.srx_accession == srx_srr.srx_accession)
        .where(srx_metadata.srx_accession.isin(subquery))
        .select(
            srx_metadata.srx_accession.as_("sample"),
            srx_srr.srr_accession.as_("accession"),
            srx_metadata.entrez_id.as_("entrez_id"),
            srx_metadata.tech_10x.as_("tech_10x"),
            srx_metadata.organism.as_("organism"),
        )
    )

    # fetch as pandas dataframe
    return pd.read_sql(str(stmt), conn)


def db_get_srx_accessions(conn: connection, database: str = "sra") -> Set[str]:
    """
    Get all SRX accessions in the screcounter database
    Args:
        conn: Connection to the database.
        database: Name of the sequence database (e.g., sra)
    Returns:
        Set of SRX accessions in the database.
    """
    srx_metadata = Table("srx_metadata")
    stmt = (
        Query.from_(srx_metadata)
        .where(srx_metadata.database == database)
        .select(srx_metadata.srx_accession)
        .distinct()
    )

    # fetch records
    with conn.cursor() as cur:
        cur.execute(str(stmt))
        return set([row[0] for row in cur.fetchall()])


def db_get_entrez_ids(conn: connection, database: str = "sra") -> Set[str]:
    """
    Get all Entrez IDs in the screcounter database
    Args:
        conn: Connection to the database.
        database: Name of the sequence database (e.g., sra)
    Returns:
        Set of Entrez IDs in the database.
    """
    srx_metadata = Table("srx_metadata")
    stmt = (
        Query.from_(srx_metadata)
        .where(srx_metadata.database == database)
        .select(srx_metadata.entrez_id)
        .distinct()
    )

    # fetch records
    with conn.cursor() as cur:
        cur.execute(str(stmt))
        return set([row[0] for row in cur.fetchall()])


def db_get_eval(conn: connection, dataset_ids: List[str]) -> pd.DataFrame:
    """
    Get the entrez_id values of all SRX records in the database.
    Args:
        conn: Connection to the database.
        dataset_ids: List of dataset_ids to return
    Returns:
        List of entrez_id values of SRX records that have not been processed.
    """
    tbl = Table("eval")
    stmt = (
        Query.from_(tbl)
        .select(target_column)
        .distinct()
        .where(tbl.dataset_id.isin(dataset_ids))
    )

    # Fetch the results and return a list of {target_column} values
    return [row[0] for row in execute_query(stmt, conn)]


# main
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    from SRAgent.db.connect import db_connect

    os.environ["DYNACONF"] = "test"
    with db_connect() as conn:
        # print(db_get_eval(conn, ["eval1"]))

        # print(db_get_srx_records(conn))
        # print(db_get_unprocessed_records(conn))
        print(len(db_get_srx_accessions(conn)))
        # print(db_find_srx(["SRX19162973"], conn))
