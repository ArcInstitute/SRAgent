#!/usr/bin/env python3
"""
CLI script to fetch SRA project accessions (SRP) for SRX accessions in the database.

This script queries the PostgreSQL database for SRX accessions and uses BigQuery
to retrieve the corresponding SRA project accessions (SRP).
"""

# import
## batteries
from __future__ import annotations
import os
import sys
import argparse
import logging
from typing import List, Optional

## 3rd party
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv

## package (assuming SRAgent is installed or in path)
try:
    from SRAgent.db.connect import db_connect
    from SRAgent.tools.utils import join_accs, to_json
except ImportError:
    print("Error: SRAgent package not found. Please install or add to PYTHONPATH.")
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s- %(message)s")
logger = logging.getLogger(__name__)


def get_srx_accessions_from_db(
    conn, table_name: str = "srx_metadata", limit: Optional[int] = None
) -> List[str]:
    """Get SRX accessions from the PostgreSQL database.

    Args:
        conn: Database connection object
        table_name: Name of the table to query (default: srx_metadata)
        limit: Maximum number of records to fetch (None for all)

    Returns:
        List of SRX accession strings
    """
    logger.info(f"Fetching SRX accessions from table: {table_name}")

    # Build query
    query = f"SELECT DISTINCT srx_accession FROM {table_name}"
    if limit:
        query += f" LIMIT {limit}"

    # Execute query
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()

    srx_accessions = [row[0] for row in results if row[0]]
    logger.info(f"Found {len(srx_accessions)} SRX accessions")

    return srx_accessions


def get_project_accessions_from_bigquery(
    srx_accessions: List[str], batch_size: int = 100
) -> pd.DataFrame:
    """Query BigQuery to get SRA project accessions for SRX accessions.

    Args:
        srx_accessions: List of SRX accession strings
        batch_size: Number of accessions to query at once

    Returns:
        DataFrame with columns: experiment, sra_study, bioproject
    """
    logger.info(f"Querying BigQuery for {len(srx_accessions)} SRX accessions")

    # Initialize BigQuery client
    client = bigquery.Client()

    all_results = []

    # Process in batches
    for i in range(0, len(srx_accessions), batch_size):
        batch = srx_accessions[i : i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} ({len(batch)} accessions)")

        # Build query
        query = f"""
        SELECT DISTINCT
            m.experiment,
            m.sra_study,
            m.bioproject
        FROM `nih-sra-datastore.sra.metadata` as m
        WHERE m.experiment IN ({join_accs(batch)})
        """

        try:
            # Execute query
            query_job = client.query(query)
            results = query_job.result()

            # Convert to list of dicts
            batch_results = [dict(row) for row in results]
            all_results.extend(batch_results)
            logger.info(f"Retrieved {len(batch_results)} records from this batch")

        except Exception as e:
            logger.error(f"Error querying batch: {e}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    logger.info(f"Total records retrieved: {len(df)}")

    return df


def update_database_with_projects(
    conn, df: pd.DataFrame, table_name: str = "srx_metadata"
) -> None:
    """Update the PostgreSQL database with SRA project accessions.

    Args:
        conn: Database connection object
        df: DataFrame with experiment, sra_study, bioproject columns
        table_name: Name of the table to update
    """
    logger.info(f"Updating table {table_name} with project accessions")

    # Check if columns exist in the table
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}'
    """)
    existing_columns = [row[0] for row in cursor.fetchall()]

    # Determine which columns to update
    update_cols = []
    if "sra_study" in existing_columns:
        update_cols.append("sra_study")
    if "bioproject" in existing_columns:
        update_cols.append("bioproject")

    if not update_cols:
        logger.warning(
            f"Table {table_name} does not have sra_study or bioproject columns"
        )
        cursor.close()
        return

    # Update records
    updated_count = 0
    for _, row in df.iterrows():
        srx = row["experiment"]

        # Build update statement and values
        set_clause = ", ".join([f"{col} = %s" for col in update_cols])

        # Map DataFrame columns to database columns
        col_mapping = {"sra_study": "sra_study", "bioproject": "bioproject"}
        values = [row.get(col_mapping.get(col, col)) for col in update_cols]
        values.append(srx)

        update_query = f"""
            UPDATE {table_name}
            SET {set_clause}, updated_at = CURRENT_TIMESTAMP
            WHERE srx_accession = %s
        """

        try:
            cursor.execute(update_query, values)
            updated_count += cursor.rowcount
        except Exception as e:
            logger.error(f"Error updating SRX {srx}: {e}")
            continue

    conn.commit()
    cursor.close()
    logger.info(f"Updated {updated_count} records in the database")


def main():
    """Main function to run the CLI script."""

    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        description="Fetch SRA project accessions for SRX accessions using BigQuery",
        formatter_class=CustomFormatter,
    )

    parser.add_argument(
        "--table",
        type=str,
        default="srx_metadata",
        help="Database table to query for SRX accessions",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of SRX accessions to process",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of accessions to query at once via SRA BigQuery",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path. If None, print to stdout",
    )

    parser.add_argument(
        "--update-db",
        action="store_true",
        help="Update the database with retrieved project accessions",
    )

    parser.add_argument(
        "--env-file", type=str, default=".env", help="Path to .env file"
    )

    parser.add_argument(
        "--dynaconf",
        type=str,
        default=os.getenv("DYNACONF") or "test",
        help="Dynaconf environment to use",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(args.env_file, override=True)
    os.environ["DYNACONF"] = args.dynaconf

    # Connect to database
    logger.info(f"Connecting to database (DYNACONF={args.dynaconf})")

    try:
        with db_connect() as conn:
            # Get SRX accessions from database
            srx_accessions = get_srx_accessions_from_db(
                conn, table_name=args.table, limit=args.limit
            )

            if not srx_accessions:
                logger.warning("No SRX accessions found in database")
                return

            # Query BigQuery for project accessions
            df = get_project_accessions_from_bigquery(
                srx_accessions, batch_size=args.batch_size
            )

            if df.empty:
                logger.warning("No project accessions found in BigQuery")
                return

            # Output results
            if args.output:
                df.to_csv(args.output, index=False)
                logger.info(f"Results saved to {args.output}")
            else:
                print("\nResults:")
                print(df.to_string(index=False))

            # Print summary statistics
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Total SRX accessions queried: {len(srx_accessions)}")
            print(f"Records found in BigQuery: {len(df)}")
            print(f"Unique SRA studies (SRP): {df['sra_study'].nunique()}")
            print(f"Unique BioProjects: {df['bioproject'].nunique()}")

            # Show SRX accessions not found
            found_srx = set(df["experiment"].unique())
            missing_srx = set(srx_accessions) - found_srx
            if missing_srx:
                print(f"\nSRX accessions not found in BigQuery: {len(missing_srx)}")
                if len(missing_srx) <= 10:
                    print(f"Missing: {', '.join(sorted(missing_srx))}")
                else:
                    print(
                        f"First 10 missing: {', '.join(sorted(list(missing_srx)[:10]))}"
                    )
            print("=" * 60)

            # Update database if requested
            if args.update_db:
                update_database_with_projects(conn, df, table_name=args.table)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
