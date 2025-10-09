#!/usr/bin/env python
# import
## batteries
from __future__ import annotations
import os
import sys
import argparse
from datetime import datetime
from typing import Any
import concurrent.futures

## 3rd party
from dotenv import load_dotenv
import pandas as pd
from psycopg2.extensions import connection

## package
from SRAgent.db.connect import db_connect
from SRAgent.db.upsert import db_upsert
from SRAgent.db.utils import db_list_tables, db_glimpse_tables
from SRAgent.db.get import db_find_srx
from SRAgent.db.create import create_table, create_table_router


def parse_cli_args():
    # get current date in YYYY-MM-DD format
    today = datetime.today().strftime("%Y-%m-%d")

    # argparse
    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
    ):
        pass

    desc = "Database tools"
    epi = """DESCRIPTION:
# list all tables
db-tools.py --list

# glimpse all tables
db-tools.py --glimpse

# dump all tables
db-tools.py --dump

# dump all tables with 4 parallel processes and 10000 chunk size
db-tools.py --dump --parallel 4 --chunk-size 10000

# drop >=1 table 
db-tools.py --drop srx_metadata

# delete >=1 SRX accession from all tables
db-tools.py --delete-srx SRR1234567

# delete >=1 SRX accession from the scRecounter tables
db-tools.py --delete-srx-screcounter SRR1234567

# upsert from a csv
db-tools.py --upsert-target srx_metadata --upsert-csv db_bkup/2024-12-17/srx_metadata.csv

# count records in all tables
db-tools.py --count-records

# display schema for all tables
db-tools.py --schema
    """
    parser = argparse.ArgumentParser(
        description=desc, epilog=epi, formatter_class=CustomFormatter
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        default=False,
        help="Dump all tables to CSV files",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=os.path.join("db_bkup", today),
        help="Directory to dump CSV files",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="Number of parallel processes for dumping tables",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Chunk size for reading large tables (rows per chunk)",
    )
    parser.add_argument(
        "--list", action="store_true", default=False, help="List all tables in database"
    )
    parser.add_argument(
        "--glimpse",
        action="store_true",
        default=False,
        help="Glimpse all tables in database",
    )
    parser.add_argument(
        "--count-records",
        action="store_true",
        default=False,
        help="Count records in all tables in database",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        default=False,
        help="Display the schema (column definitions) for all tables",
    )
    parser.add_argument("--view", type=str, default=None, help="View table in database")
    parser.add_argument(
        "--create",
        type=str,
        default=None,
        nargs="+",
        choices=list(create_table_router().keys()) + ["ALL"],
        help="Create >=1 table in the database",
    )
    parser.add_argument(
        "--find-srx",
        type=str,
        default=None,
        nargs="+",
        help="Find SRX accessions in the database",
    )
    parser.add_argument(
        "--upsert-csv", type=str, default=None, help="CSV file to upsert into database"
    )
    parser.add_argument(
        "--upsert-target", type=str, default=None, help="Table to upsert into database"
    )
    parser.add_argument(
        "--drop",
        type=str,
        default=None,
        nargs="+",
        help=">=1 table to delete from database.",
    )
    parser.add_argument(
        "--delete-srx",
        type=str,
        default=None,
        nargs="+",
        help=">=1 SRX accession to delete from the entire database, except the Eval table.",
    )
    parser.add_argument(
        "--delete-srx-screcounter",
        type=str,
        default=None,
        nargs="+",
        help=">=1 SRX accession to delete from the scRecounter tables in the database.",
    )
    parser.add_argument(
        "--tenant",
        type=str,
        default=os.getenv("DYNACONF"),
        choices=["test", "prod"],
        help="SRAgent database tenant to use. Defaults to DYNACONF env variable",
    )
    return parser.parse_args()


def dump_table(
    table: str,
    dump_dir: str,
    conn_params: dict[str, Any],
    chunk_size: int | None = None,
) -> str:
    """Dump a single table to a CSV file
    Args:
        table: Table name
        dump_dir: Directory to dump CSV file
        conn_params: Connection parameters for creating a new connection
        chunk_size: Number of rows to read at once (for large tables)
    Returns:
        Path to output file
    """
    outfile = os.path.join(dump_dir, f"{table}.csv")

    # Ensure any existing file is deleted before writing
    if os.path.exists(outfile):
        os.remove(outfile)
        print(f"Removed existing file: {outfile}")

    with db_connect(**conn_params) as conn:
        if chunk_size:
            # For large tables, process in chunks to avoid memory issues
            total_count = 0
            with open(outfile, "w") as f:
                # Get total count for progress tracking
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    total_rows = cur.fetchone()[0]

                # First chunk includes headers
                first_chunk = True
                offset = 0

                while True:
                    query = f"SELECT * FROM {table} LIMIT {chunk_size} OFFSET {offset}"
                    chunk_df = pd.read_sql(query, conn)

                    if chunk_df.empty:
                        break

                    # Write header only for first chunk
                    chunk_df.to_csv(f, header=first_chunk, index=False, mode="a")
                    first_chunk = False

                    # Update progress
                    rows_processed = min(offset + chunk_size, total_rows)
                    total_count += len(chunk_df)
                    offset += chunk_size

                    print(
                        f"  {table}: processed {rows_processed}/{total_rows} rows ({int(rows_processed / total_rows * 100)}%)"
                    )

                    if len(chunk_df) < chunk_size:
                        break
        else:
            # For smaller tables, read all at once
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
            df.to_csv(outfile, index=False)
            total_count = len(df)

    print(f"Dumped {table} to {outfile} ({total_count} records)")
    return outfile


# functions
def dump_all_tables(
    dump_dir: str, tenant: str, parallel: int = 1, chunk_size: int | None = None
) -> list[str]:
    """Dump all tables to CSV files
    Args:
        dump_dir: Directory to dump CSV files
        tenant: Database tenant
        conn: Database connection
        parallel: Number of parallel processes to use
        chunk_size: Number of rows to read at once (for large tables)
    Returns:
        List of paths to dumped CSV files
    """
    # get all tables
    db_tables = []
    with db_connect() as conn:
        db_tables = db_list_tables(conn)

    # skip tables
    db_tables = [x for x in db_tables if x not in ["screcounter_trace"]]

    # create dump directory
    dump_dir = os.path.join(dump_dir, tenant)
    os.makedirs(dump_dir, exist_ok=True)

    # Extract connection parameters from current connection to create new connections in worker threads
    conn_params = {}
    outfiles = []

    # Use parallel processing if requested
    if parallel > 1 and len(db_tables) > 1:
        print(f"Dumping {len(db_tables)} tables using {parallel} parallel processes")
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(
                    dump_table, table, dump_dir, conn_params, chunk_size
                ): table
                for table in db_tables
            }
            for future in concurrent.futures.as_completed(futures):
                table = futures[future]
                try:
                    outfile = future.result()
                    outfiles.append(outfile)
                except Exception as e:
                    print(f"Error dumping {table}: {e}")
    else:
        # Sequential processing
        for table in db_tables:
            try:
                outfile = dump_table(table, dump_dir, conn_params, chunk_size)
                outfiles.append(outfile)
            except Exception as e:
                print(f"Error dumping {table}: {e}")

    return outfiles


def count_records_per_table(conn: connection) -> None:
    """Count records in each table in the database
    Args:
        conn: Database connection
    """
    db_tables = db_list_tables(conn)

    for table in db_tables:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            print(f"{table}: {count}")


def display_table_schemas(conn: connection) -> None:
    """Display the schema for each table in the database.
    Args:
        conn: Database connection
    """
    db_tables = db_list_tables(conn)
    if not db_tables:
        print("No tables found in the database.")
        return

    print("Table Schemas:")
    with conn.cursor() as cur:
        for table in db_tables:
            print(f"\nTable: {table}")
            query = """
            SELECT
                column_name,
                data_type,
                character_maximum_length,
                is_nullable
            FROM
                information_schema.columns
            WHERE
                table_schema = 'public' -- Assuming tables are in the public schema
            AND
                table_name = %s
            ORDER BY
                ordinal_position;
            """
            try:
                cur.execute(query, (table,))
                columns = cur.fetchall()
                if not columns:
                    print("  No columns found.")
                    continue

                for col_name, data_type, max_len, is_nullable in columns:
                    type_str = data_type.upper()
                    if max_len is not None:
                        type_str += f"({max_len})"
                    null_str = "NULL" if is_nullable == "YES" else "NOT NULL"
                    print(f"  {col_name}: {type_str} ({null_str})")
            except Exception as e:
                print(f"  Error fetching schema for table {table}: {e}")


# functions
def main(args):
    # set pandas options
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 100)

    # set tenant
    if args.tenant:
        os.environ["DYNACONF"] = args.tenant

    # list tables
    if args.list:
        with db_connect() as conn:
            print("Tables in the database:")
            print("\n".join(db_list_tables(conn)))

    # count records in tables
    if args.count_records:
        with db_connect() as conn:
            count_records_per_table(conn)

    # glimpse tables
    if args.glimpse:
        with db_connect() as conn:
            db_glimpse_tables(conn)

    # view table
    if args.view:
        with db_connect() as conn:
            tbl_names = db_list_tables(conn)
            if args.view not in tbl_names:
                print(f"Table {args.view} not found in database")
                sys.exit(1)
            df = pd.read_sql(f"SELECT * FROM {args.view}", conn)
            df.to_csv(sys.stdout, index=False)

    # display schemas
    if args.schema:
        with db_connect() as conn:
            display_table_schemas(conn)

    # find SRX accessions
    if args.find_srx:
        with db_connect() as conn:
            df = db_find_srx(args.find_srx, conn)
            df.to_csv(sys.stdout, index=False)

    # create table
    if args.create:
        with db_connect() as conn:
            for table_name in args.create:
                create_table(table_name, conn)
                print(f"Created table: {table_name}")

    # dump tables
    if args.dump:
        dump_all_tables(args.dump_dir, args.tenant, args.parallel, args.chunk_size)

    # upsert tables
    if args.upsert_csv:
        if not args.upsert_target:
            print("Please provide --upsert-target")
            sys.exit(1)
        with db_connect() as conn:
            tbl_names = db_list_tables(conn)
            if args.upsert_target not in tbl_names:
                print(f"Table {args.upsert_target} not found in database")
                sys.exit(1)
            # determine separator from file extension
            df = pd.read_csv(args.upsert_csv, sep=get_sep(args.upsert_csv))
            db_upsert(df, args.upsert_target, conn)
            print(f"Upserted {args.upsert_csv} into {args.upsert_target}")

    # drop tables
    if args.drop:
        with db_connect() as conn:
            tbl_names = db_list_tables(conn)
            for table in args.drop:
                print(f'Attempting to drop: "{table}"...')
                if table not in tbl_names:
                    print(f"  Table {table} not found in database")
                    continue
                with conn.cursor() as cur:
                    cur.execute(f"DROP TABLE {table}")
                    conn.commit()
                print(f"  Dropped: {table}")

    # delete SRX accessions from all tables, except eval
    if args.delete_srx:
        with db_connect() as conn:
            for srx in args.delete_srx:
                # srx metadata & srx_srr
                for tbl_name in ["srx_metadata", "srx_srr"]:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"DELETE FROM {tbl_name} WHERE srx_accession = '{srx}'"
                        )
                        conn.commit()
                # screcounter_log & screcounter_star
                for tbl_name in [
                    "screcounter_log",
                    "screcounter_star_params",
                    "screcounter_star_results",
                ]:
                    with conn.cursor() as cur:
                        cur.execute(f"DELETE FROM {tbl_name} WHERE sample = '{srx}'")
                        conn.commit()
                print(f"Deleted: {srx}")

    # delete SRX accessions from scRecounter tables
    if args.delete_srx_screcounter:
        with db_connect() as conn:
            for srx in args.delete_srx_screcounter:
                # screcounter_log & screcounter_star
                for tbl_name in [
                    "screcounter_log",
                    "screcounter_star_params",
                    "screcounter_star_results",
                ]:
                    with conn.cursor() as cur:
                        cur.execute(f"DELETE FROM {tbl_name} WHERE sample = '{srx}'")
                        conn.commit()
                print(f"Deleted: {srx}")


def get_sep(infile: str) -> str:
    """Determine separator from file extension
    Args:
        infile: Input file
    Returns:
        Separator
    """
    sep = ","
    if infile.endswith(".csv"):
        sep = ","
    elif infile.endswith(".tsv"):
        sep = "\t"
    else:
        print("Input file must be CSV or TSV")
        sys.exit(1)
    return sep


# Example usage
if __name__ == "__main__":
    load_dotenv()
    args = parse_cli_args()
    main(args)
