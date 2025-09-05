# import
## batteries
import os
import sys
from typing import List, Tuple, Optional

## 3rd party
import psycopg2
import pandas as pd
from pypika import Query, Table
from psycopg2.extensions import connection


# functions
def get_unique_columns(table: str, conn: connection) -> List[str]:
    """
    Get all unique constraint columns for a table from the database schema.
    Prioritizes composite unique constraints over primary keys.

    Args:
        table: Name of the table
        conn: Database connection

    Returns:
        List of column names that form the most appropriate unique constraint
    """
    query = """
    SELECT c.contype, ARRAY_AGG(a.attname ORDER BY array_position(c.conkey, a.attnum)) as columns
    FROM pg_constraint c
    JOIN pg_class t ON c.conrelid = t.oid
    JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(c.conkey)
    WHERE t.relname = %s 
    AND c.contype IN ('p', 'u')  -- primary key or unique constraint
    GROUP BY c.conname, c.contype
    ORDER BY c.contype DESC;  -- 'u'nique before 'p'rimary key
    """

    with conn.cursor() as cur:
        cur.execute(query, (table,))
        constraints = cur.fetchall()

    if not constraints:
        raise ValueError(f"No unique constraints found in table {table}")

    # Prefer composite unique constraints over single-column primary keys
    for constraint_type, columns in constraints:
        if len(columns) > 1 or constraint_type == "u":
            return columns

    # Fall back to primary key if no other suitable constraint found
    return constraints[0][1]


def db_list_tables(conn: connection) -> List[str]:
    """
    List all tables in the public schema of the database.
    Args:
        conn: Connection to the database.
    Returns:
        List of table names in the public schema.
    """
    tables = Table("tables", schema="information_schema")
    query = (
        Query.from_(tables).select("table_name").where(tables.table_schema == "public")
    )
    with conn.cursor() as cur:
        cur.execute(str(query))
        ret = cur.fetchall()
        if isinstance(ret, list):
            ret = [x[0] for x in ret]
        return ret


def db_get_table(table_name: str, conn: connection) -> pd.DataFrame:
    """
    Return all records in a table as a pandas dataframe
    Args:
        table_name: Name of the table.
        conn: Connection to the database.
    Returns:
        Pandas dataframe of the table
    """
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql(query, conn)


def db_glimpse_tables(conn: connection) -> None:
    """
    Print the first 5 rows of each table in the database.
    Args:
        conn: Connection to the database.
    """
    for table in db_list_tables(conn):
        print(f"#-- Table: {table} --#")
        df = pd.read_sql(f"SELECT * FROM {table} LIMIT 5", conn)
        df.to_csv(sys.stdout, sep="\t", index=False)
        print()


def execute_query(stmt, conn: connection) -> Optional[List[Tuple]]:
    """
    Execute a query and return the results, if any.
    Args:
        stmt: Query to execute.
        conn: Connection to the database.
    Returns:
        Results of the query, if any.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(str(stmt))
            if cur.description:  # If cur.description exists, it's a SELECT query
                results = cur.fetchall()
                return results if results else []  # Return empty list for no results
            else:
                conn.commit()  # Commit only for data-modifying queries
                return None
    except psycopg2.errors.DuplicateTable as e:
        print(f"Table already exists: {e}")
        return None
    except psycopg2.ProgrammingError as e:
        print(f"SQL Programming Error: {e}")
        return None
    except psycopg2.Error as e:
        print(f"Database Error: {e}")
        conn.rollback()
        raise
    except Exception as e:
        print(f"Unexpected Error: {e}")
        raise


# main
if __name__ == "__main__":
    from dotenv import load_dotenv
    from SRAgent.db.connect import db_connect

    load_dotenv()

    os.environ["DYNACONF"] = "test"
    with db_connect() as conn:
        # list unisue columns
        for table in db_list_tables(conn):
            unique_cols = get_unique_columns(table[0], conn)
            print(f"Table: {table[0]} - Unique columns: {unique_cols}")

        # view tables
        db_glimpse_tables(conn)
