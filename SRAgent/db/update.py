# import
## batteries
import os
from typing import List, Tuple

## 3rd party
import pandas as pd
from psycopg2.extensions import connection

## package
from SRAgent.db.utils import get_unique_columns


# functions
def db_update(df: pd.DataFrame, table_name: str, conn: connection) -> None:
    """
    Update existing records in a PostgreSQL table using a Pandas DataFrame.
    If any record does not exist, rollback the transaction and raise an error.

    Args:
        df (pd.DataFrame): DataFrame containing the updated data.
        table_name (str): Name of the target table.
        conn (connection): psycopg2 database connection object.

    Raises:
        Exception: If records do not exist or an error occurs during the update.
    """

    # Return immediately if DataFrame is empty
    if df.empty:
        return None

    # Ensure df is a Pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception as e:
            raise Exception(f"Error converting input to DataFrame: {str(e)}")

    # Retrieve column names
    columns: List[str] = list(df.columns)

    # Get unique constraint columns from the database
    unique_columns: List[str] = get_unique_columns(table_name, conn)

    # Remove the 'id' column if it exists
    if "id" in columns:
        df = df.drop(columns=["id"])
        columns.remove("id")

    # Drop duplicate records based on unique constraints to avoid redundant updates
    df.drop_duplicates(subset=unique_columns, keep="first", inplace=True)

    # Convert DataFrame rows into a list of tuples
    values: List[Tuple] = [tuple(x) for x in df.to_numpy()]

    # Identify columns that should be updated (non-unique columns)
    non_unique_columns: List[str] = [
        col for col in columns if col not in unique_columns
    ]

    # If there are no columns to update, raise an exception
    if not non_unique_columns:
        raise Exception("No non-unique columns available to update")

    # Construct the SET clause for the UPDATE query
    set_clause = ", ".join(f"{col} = nv.{col}" for col in non_unique_columns)

    # Construct the JOIN condition for matching unique keys
    join_condition = " AND ".join(f"t.{col} = nv.{col}" for col in unique_columns)

    # Execute the update operation
    with conn.cursor() as cur:
        try:
            # Create a VALUES template string for batch updates
            value_template = "(" + ", ".join(["%s"] * len(columns)) + ")"

            # Generate the VALUES clause with formatted query arguments
            args_str = b",".join(cur.mogrify(value_template, val) for val in values)

            # Construct the SQL query using a WITH clause for batch updates
            update_query = f"""
                WITH new_values ({", ".join(columns)}) AS (
                    VALUES {args_str.decode("utf-8")}
                )
                UPDATE {table_name} AS t
                SET {set_clause}
                FROM new_values nv
                WHERE {join_condition}
                RETURNING 1;
            """

            # Execute the query
            cur.execute(update_query)

            # Ensure that all rows were updated; otherwise, rollback and raise an error
            if cur.rowcount != len(values):
                conn.rollback()
                raise Exception(
                    f"Update failed: expected {len(values)} rows updated, got {cur.rowcount}"
                )

            # Commit the transaction if successful
            conn.commit()

        except Exception as e:
            # Rollback in case of any error
            conn.rollback()
            raise Exception(f"Error updating data in {table_name}: {str(e)}")


# Main execution for testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    from SRAgent.db.connect import db_connect

    # Load environment variables
    load_dotenv(override=True)
    os.environ["DYNACONF"] = "test"

    # Test DataFrame
    df = pd.DataFrame(
        {
            "database": ["sra"],
            "entrez_id": [33249542],
            "srx_accession": ["SRX24914804"],
            "organism": ["human"],
        }
    )

    # Establish database connection and execute update function
    with db_connect() as conn:
        db_update(df, "srx_metadata", conn)
