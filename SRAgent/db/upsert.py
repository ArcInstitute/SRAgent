# import
## batteries
from typing import List, Dict, Any, Tuple, Optional

## 3rd party
import pandas as pd
from psycopg2.extras import execute_values
from psycopg2.extensions import connection

## package
from SRAgent.db.utils import get_unique_columns


# functions
def db_upsert(df: pd.DataFrame, table_name: str, conn: connection) -> None:
    """
    Upload a pandas DataFrame to PostgreSQL, performing an upsert operation.
    If records exist (based on unique constraints), update them; otherwise insert new records.
    Args:
        df: pandas DataFrame to upload
        table_name: name of the target table
        conn: psycopg2 connection object
    """
    # if df is empty, return
    if df.empty:
        return
    # if df is not dataframe, try to convert
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception as e:
            raise Exception(f"Error converting input to DataFrame: {str(e)}")

    # Get DataFrame columns
    columns = list(df.columns)

    # Create ON CONFLICT clause based on unique constraints
    unique_columns = get_unique_columns(table_name, conn)

    # Exclude 'id' column from the upsert
    if "id" in columns:
        df = df.drop(columns=["id"])
        columns.remove("id")

    # Drop duplicate records based on unique columns
    df.drop_duplicates(subset=unique_columns, keep="first", inplace=True)

    # Convert DataFrame to list of tuples
    values = [tuple(x) for x in df.to_numpy()]

    # Create the INSERT statement with ON CONFLICT clause
    insert_stmt = f"INSERT INTO {table_name} ({', '.join(columns)})"
    insert_stmt += f"\nVALUES %s"

    # Add DO UPDATE SET clause for non-unique columns
    do_update_set = [col for col in columns if col not in unique_columns]
    if do_update_set:
        do_update_set = ", ".join(f"{col} = EXCLUDED.{col}" for col in do_update_set)
        insert_stmt += f"\nON CONFLICT ({', '.join(unique_columns)})"
        insert_stmt += f"\nDO UPDATE SET {do_update_set}"
    else:
        # if no non-unique columns, add DO NOTHING clause
        insert_stmt += f"\nON CONFLICT ({', '.join(unique_columns)}) DO NOTHING"

    # Execute the query
    try:
        with conn.cursor() as cur:
            execute_values(cur, insert_stmt, values)
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise Exception(
            f"Error uploading data to {table_name}:\n{str(e)}\n\nSQL:{insert_stmt}\n\nValues:{str(values)}"
        )


# main
if __name__ == "__main__":
    from dotenv import load_dotenv
    from SRAgent.db.connect import db_connect

    load_dotenv()

    # test data
    df = pd.DataFrame(
        {
            "database": ["sra"],
            "entrez_id": ["25200088"],
            "srx_accession": ["SRX18216984"],
        }
    )
    with db_connect() as conn:
        db_upsert(df, "srx_metadata", conn)
