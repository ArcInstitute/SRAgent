# import
## batteries
from typing import List

## 3rd party
import psycopg2
from psycopg2.extensions import connection


# functions
def get_blocking(conn: connection) -> None:
    query = """
SELECT
    pid,
    usename,
    pg_blocking_pids(pid) AS blocked_by,
    state,
    query,
    NOW() - query_start AS duration
FROM pg_stat_activity
WHERE state != 'idle';
    """

    PIDs = []
    with conn.cursor() as cur:
        cur.execute(query)
        for x in cur.fetchall():
            PIDs.append(x[0])
    print(f"No. of blocking processes: {len(PIDs)}")
    return PIDs


def delete_blocking(PIDs: List[int]) -> None:
    # Terminate blocking processes using a new connection
    termination_query = "SELECT pg_terminate_backend(%s);"
    for pid in PIDs:
        print(f"Terminating process: {pid}")
        try:
            with db_connect() as termination_conn:  # New connection for termination
                with termination_conn.cursor() as termination_cur:
                    termination_cur.execute(termination_query, (pid,))
                    termination_conn.commit()
        except psycopg2.Error as e:
            print(f"Error terminating process {pid}: {e}")


# main
if __name__ == "__main__":
    from dotenv import load_dotenv
    from SRAgent.db.connect import db_connect

    load_dotenv()

    PIDs = None
    with db_connect() as conn:
        PIDs = get_blocking(conn)

    if PIDs:
        delete_blocking(PIDs)
