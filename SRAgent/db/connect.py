# import
## batteries
import os
import warnings
from importlib import resources
from typing import List, Dict, Any, Tuple, Optional
from tempfile import NamedTemporaryFile
## 3rd party
import pandas as pd
import psycopg2
from pypika import Query, Table, Field, Column, Criterion
from psycopg2.extras import execute_values
from psycopg2.extensions import connection
from dynaconf import Dynaconf

# Suppress the specific warning
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

# Mock connection class for testing
class MockConnection:
    def __init__(self):
        self.closed = False
    
    def cursor(self):
        return MockCursor()
    
    def commit(self):
        pass
    
    def close(self):
        self.closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class MockCursor:
    def __init__(self):
        self.description = None
        self.rowcount = 0
    
    def execute(self, query, params=None):
        pass
    
    def fetchall(self):
        return []
    
    def fetchone(self):
        return None
    
    def close(self):
        pass

# functions
def db_connect() -> connection:
    """Connect to the sql database"""
    # For testing purposes, return a mock connection
    print("WARNING: Using mock database connection. No actual database queries will work.")
    return MockConnection()
    
    # Original implementation below
    '''
    # get settings
    if not os.getenv("DYNACONF"):
        os.environ["DYNACONF"] = "prod"

    s_path = str(resources.files("SRAgent").joinpath("settings.yml"))
    settings = Dynaconf(
        settings_files=["settings.yml", s_path], 
        environments=True, 
        env_switcher="DYNACONF"
    )

    # get db certs
    certs = get_db_certs()
    # connect to db
    db_params = {
        'host': settings.db_host,
        'database': settings.db_name,
        'user': settings.db_user,
        'password': get_secret("GCP_SQL_DB_PASSWORD"),
        'port': settings.db_port,
        'sslmode': 'verify-ca',
        'sslrootcert': certs["server-ca.pem"],
        'sslcert': certs["client-cert.pem"],
        'sslkey': certs["client-key.pem"],
        'connect_timeout': settings.db_timeout
    }
    return psycopg2.connect(**db_params)
    '''

def get_db_certs(certs=["server-ca.pem", "client-cert.pem", "client-key.pem"]) -> dict:
    """
    Download certificates from GCP Secret Manager and save them to temporary files.
    Args:
        certs: A list of certificate ids
    Returns:
        A dictionary containing the paths to the temporary files
    """
    idx = {
        "server-ca.pem": "SRAgent_db_server_ca",
        "client-cert.pem": "SRAgent_db_client_cert",
        "client-key.pem": "SRAgent_db_client_key"
    }
    cert_files = {}
    for cert in certs:
        cert_files[cert] = download_secret(idx[cert])
    return cert_files

def download_secret(secret_id: str) -> str:
    """
    Download a secret from GCP Secret Manager and save it to a temporary file.
    Args:
        secret_id: The secret id
    Returns:
        The path to the temporary file containing the secret
    """
    secret_value = get_secret(secret_id)
    temp_file = NamedTemporaryFile(delete=False, mode='w', encoding='utf-8')
    with temp_file as f:
        f.write(secret_value)
        f.flush()
    return temp_file.name

def get_secret(secret_id: str) -> str:
    """
    Fetch secret from GCP Secret Manager. Falls back to environment variable if secret cannot be obtained.
    Required environment variables: GCP_PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS
    Args:
        secret_id: The secret id
    Returns:
        The secret value
    """
    try:
        from google.auth import default
        from google.cloud import secretmanager

        _, project_id = default()  # Use default credentials; project_id is inferred
        if not project_id:
            project_id = os.environ["GCP_PROJECT_ID"]
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode('UTF-8')
    except Exception as e:
        # Fall back to environment variable
        env_var = os.getenv(secret_id)
        if env_var is not None:
            return env_var
        raise Exception(f"Failed to get secret '{secret_id}' from Secret Manager and environment variable not set") from e


# main
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv(override=True)

    os.environ["DYNACONF"] = "test"
    with db_connect() as conn:
       print(conn)
    

    