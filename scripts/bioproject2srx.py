#!/usr/bin/env python3
import sys
import time
import random
import argparse
import urllib.error
from Bio import Entrez

def retry_with_backoff(func, max_retries=10, initial_delay=1.0, backoff_factor=2.0):
    """
    Decorator to retry a function with exponential backoff when HTTP 429 errors occur.
    
    Args:
        func: The function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplicative factor for exponential backoff
        
    Returns:
        The wrapped function
    """
    def wrapper(*args, **kwargs):
        retries = 0
        delay = initial_delay
        
        while True:
            try:
                return func(*args, **kwargs)
            except urllib.error.HTTPError as e:
                if e.code == 429 and retries < max_retries:
                    # Add jitter to prevent synchronized retries
                    jitter = random.uniform(0, 0.1 * delay)
                    sleep_time = delay + jitter
                    
                    print(f"HTTP 429 error: Too many requests. Retrying in {sleep_time:.2f} seconds... "
                          f"(Attempt {retries+1}/{max_retries})", file=sys.stderr)
                    
                    time.sleep(sleep_time)
                    retries += 1
                    delay *= backoff_factor
                else:
                    raise
    return wrapper

def fetch_srx_accessions(bioproject_id, email, max_records=None, max_retries=5, initial_delay=1.0, backoff_factor=2.0):
    # set email
    Entrez.email = email

    # Create retry-enabled versions of Entrez functions
    esearch_with_retry = retry_with_backoff(Entrez.esearch, max_retries, initial_delay, backoff_factor)
    elink_with_retry = retry_with_backoff(Entrez.elink, max_retries, initial_delay, backoff_factor)
    efetch_with_retry = retry_with_backoff(Entrez.efetch, max_retries, initial_delay, backoff_factor)
    
    # run esearch
    print(f"esearch of bioproject for: {bioproject_id}", file=sys.stderr)
    handle = esearch_with_retry(db="bioproject", term=bioproject_id)
    record = Entrez.read(handle)
    if not record["IdList"]:
        raise ValueError(f"No UID found for BioProject {bioproject_id}")
    bioproject_uid = record["IdList"][0]

    # run elink
    print(f"elink of bioproject for: {bioproject_id}", file=sys.stderr)
    handle = elink_with_retry(dbfrom="bioproject", id=bioproject_uid, db="sra")
    linkset = Entrez.read(handle)
    if not linkset[0]["LinkSetDb"]:
        raise ValueError(f"No linked SRA records found for {bioproject_id}")
    sra_ids = [link["Id"] for link in linkset[0]["LinkSetDb"][0]["Link"]]
    print(f"  Total SRA records: {len(sra_ids)}", file=sys.stderr)

    # filter to max records
    if max_records is not None:
        sra_ids = sra_ids[:max_records]

    # fetch SRX accessions and titles
    srx_records = {}
    for sra_id in sra_ids:
        print(f"efetch of sra for: {sra_id}", file=sys.stderr)
        handle = efetch_with_retry(db="sra", id=sra_id, rettype="runinfo", retmode="text")
        lines = handle.read().decode('utf-8').splitlines()
        # get header
        header = {x:i for i,x in enumerate(lines[0].split(","))}

        for line in lines[1:]:
            fields = line.split(",")
            try:
                experiment_acc = fields[header["Experiment"]]
                # Extract the experiment title/name if available, otherwise use a placeholder
                experiment_name = fields[header["LibraryName"]] if "LibraryName" in header and fields[header["LibraryName"]] else "No title available"
                srx_records[experiment_acc] = experiment_name
            except KeyError:
                continue
    print(f"  Total SRX records: {len(srx_records)}", file=sys.stderr)
    return [(acc, name) for acc, name in sorted(srx_records.items())]

def main():
    parser = argparse.ArgumentParser(description="Fetch all SRX accessions from a BioProject")
    parser.add_argument("bioproject_id", help="NCBI BioProject accession (e.g., PRJNA123456)")
    parser.add_argument("--email", required=True, help="Your email for NCBI Entrez access")
    parser.add_argument("--max-retries", type=int, default=5, 
                       help="Maximum number of retry attempts for HTTP 429 errors")
    parser.add_argument("--initial-delay", type=float, default=1.0,
                       help="Initial delay in seconds between retries")
    parser.add_argument("--backoff-factor", type=float, default=2.0,
                       help="Multiplicative factor for exponential backoff")
    parser.add_argument("--format", choices=['tab', 'csv'], default='tab', 
                       help="Output format: tab-delimited or comma-separated")
    parser.add_argument("--max-records", type=int, default=None, 
                       help="Maximum number of records to fetch")
    args = parser.parse_args()

    try:
        srx_records = fetch_srx_accessions(
            args.bioproject_id, 
            args.email,
            max_records=args.max_records,
            max_retries=args.max_retries,
            initial_delay=args.initial_delay,
            backoff_factor=args.backoff_factor
        )
        delimiter = '\t' if args.format == 'tab' else ','
        # Print header
        print(f"Accession{delimiter}Name")
        # Print records
        for acc, name in srx_records:
            print(f"{acc}{delimiter}{name}")
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print(f"Error: HTTP 429 Too Many Requests - Maximum retry attempts exceeded. "
                  f"Please wait and try again later.", file=sys.stderr)
        else:
            print(f"HTTP Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()