# import
## batteries
import os
import json
import random
import decimal
from subprocess import Popen, PIPE
from typing import Annotated, List, Dict, Tuple, Any, Optional
import xml.etree.ElementTree as ET
from xml.parsers.expat import ExpatError
from Bio import Entrez
## 3rd party
import xmltodict
import re

# functions
def batch_ids(ids: List[str], batch_size: int) -> List[List[str]]:
    """
    Batch a list of IDs into smaller lists of a given size.
    Args:
        ids: List of IDs.
        batch_size: Size of each batch.
    Returns:
        List of batches.
    """
    for i in range(0, len(ids), batch_size):
        yield ids[i:i + batch_size]

def truncate_values(record, max_length: int) -> str:
    """
    Truncate long values in the record.
    Args:
        record: XML record to truncate.
        max_length: Maximum length of the value.
    Returns:
        Truncated record.
    """
    if record is None:
        return None
    try:
        root = ET.fromstring(record)
    except ET.ParseError:
        return record
    for item in root.findall(".//Item"):
        if item.text and len(item.text) > max_length:
            item.text = item.text[:max_length] + "...[truncated]"
    # convert back to string
    return ET.tostring(root, encoding="unicode")

def xml2json(record: str) -> Dict[str, Any]:
    """
    Convert an XML record to a JSON object.
    Args:
        record: XML record.
    Returns:
        JSON object.
    """
    try:
        return json.dumps(xmltodict.parse(record), indent=2)
    except ExpatError:
        return record

def run_cmd(cmd: list) -> Tuple[int, str, str]:
    """
    Run sub-command and return returncode, output, and error.
    Args:
        cmd: Command to run
    Returns:
        tuple: (returncode, output, error)
    """
    cmd = [str(i) for i in cmd]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    return p.returncode, output, err

def to_json(results, indent: int=None):
    """
    Convert a dictionary to a JSON string.
    Args:
        results: a bigquery query result object
    Returns:
        str: JSON string
    """
    def datetime_handler(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif isinstance(obj, decimal.Decimal):
            return str(obj)
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')
    # convert to json
    ret = json.dumps(
        [dict(row) for row in results],
        default=datetime_handler,
        indent=indent
    )
    if ret == "[]":
        return "No results found"
    return ret

def join_accs(accessions: List[str]) -> str:
    """
    Join a list of accessions into a string.
    Args:
        accessions: list of accessions
    Returns:
        str: comma separated string of accessions
    """
    return ', '.join([f"'{acc}'" for acc in accessions])

def set_entrez_access() -> None:
    """
    Set the Entrez access email and API key.
    The email and API key are stored in the environment variables.
    """
    i = 0
    while True:
        if os.getenv(f"EMAIL{i}"):
            i += 1
        else:
            break
    # if no numbered email and API key are found
    if i == 0:
        Entrez.email = os.getenv("EMAIL")
        Entrez.api = os.getenv("NCBI_API_KEY")
        return None
    # random selection from 1 to i
    n = random.randint(1, i)
    Entrez.email = os.getenv(f"EMAIL{n}", os.getenv("EMAIL"))
    Entrez.api = os.getenv(f"NCBI_API_KEY{n}", os.getenv("NCBI_API_KEY"))

def determine_database(accession: str) -> Optional[str]:
    """
    Determine the appropriate Entrez database based on the accession format.
    """
    # SRA accessions
    if re.match(r"^SRR\d+$", accession) or re.match(r"^ERR\d+$", accession) or re.match(r"^DRR\d+$", accession):
        return "sra"
    elif re.match(r"^SRX\d+$", accession) or re.match(r"^ERX\d+$", accession) or re.match(r"^DRX\d+$", accession):
        return "sra"
    elif re.match(r"^SRP\d+$", accession) or re.match(r"^ERP\d+$", accession) or re.match(r"^DRP\d+$", accession):
        return "sra"
    elif re.match(r"^PRJNA\d+$", accession) or re.match(r"^PRJEB\d+$", accession) or re.match(r"^PRJDB\d+$", accession):
        return "bioproject"
    
    # GEO accessions
    elif re.match(r"^GSE\d+$", accession):
        return "gds"
    elif re.match(r"^GSM\d+$", accession):
        return "gds"
    elif re.match(r"^GPL\d+$", accession):
        return "gds"
    
    # If no match, try to guess based on prefix
    elif accession.startswith("SRA"):
        return "sra"
    elif accession.startswith("GEO"):
        return "gds"
    elif accession.startswith("PRJ"):
        return "bioproject"
    
    # Default to sra if we can't determine
    return "sra"

# main
if __name__ == '__main__':
    set_entrez_access()