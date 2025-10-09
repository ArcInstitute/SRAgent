# import
## batteries
from __future__ import annotations
import os
import json
import random
import decimal
from subprocess import Popen, PIPE
import xml.etree.ElementTree as ET
from xml.parsers.expat import ExpatError
from Bio import Entrez

## 3rd party
import xmltodict


# functions
def batch_ids(ids: list[str], batch_size: int) -> list[list[str]]:
    """
    Batch a list of IDs into smaller lists of a given size.
    Args:
        ids: List of IDs.
        batch_size: Size of each batch.
    Returns:
        Generator yielding batches of IDs.
    """
    for i in range(0, len(ids), batch_size):
        yield ids[i : i + batch_size]


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


def xml2json(
    record: str, indent: int | None = None, max_records: int | None = None
) -> str:
    """
    Convert an XML record to a JSON object.
    Args:
        record: XML record.
        indent: Number of spaces to indent the JSON.
        max_records: Maximum number of records to return.
    Returns:
        JSON object or original record if conversion fails.
    """
    if not record:
        return ""
    try:
        d = xmltodict.parse(record)
        return json.dumps(truncate_data(d, max_records), indent=indent)
    except (ExpatError, TypeError, ValueError) as e:
        return record


def run_cmd(cmd: list) -> tuple[int, str, str]:
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


def to_json(results, indent: int = None) -> str:
    """
    Convert a dictionary to a JSON string.
    Args:
        results: a bigquery query result object
    Returns:
        str: JSON string
    """
    if results is None:
        return "No results found"

    def datetime_handler(obj):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        elif isinstance(obj, decimal.Decimal):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # convert to json
    try:
        ret = json.dumps(
            [dict(row) for row in results], default=datetime_handler, indent=indent
        )
        if ret == "[]":
            return "No results found"
        return ret
    except Exception as e:
        return f"Error converting results to JSON: {str(e)}"


def join_accs(accessions: list[str]) -> str:
    """
    Join a list of accessions into a string.
    Args:
        accessions: list of accessions
    Returns:
        str: comma separated string of accessions
    """
    return ", ".join([f"'{acc}'" for acc in accessions])


def set_entrez_access() -> None:
    """
    Set the Entrez access email and API key.
    The email and API key are stored in the environment variables.
    If no numbered email and API key are found, the default email and API key are used.
    If numbered email and API key are found, a random selection from the numbered ones is used.
    """
    # get number of emails and API keys
    email_indices = []
    for i in range(11):
        if os.getenv(f"EMAIL{i}"):
            email_indices.append(i)
    # if no numbered email and API key are found
    if len(email_indices) == 0:
        Entrez.email = os.getenv("EMAIL")
        Entrez.api_key = os.getenv("NCBI_API_KEY")
        return None

    # random selection from 1 to i
    n = random.choice(email_indices)
    Entrez.email = os.getenv(f"EMAIL{n}", os.getenv("EMAIL"))
    Entrez.api_key = os.getenv(f"NCBI_API_KEY{n}", os.getenv("NCBI_API_KEY"))


def truncate_data(data, max_items: int | None = None) -> dict:
    """
    Limits the number of leaf nodes in a nested data structure.
    Args:
        data: A nested structure of dicts, lists, and primitive values
        max_items: Maximum number of leaf nodes to include
    Returns:
        Limited data structure
    """
    if max_items is None:
        return data
    count = 0

    def process(obj):
        nonlocal count

        # Base case: primitive values
        if isinstance(obj, (str, int, float, bool, type(None))):
            count += 1
            return obj, (count <= max_items)

        # Handle dictionaries
        elif isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if count >= max_items:
                    break
                processed_value, continue_processing = process(value)
                if continue_processing:
                    result[key] = processed_value
                if not continue_processing:
                    break
            return result, (count <= max_items)

        # Handle lists
        elif isinstance(obj, list):
            result = []
            for item in obj:
                if count >= max_items:
                    break
                processed_item, continue_processing = process(item)
                if continue_processing:
                    result.append(processed_item)
                if not continue_processing:
                    break
            return result, (count <= max_items)

        # Other types treated as primitives
        else:
            count += 1
            return obj, (count <= max_items)

    limited_data, _ = process(data)
    return limited_data


# main
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    set_entrez_access()
