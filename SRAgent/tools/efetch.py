# import
## batteries
import os
import re
import time
from typing import Annotated, List

## 3rd party
from Bio import Entrez
from langchain_core.tools import tool

## package
from SRAgent.tools.utils import batch_ids, truncate_values, xml2json, set_entrez_access


@tool
def efetch(
    entrez_ids: Annotated[List[str], "List of Entrez IDs"],
    database: Annotated[str, "Database name (e.g., sra, gds, or pubmed)"],
) -> Annotated[str, "efetch results in XML format"]:
    """
    Run an Entrez efetch query on Entrez IDs to obtain metadata for the records.
    Useful for obtaining metadata for specific records.
    """
    set_entrez_access()
    batch_size = 200  # Maximum number of IDs per request as per NCBI guidelines
    records = []
    regex = re.compile(r'"(PAIRED|SINGLE)": +null')

    for id_batch in batch_ids(entrez_ids, batch_size):
        time.sleep(0.34)  # Respect the rate limit of 3 requests per second
        id_str = ",".join(id_batch)
        handle = None
        try:
            # Fetch the records for the current batch of IDs
            handle = Entrez.efetch(db=database, id=id_str, retmode="xml")
            batch_record = handle.read()
        except Exception as e:
            batch_record = f"Failed to fetch record for IDs: {id_str}. Error: {e}"
        finally:
            if handle is not None:
                try:
                    handle.close()
                except:
                    pass  # Handle cases where handle might not be open

        # Decode the record if necessary
        if isinstance(batch_record, bytes):
            try:
                batch_record = batch_record.decode("utf-8")
            except Exception as e:
                batch_record = f"Decoding error: {e}"

        # Truncate long values in the record
        batch_record = truncate_values(batch_record, max_length=1000)

        # convert to XML to JSON
        batch_record = xml2json(batch_record)

        # fix values for PAIRED and SINGLE
        batch_record = re.sub(regex, '\\1: "yes"', batch_record)

        # Check for errors in the response
        if "Error occurred: cannot get document summary" in batch_record:
            batch_record = (
                f"Failed to fetch record for IDs: {id_str}. Try a different database."
            )

        records.append(batch_record)

    # Combine all records into a single string
    return "\n".join(records)


if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv

    load_dotenv()
    Entrez.email = os.getenv("EMAIL")

    # Test efetch
    # input = { "entrez_ids" : ["35966237"], "database" : "sra"}
    # input = { "entrez_ids" : ["200254051"], "database" : "gds"}
    # input = {'entrez_ids': ['ERS23955837'], 'database': 'sra'}
    print(efetch.invoke(input))
