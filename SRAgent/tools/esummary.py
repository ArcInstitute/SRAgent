# import
## batteries
import os
import time
from typing import Annotated, List

## 3rd party
from Bio import Entrez
from langchain_core.tools import tool

## package
from SRAgent.tools.utils import batch_ids, truncate_values, xml2json, set_entrez_access


# functions
@tool
def esummary(
    entrez_ids: Annotated[List[str], "List of Entrez IDs"],
    database: Annotated[str, "Database name (e.g., sra, gds, or pubmed)"],
) -> Annotated[str, "eSummary results in XML format"]:
    """
    Run an Entrez esummary query on Entrez IDs to obtain summary information for the records.
    Useful for obtaining summary information for specific records.
    """
    set_entrez_access()
    batch_size = 200  # Maximum number of IDs per request as per NCBI guidelines
    max_string_length = 500  # Maximum length of a string in the record
    records = []

    for id_batch in batch_ids(entrez_ids, batch_size):
        time.sleep(
            0.34
        )  # Respect NCBI's rate limits (no more than 3 requests per second)
        id_str = ",".join(id_batch)
        handle = None

        try:
            # Fetch summary record for the current batch
            handle = Entrez.esummary(db=database, id=id_str, retmode="xml")
            batch_record = handle.read()
        except Entrez.Parser.ValidationError:
            batch_record = (
                f"Failed to fetch summary for IDs: {id_str}. Check if the IDs exist."
            )
        except Exception as e:
            batch_record = f"An error occurred: {e}"
        finally:
            if handle is not None:
                try:
                    handle.close()
                except:
                    pass  # Handle cases where the handle might not be open

        # Decode the record if necessary
        if isinstance(batch_record, bytes):
            try:
                batch_record = batch_record.decode("utf-8")
            except Exception as e:
                print(f"Decoding error: {e}")
                continue

        # Truncate long values in the record
        batch_record = truncate_values(batch_record, max_length=500)

        # convert to XML to JSON
        batch_record = xml2json(batch_record)

        # Check for errors in the response
        if "ERROR" in batch_record.upper() or "INVALID_ID" in batch_record.upper():
            batch_record = f"Failed to fetch summary for IDs: {id_str}. Try a different database or verify the IDs."

        # Append the batch record to the list of records
        records.append(batch_record)

    # Combine all batch records into a single string
    combined_records = "\n".join(records)
    return combined_records


if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv

    load_dotenv()
    Entrez.email = os.getenv("EMAIL")

    # test esummary
    # input = {"entrez_ids" : ["35966237"], "database" : "sra"}
    # input = {"entrez_ids" : ["200121737"], "database" : "sra"}
    input = {"entrez_ids": ["27978912"], "database": "sra"}
    print(esummary.invoke(input))
