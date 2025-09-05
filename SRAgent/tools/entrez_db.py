# import
## batteries
import os
import time
from typing import Annotated, List

## 3rd party
from Bio import Entrez
from langchain_core.tools import tool

## package
from SRAgent.tools.utils import batch_ids


@tool
def which_entrez_databases(
    entrez_ids: Annotated[List[str], "List of Entrez IDs"],
    databases: Annotated[List[str], "Which NCBI databases to check"] = [
        "sra",
        "gds",
        "pubmed",
        "biosample",
        "bioproject",
    ],
) -> Annotated[str, "List of databases where each Entrez ID is found."]:
    """
    Determine which Entrez databases contain the provided Entrez IDs.
    """
    found_in = {entrez_id: [] for entrez_id in entrez_ids}

    # Query each database for the provided Entrez IDs
    for db in databases:
        for id_batch in batch_ids(entrez_ids, 200):
            time.sleep(0.34)  # Respect the rate limit
            handle = None
            try:
                handle = Entrez.esummary(db=db, id=",".join(id_batch))
                records = Entrez.read(handle)
                # Extract the IDs that were successfully retrieved
                if isinstance(records, list):
                    found_ids = {record["Id"] for record in records}
                else:
                    # In case only one record is returned
                    found_ids = {records["Id"]}
                for entrez_id in found_ids:
                    found_in[entrez_id].append(db)
            except Exception as e:
                continue
            finally:
                if handle is not None:
                    try:
                        handle.close()
                    except:
                        pass  # Handle cases where the handle might not be open

    # Prepare the output
    output_lines = []
    for entrez_id in entrez_ids:
        if not found_in[entrez_id]:
            output_lines.append(f"Entrez ID {entrez_id} not found in any databases.")
        else:
            output_lines.append(
                f"Entrez ID {entrez_id} found in: {', '.join(found_in[entrez_id])}."
            )

    # return the output as a string
    return "\n".join(output_lines)


if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv

    load_dotenv()
    Entrez.email = os.getenv("EMAIL")

    # test esummary
    input = {"entrez_ids": ["200121737", "100024679", "303444964"]}
    # print(which_entrez_databases.invoke(input))

    input = {"entrez_ids": ["34748561"]}
    print(which_entrez_databases.invoke(input))
