# import
## batteries
from __future__ import annotations
import os
import re
import sys
import time
import shutil
import tarfile
import tempfile
import requests
from typing import Annotated
from functools import lru_cache

## 3rd party
from langchain_core.tools import tool
import obonet
import networkx as nx
import appdirs

## package
from SRAgent.tools.vector_db import load_vector_store


# functions
@tool
def query_vector_db(
    query: Annotated[str, "The semantic search query"],
    k: Annotated[int, "The number of results to return"] = 3,
) -> str:
    """
    Perform a semantic search by querying a vector store
    """
    # Determine the cache directory using appdirs
    cache_dir = appdirs.user_cache_dir("SRAgent")
    chroma_dir_name = "mondo_chroma"
    chroma_dir_path = os.path.join(cache_dir, chroma_dir_name)

    # Create the cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Check if the Chroma DB directory exists
    if not os.path.exists(chroma_dir_path) or not os.listdir(chroma_dir_path):
        # Download and extract the tarball
        gcp_url = "https://storage.googleapis.com/arc-scbasecount/2025-02-25/disease_ontology/mondo_chroma.tar.gz"
        tarball_path = os.path.join(cache_dir, "mondo_chroma.tar.gz")

        print(f"Downloading Chroma DB from {gcp_url}...", file=sys.stdout)
        try:
            # Download the tarball
            response = requests.get(gcp_url, stream=True)
            response.raise_for_status()
            with open(tarball_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Downloaded to {tarball_path}, extracting...", file=sys.stdout)

            # Extract the tarball
            with tarfile.open(tarball_path) as tar:
                # Create a temporary directory for extraction
                with tempfile.TemporaryDirectory() as temp_dir:
                    tar.extractall(path=temp_dir)
                    # Find the extracted directory
                    extracted_items = os.listdir(temp_dir)
                    if extracted_items:
                        extracted_dir = os.path.join(temp_dir, extracted_items[0])
                        # If the extraction created a directory with the right name, move it
                        if os.path.isdir(extracted_dir):
                            # Remove any existing directory
                            if os.path.exists(chroma_dir_path):
                                shutil.rmtree(chroma_dir_path)
                            # Move the extracted directory to the cache
                            shutil.move(extracted_dir, chroma_dir_path)

            print(
                f"Extraction complete, Chroma DB available at {chroma_dir_path}",
                file=sys.stdout,
            )

            # Clean up the downloaded tarball
            os.remove(tarball_path)
        except Exception as e:
            return f"Error downloading or extracting Chroma DB: {e}"

    # Load the vector store
    vector_store = load_vector_store(chroma_dir_path, collection_name="mondo")

    # Query the vector store
    message = ""
    try:
        results = vector_store.similarity_search(query, k=k)
        message += f'# Results for query: "{query}"\n'
        for i, res in enumerate(results, 1):
            id = res.metadata.get("id", "No ID available")
            if not id:
                continue
            message += f"{i}. {id}\n"
            name = res.metadata.get("name", "No name available")
            message += f"  Ontology name: {name}\n"
            message += f"  Description: {res.page_content}\n"
    except Exception as e:
        return f"Error performing search: {e}"
    if not message:
        message = (
            f'No results found for query: "{query}". Consider refining your query.'
        )
    return message


# Cache for the ontology graph
_ONTOLOGY_GRAPH = None


@lru_cache(maxsize=1)
def get_mondo_ontology_graph(obo_path: str) -> nx.MultiDiGraph:
    """
    Load and cache the ontology graph from the OBO file.
    Uses lru_cache to ensure the graph is only loaded once.
    Args:
        obo_path: Path to the OBO file
    Returns:
        The ontology graph as a NetworkX MultiDiGraph
    """
    return obonet.read_obo(obo_path)


def all_neighbors(g, node):
    """Get all neighbors of a node in a directed graph, regardless of edge direction."""
    return set(g.predecessors(node)) | set(g.successors(node))


@tool
def get_neighbors(
    mondo_id: Annotated[str, "The MONDO ID (MONDO:XXXXXXX) or PATO ID (PATO:XXXXXXX)"],
) -> str:
    """
    Get the neighbors of a given MONDO ID in the MONDO disease ontology.
    """
    # check the ID format
    if not re.match(r"MONDO:\d{7}|PATO:\d{7}", mondo_id):
        return f'Invalid MONDO ID format: "{mondo_id}". The format must be "MONDO:XXXXXXX" or "PATO:XXXXXXX".'

    # Determine the cache directory using appdirs
    cache_dir = appdirs.user_cache_dir("SRAgent")
    obo_filename = "mondo.obo"
    obo_path = os.path.join(cache_dir, obo_filename)

    # Create the cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Download the OBO file if it doesn't exist
    if not os.path.exists(obo_path) or not os.listdir(cache_dir):
        obo_url = "https://purl.obolibrary.org/obo/mondo.obo"
        print(f"Downloading MONDO ontology from {obo_url}...", file=sys.stdout)
        try:
            response = requests.get(obo_url)
            response.raise_for_status()
            with open(obo_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded and saved to {obo_path}", file=sys.stdout)
        except Exception as e:
            return f"Error downloading MONDO ontology: {e}"

    # Get the cached ontology graph or load it if not available
    g = get_mondo_ontology_graph(obo_path)

    # get neighbors
    target_prefix = ["MONDO:", "PATO:"]
    message = ""
    try:
        message += f'# Neighbors in the ontology for: "{mondo_id}"\n'
        for i, node_id in enumerate(all_neighbors(g, mondo_id), 1):
            # filter out non-MONDO nodes
            if (
                not any(node_id.startswith(prefix) for prefix in target_prefix)
                or not g.nodes[node_id]
            ):
                continue
            # extract node name and description
            node_name = g.nodes[node_id]["name"]
            node_def = g.nodes[node_id].get("def")
            message += f"{i}. {node_id}\n"
            message += f"  Ontology name: {node_name}\n"
            message += f"  Description: {node_def}\n"
            # limit to 50 neighbors
            if i >= 50:
                break
    except Exception as e:
        return f"Error getting neighbors: {e}"

    if not message:
        message = f'No neighbors found for ID: "{mondo_id}".'
    return message


@tool
def query_mondo_ols(
    search_term: Annotated[str, "The term to search for in the MONDO ontology"],
) -> str:
    """
    Query the Ontology Lookup Service (OLS) for MONDO terms matching the search term.

    Args:
        search_term: The disease/condition term to search for

    Returns:
        Formatted string with MONDO search results
    """
    # Format search term for URL (handle special characters)
    import urllib.parse

    encoded_search_term = urllib.parse.quote(search_term)

    url = f"https://www.ebi.ac.uk/ols/api/search?q={encoded_search_term}&ontology=mondo"
    max_retries = 2
    retry_delay = 1

    for retry in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            break
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            return f"Error querying OLS API after {max_retries} attempts: {e}"

    results = data.get("response", {}).get("docs", [])
    if not results:
        return f"No results found for search term: '{search_term}'."

    message = f"# Results from OLS for '{search_term}':\n"
    for i, doc in enumerate(results, 1):
        # Each doc should have an 'obo_id', a 'label', and possibly a 'description'
        obo_id = doc.get("obo_id", "No ID")
        if not obo_id.startswith("MONDO:"):
            continue
        label = doc.get("label", "No label")
        description = doc.get("description", ["None provided"])
        try:
            description = description[0]
        except IndexError:
            pass
        if not description:
            description = "None provided"

        # MONDO often has synonyms which can be useful
        synonyms = doc.get("synonym", [])

        message += f"{i}. {obo_id} - {label}\n   Description: {description}\n"
        if synonyms:
            message += (
                f"   Synonyms: {', '.join(synonyms[:5])}"  # Show first 5 synonyms
            )
            if len(synonyms) > 5:
                message += f" (and {len(synonyms) - 5} more)"
            message += "\n"
    return message


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)

    # semantic search
    # query = "pelvic organ"
    # results = query_vector_db.invoke({"query" : query})
    # print(results); exit();

    # get neighbors
    # input = {'mondo_id': 'MONDO:0005267'}
    # neighbors = get_neighbors.invoke(input)
    # print(neighbors); exit();

    # query OLS
    input = {"search_term": "heart disorder"}
    results = query_mondo_ols.invoke(input)
    print(results)
