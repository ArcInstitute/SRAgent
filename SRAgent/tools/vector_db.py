# import
## batteries
import os
import sys

## 3rd party
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# functions
def verify_collection(
    persistent_client: chromadb.PersistentClient, collection_name: str
) -> None:
    """
    Verify that the collection exists and has documents
    Args:
        persistent_client: The persistent Chroma client
        collection_name: The name of the collection to verify
    Returns:
        None
    Raises:
        Exception: If the collection does not exist or has no documents
    """
    try:
        collection = persistent_client.get_collection(collection_name)
        count = collection.count()
        print(
            f"Found {count} documents in collection '{collection_name}'",
            file=sys.stdout,
        )
    except Exception as e:
        msg = f"Error accessing collection: {e}"
        msg += f"\nAvailable collections: {persistent_client.list_collections()}"
        raise Exception(msg)


def load_vector_store(chroma_path: str, collection_name: str = "uberon") -> Chroma:
    """
    Load a Chroma vector store from the specified path.
    Args:
        chroma_path: The path to the Chroma DB directory
        collection_name: The name of the collection to load
    Returns:
        A Chroma vector store
    Raises:
        FileNotFoundError: If the Chroma DB directory does not exist
        Exception: If the collection does not exist or has no documents
    """
    # Ensure the path exists
    if not os.path.exists(chroma_path):
        raise FileNotFoundError(f"Chroma DB directory not found: {chroma_path}")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load the persistent Chroma client
    persistent_client = chromadb.PersistentClient(path=chroma_path)

    # Load the existing vector store
    vector_store = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    return vector_store
