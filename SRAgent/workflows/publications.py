#!/usr/bin/env python
"""
Functions to process a DataFrame of accessions and find publications for each study.
"""

import os
import sys
import asyncio
import pandas as pd
import logging
from dotenv import load_dotenv
import nest_asyncio
from typing import Dict, Any

from SRAgent.agents.publications import create_publications_agent_stream, configure_logging

# Apply nest_asyncio to allow nested event loops (needed for Jupyter)
nest_asyncio.apply()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure logging to suppress specific messages
configure_logging()

# Load environment variables from .env file
load_dotenv()

async def find_publication_for_study(row: pd.Series) -> Dict[str, Any]:
    """
    Find publications for a single study (row in the DataFrame).
    
    Args:
        row: A pandas Series containing the accessions for a study.
        
    Returns:
        A dictionary with the publication information.
    """
    # Collect all non-null accessions
    accessions = []
    if pd.notna(row.get('sra_id')):
        accessions.append(row['sra_id'])
    if pd.notna(row.get('prj_id')):
        accessions.append(row['prj_id'])
    if pd.notna(row.get('gse_id')):
        accessions.append(row['gse_id'])
    
    if not accessions:
        logger.warning("No valid accessions found for this study")
        return {
            "pmid": None,
            "pmcid": None,
            "preprint_doi": None,
            "title": None,
            "message": "No valid accessions found for this study",
            "source": "not_found",
            "multiple_publications": False,
            "all_publications": []
        }
    
    # Create input message with all accessions
    accessions_str = " and ".join(accessions)
    input_message = {"messages": [{"role": "user", "content": f"Find publications for {accessions_str}. These accessions are linked to the same publication."}]}
    
    try:
        # Run the agent
        result = await create_publications_agent_stream(input_message)
        
        # Create a default result dictionary if None was returned
        if result is None:
            result = {
                "pmid": None,
                "pmcid": None,
                "preprint_doi": None,
                "title": None,
                "message": f"No valid result returned for accessions: {accessions_str}",
                "source": "error",
                "multiple_publications": False,
                "all_publications": []
            }
        
        # Add original accessions to the result
        result["accessions"] = accessions
        
        return result
    except Exception as e:
        logger.error(f"Error finding publication for {accessions_str}: {e}")
        return {
            "pmid": None,
            "pmcid": None,
            "preprint_doi": None,
            "title": None,
            "message": f"Error: {str(e)}",
            "source": "error",
            "multiple_publications": False,
            "all_publications": [],
            "accessions": accessions
        }

async def process_dataframe(df: pd.DataFrame, output_file: str, batch_size: int = 10) -> pd.DataFrame:
    """
    Process a DataFrame of accessions and find publications for each study.
    
    Args:
        df: A pandas DataFrame containing the accessions for each study.
        output_file: Path to save the results.
        batch_size: Number of studies to process in parallel.
        
    Returns:
        A pandas DataFrame with the publication information for each study.
    """
    # Set email and API key for Entrez from environment variables
    from Bio import Entrez
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
    # Initialize results list
    results = []
    
    # Process the DataFrame in batches
    total_batches = (len(df) + batch_size - 1) // batch_size
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        # Create tasks for each row in the batch
        tasks = [find_publication_for_study(row) for _, row in batch.iterrows()]
        
        # Run tasks concurrently
        batch_results = await asyncio.gather(*tasks)
        
        # Add results to the list
        results.extend(batch_results)
        
        # Save intermediate results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        
        # Log progress
        current_batch = i // batch_size + 1
        logger.info(f"Processed batch {current_batch}/{total_batches} ({min(i+batch_size, len(df))}/{len(df)} studies)")
    
    # Create final DataFrame
    results_df = pd.DataFrame(results)
    
    # Save final results
    results_df.to_csv(output_file, index=False)
    
    return results_df

def run_in_notebook(df, output_file, batch_size=10, email=None, api_key=None, env_file=None):
    """
    Run the publication processing in a Jupyter notebook.
    
    Args:
        df: A pandas DataFrame containing the accessions for each study.
        output_file: Path to save the results.
        batch_size: Number of studies to process in parallel.
        email: Email for Entrez (optional if set as environment variable).
        api_key: NCBI API key (optional if set as environment variable).
        env_file: Path to .env file (optional).
        
    Returns:
        A pandas DataFrame with the publication information for each study.
    """
    # Load environment variables from specified .env file if provided
    if env_file:
        load_dotenv(env_file)
    
    # Set environment variables if provided
    if email:
        os.environ["EMAIL"] = email
    if api_key:
        os.environ["NCBI_API_KEY"] = api_key
    
    # Use asyncio.run which handles the event loop properly
    try:
        # For Jupyter notebooks, we can use the current event loop
        # since nest_asyncio is applied
        results_df = asyncio.run(process_dataframe(df, output_file, batch_size))
    except RuntimeError:
        # If there's an issue with the event loop, try this alternative approach
        loop = asyncio.get_event_loop()
        results_df = loop.run_until_complete(process_dataframe(df, output_file, batch_size))
    
    print(f"Results saved to {output_file}")
    return results_df 