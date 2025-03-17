#!/usr/bin/env python
"""
Script to process a DataFrame of accessions and find publications for each study.
"""

import os
import sys
import asyncio
import pandas as pd
import logging
from tqdm import tqdm
from Bio import Entrez
from typing import List, Dict, Any, Optional

# Add the parent directory to the path so we can import from SRAgent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SRAgent.agents.publications import create_publications_agent_stream, configure_logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure logging to suppress specific messages
configure_logging()

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
    # Set email and API key for Entrez
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
    # Initialize results list
    results = []
    
    # Process the DataFrame in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
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
        logger.info(f"Processed {min(i+batch_size, len(df))}/{len(df)} studies")
    
    # Create final DataFrame
    results_df = pd.DataFrame(results)
    
    # Save final results
    results_df.to_csv(output_file, index=False)
    
    return results_df

def main():
    """Main function to run the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process a DataFrame of accessions and find publications for each study.")
    parser.add_argument("input_file", help="Path to the input CSV file containing the accessions.")
    parser.add_argument("output_file", help="Path to save the results.")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of studies to process in parallel.")
    
    args = parser.parse_args()
    
    # Read the input DataFrame
    df = pd.read_csv(args.input_file)
    
    # Process the DataFrame
    asyncio.run(process_dataframe(df, args.output_file, args.batch_size))
    
    logger.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 