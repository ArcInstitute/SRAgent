#!/usr/bin/env python
"""
Command-line script to run evaluations for SRAgent.
"""

import os
import sys
import asyncio
import argparse
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from Bio import Entrez

# Add the SRAgent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import evaluation functions
from SRAgent.evals.publications_eval import evaluate_publications_agent, TEST_CASES

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_evaluations(args):
    """
    Run the specified evaluations.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        A dictionary containing the evaluation results.
    """
    results = {}
    
    # Run publications agent evaluation if specified
    if args.publications:
        logger.info("Running publications agent evaluation...")
        pub_results = await evaluate_publications_agent()
        results["publications"] = pub_results
        
        # Print summary
        print("\nPublications Agent Evaluation Summary:")
        print(f"Total test cases: {pub_results['summary']['total_test_cases']}")
        print(f"Successful test cases: {pub_results['summary']['successful_test_cases']}")
        print(f"Failed test cases: {pub_results['summary']['failed_test_cases']}")
    
    # Save results to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    return results

def main():
    """Parse command-line arguments and run evaluations."""
    # Load environment variables
    load_dotenv()
    
    # Set email and API key for Entrez
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run evaluations for SRAgent.')
    parser.add_argument('--publications', action='store_true', help='Run publications agent evaluation')
    parser.add_argument('--all', action='store_true', help='Run all evaluations')
    parser.add_argument('--output', type=str, help='Output file for evaluation results (JSON format)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # If --all is specified, run all evaluations
    if args.all:
        args.publications = True
    
    # If no evaluations are specified, run all
    if not any([args.publications]):
        logger.info("No specific evaluations specified, running all...")
        args.publications = True
    
    # Run evaluations
    results = asyncio.run(run_evaluations(args))
    
    # Determine exit code based on results
    success = True
    if "publications" in results:
        success = success and results["publications"]["summary"]["failed_test_cases"] == 0
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 