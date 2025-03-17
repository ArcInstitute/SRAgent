"""
Evaluation module for the publications agent.
This module contains functions to evaluate the performance of the publications agent.
"""

import os
import sys
import asyncio
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from Bio import Entrez
from langchain_core.messages import HumanMessage
import re

from SRAgent.agents.publications import create_publications_agent_stream

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PublicationTestCase:
    """Test case for the publications agent."""
    name: str
    accessions: List[str]  # List of accessions that should lead to the same publication
    expected_pmid: Optional[str] = None
    expected_pmcid: Optional[str] = None
    description: str = ""
    
    def __str__(self) -> str:
        """String representation of the test case."""
        return (
            f"Test Case: {self.name}\n"
            f"Accessions: {', '.join(self.accessions)}\n"
            f"Expected PMID: {self.expected_pmid or 'Not specified'}\n"
            f"Expected PMCID: {self.expected_pmcid or 'Not specified'}\n"
            f"Description: {self.description}"
        )

# Define test cases
TEST_CASES = [
    PublicationTestCase(
        name="SRP270870_PRJNA644744",
        accessions=["SRP270870", "PRJNA644744"],
        expected_pmid="36602862",
        expected_pmcid="PMC10014110",
        description="This study should be findable through Google search but not through SRA links."
    ),
    PublicationTestCase(
        name="ERP149679_PRJEB64504_E-MTAB-8142",
        accessions=["ERP149679", "PRJEB64504", "E-MTAB-8142"],
        expected_pmid="33479125",
        expected_pmcid="PMC7611557",
        description="Test with three different accession IDs from different repositories."
    ),
    # Add more test cases as needed
]

def extract_pmid_pmcid(result: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract PMID and PMCID from the agent's response.
    
    Args:
        result: The agent's response as a string.
        
    Returns:
        A tuple of (pmid, pmcid) extracted from the response, or (None, None) if not found.
    """
    pmid = None
    pmcid = None
    
    # Look for PMID in the response
    pmid_patterns = [
        r"PMID:?\s*(\d+)",
        r"PMID\s+(\d+)",
        r"PMID[:\s]*(\d+)",
        r"PubMed ID:?\s*(\d+)",
        r"PubMed\s+ID:?\s*(\d+)",
        r"\*\*PMID:\*\*\s*(\d+)",  # For markdown formatted output
        r"\*\*PMID\*\*:?\s*(\d+)",  # For markdown formatted output
        r"- \*\*PMID:\*\*\s*(\d+)",  # For markdown list items
        r"- \*\*PMID\*\*:?\s*(\d+)",  # For markdown list items
    ]
    
    for pattern in pmid_patterns:
        match = re.search(pattern, result, re.IGNORECASE)
        if match:
            pmid = match.group(1)
            break
    
    # Look for PMCID in the response
    pmcid_patterns = [
        r"PMCID:?\s*(PMC\d+)",
        r"PMCID\s+(PMC\d+)",
        r"PMCID[:\s]*(PMC\d+)",
        r"PMC:?\s*(\d+)",
        r"PMC\s+ID:?\s*(PMC\d+)",
        r"PMC\s+ID:?\s*(\d+)",
        r"\*\*PMCID:\*\*\s*(PMC\d+)",  # For markdown formatted output
        r"\*\*PMCID\*\*:?\s*(PMC\d+)",  # For markdown formatted output
        r"- \*\*PMCID:\*\*\s*(PMC\d+)",  # For markdown list items
        r"- \*\*PMCID\*\*:?\s*(PMC\d+)",  # For markdown list items
    ]
    
    for pattern in pmcid_patterns:
        match = re.search(pattern, result, re.IGNORECASE)
        if match:
            pmcid = match.group(1)
            # Add PMC prefix if it's just a number
            if not pmcid.startswith("PMC"):
                pmcid = f"PMC{pmcid}"
            break
    
    # If we still haven't found the PMID, try a more general approach
    if pmid is None:
        # Look for any number that appears after "PMID" in any format
        general_pmid_pattern = r"PMID.*?(\d+)"
        match = re.search(general_pmid_pattern, result, re.IGNORECASE)
        if match:
            pmid = match.group(1)
    
    # If we still haven't found the PMCID, try a more general approach
    if pmcid is None:
        # Look for PMC followed by numbers
        general_pmcid_pattern = r"PMC.*?(\d+)"
        match = re.search(general_pmcid_pattern, result, re.IGNORECASE)
        if match:
            pmcid = f"PMC{match.group(1)}"
    
    return pmid, pmcid

async def evaluate_single_test_case(test_case: PublicationTestCase) -> Dict[str, Any]:
    """
    Evaluate a single test case.
    
    Args:
        test_case: The test case to evaluate.
        
    Returns:
        A dictionary containing the evaluation results.
    """
    # Create a combined query with all accessions
    accessions_str = " and ".join(test_case.accessions)
    logger.info(f"Testing accessions together: {accessions_str}")
    
    # Create input message with all accessions, explicitly stating they are linked to the same publication
    input_message = {"messages": [HumanMessage(content=f"Find publications for {accessions_str}. These accessions are linked to the same publication.")]}
    
    try:
        # Run the agent
        start_time = asyncio.get_event_loop().time()
        result = await create_publications_agent_stream(input_message)
        end_time = asyncio.get_event_loop().time()
        
        # Get PMID and PMCID directly from the result dictionary
        pmid = result.get("pmid")
        pmcid = result.get("pmcid")
        
        # Check if the results match the expected values
        pmid_correct = pmid == test_case.expected_pmid if test_case.expected_pmid else pmid is not None
        pmcid_correct = pmcid == test_case.expected_pmcid if test_case.expected_pmcid else pmcid is not None
        
        # Store the results
        results = {
            "success": pmid_correct and pmcid_correct,
            "found_pmid": pmid,
            "found_pmcid": pmcid,
            "expected_pmid": test_case.expected_pmid,
            "expected_pmcid": test_case.expected_pmcid,
            "pmid_correct": pmid_correct,
            "pmcid_correct": pmcid_correct,
            "response": result.get("message", ""),
            "title": result.get("title"),
            "execution_time": end_time - start_time
        }
        
        logger.info(f"Results for {accessions_str}: PMID={pmid}, PMCID={pmcid}")
        logger.info(f"Success: {pmid_correct and pmcid_correct}")
        
    except Exception as e:
        logger.error(f"Error evaluating {accessions_str}: {e}")
        results = {
            "success": False,
            "error": str(e)
        }
    
    return results

async def evaluate_publications_agent(test_cases: List[PublicationTestCase] = None) -> Dict[str, Any]:
    """
    Evaluate the publications agent on a set of test cases.
    
    Args:
        test_cases: List of test cases to evaluate. If None, uses the default test cases.
        
    Returns:
        A dictionary containing the evaluation results.
    """
    # Set email and API key for Entrez
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
    # Use default test cases if none provided
    if test_cases is None:
        test_cases = TEST_CASES
    
    # Initialize results dictionary
    evaluation_results = {
        "test_cases": {},
        "summary": {
            "total_test_cases": len(test_cases),
            "successful_test_cases": 0,
            "failed_test_cases": 0
        }
    }
    
    # Evaluate each test case
    for test_case in test_cases:
        logger.info(f"Evaluating test case: {test_case.name}")
        
        # Evaluate the test case
        results = await evaluate_single_test_case(test_case)
        
        # Determine overall success for the test case
        success = results["success"]
        
        # Update summary
        if success:
            evaluation_results["summary"]["successful_test_cases"] += 1
        else:
            evaluation_results["summary"]["failed_test_cases"] += 1
        
        # Store the results
        evaluation_results["test_cases"][test_case.name] = {
            "test_case": {
                "name": test_case.name,
                "accessions": test_case.accessions,
                "expected_pmid": test_case.expected_pmid,
                "expected_pmcid": test_case.expected_pmcid,
                "description": test_case.description
            },
            "results": results,
            "success": success
        }
    
    return evaluation_results

async def main():
    """Run the evaluation and print the results."""
    # Set email and API key for Entrez
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
    # Run the evaluation
    results = await evaluate_publications_agent()
    
    # Print the results
    print(json.dumps(results, indent=2))
    
    # Print a summary
    print("\nSummary:")
    print(f"Total test cases: {results['summary']['total_test_cases']}")
    print(f"Successful test cases: {results['summary']['successful_test_cases']}")
    print(f"Failed test cases: {results['summary']['failed_test_cases']}")
    
    # Return success if all test cases passed
    return results['summary']['failed_test_cases'] == 0

if __name__ == "__main__":
    # Run the evaluation
    success = asyncio.run(main())
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 