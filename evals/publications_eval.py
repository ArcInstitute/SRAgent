"""
Evaluation module for the publications agent.
This module contains functions to evaluate the performance of the publications agent.
"""

import os
import sys
import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from Bio import Entrez

# Add the parent directory to the path so we can import from SRAgent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SRAgent.agents.publications import create_publications_agent_stream, configure_logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure logging to suppress specific messages
configure_logging()

@dataclass
class PublicationTestCase:
    """Test case for the publications agent."""
    name: str
    accessions: List[str]  # List of accessions that should lead to the same publication
    expected_pmid: Optional[str] = None
    expected_pmcid: Optional[str] = None
    expected_preprint_doi: Optional[str] = None
    description: str = ""
    
    def __str__(self) -> str:
        """String representation of the test case."""
        return (
            f"Test Case: {self.name}\n"
            f"Accessions: {', '.join(self.accessions)}\n"
            f"Expected PMID: {self.expected_pmid or 'Not specified'}\n"
            f"Expected PMCID: {self.expected_pmcid or 'Not specified'}\n"
            f"Expected Preprint DOI: {self.expected_preprint_doi or 'Not specified'}\n"
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
        name="SRP559437_PRJNA1214776_GSE287827",
        accessions=["SRP559437", "PRJNA1214776", "GSE287827"],
        expected_preprint_doi="10.1101/2025.02.26.640382",
        description="This study has a bioRxiv preprint but no PubMed publication yet. It should be findable by searching the GEO title."
    ),
    PublicationTestCase(
        name="SRP557106_PRJNA1210001",
        accessions=["SRP557106", "PRJNA1210001"],
        expected_pmid=None,
        expected_pmcid=None,
        expected_preprint_doi=None,
        description="This study has no associated publication or preprint yet."
    ),
    PublicationTestCase(
        name="ERP156277_PRJEB71477_E-MTAB-13085",
        accessions=["ERP156277", "PRJEB71477", "E-MTAB-13085"],
        expected_pmid="38165934",
        expected_pmcid="PMC10786309",
        description="This study should be findable through Google search with the E-MTAB accession."
    ),
    PublicationTestCase(
        name="GSE188367_PRJNA778547_SRP344952",
        accessions=["GSE188367", "PRJNA778547", "SRP344952"],
        expected_pmid="35926182",
        expected_pmcid="PMC9894566",
        expected_preprint_doi=None,
        description="This study should be findable through GEO or SRA links."
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

def extract_preprint_doi(result: str) -> Optional[str]:
    """
    Extract preprint DOI from the agent's response.
    
    Args:
        result: The agent's response as a string.
        
    Returns:
        The preprint DOI extracted from the response, or None if not found.
    """
    preprint_doi = None
    
    # Look for DOI in the response
    doi_patterns = [
        r"DOI:?\s*(10\.\d+/[^\s\"\']+)",
        r"doi:?\s*(10\.\d+/[^\s\"\']+)",
        r"preprint DOI:?\s*(10\.\d+/[^\s\"\']+)",
        r"preprint doi:?\s*(10\.\d+/[^\s\"\']+)",
        r"\*\*DOI:\*\*\s*(10\.\d+/[^\s\"\']+)",  # For markdown formatted output
        r"\*\*DOI\*\*:?\s*(10\.\d+/[^\s\"\']+)",  # For markdown formatted output
        r"- \*\*DOI:\*\*\s*(10\.\d+/[^\s\"\']+)",  # For markdown list items
        r"- \*\*DOI\*\*:?\s*(10\.\d+/[^\s\"\']+)",  # For markdown list items
        r"preprint_doi[\"\':]?\s*[\"\'](10\.\d+/[^\s\"\']+)[\"\']\s*",  # For JSON formatted output
        r"\"preprint_doi\":\s*\"(10\.\d+/[^\s\"\']+)\"",  # For JSON formatted output
    ]
    
    for pattern in doi_patterns:
        match = re.search(pattern, result, re.IGNORECASE)
        if match:
            preprint_doi = match.group(1)
            break
    
    # If we still haven't found the DOI, try a more general approach
    if preprint_doi is None:
        # Look for any DOI pattern
        general_doi_pattern = r"10\.\d+/[^\s\"\']{4,}"
        match = re.search(general_doi_pattern, result)
        if match:
            preprint_doi = match.group(0)
    
    return preprint_doi

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
    input_message = {"messages": [{"role": "user", "content": f"Find publications for {accessions_str}. These accessions are linked to the same publication."}]}
    
    try:
        # Run the agent
        start_time = asyncio.get_event_loop().time()
        result = await create_publications_agent_stream(input_message)
        end_time = asyncio.get_event_loop().time()
        
        # Check if result is a dictionary (new format) or a string (old format)
        if isinstance(result, dict):
            # Extract values directly from the dictionary
            pmid = result.get("pmid")
            pmcid = result.get("pmcid")
            preprint_doi = result.get("preprint_doi")
            response_text = result.get("message", "")
            
            # Ensure response_text is a string
            if response_text is None:
                response_text = f"Found results: PMID={pmid}, PMCID={pmcid}, Preprint DOI={preprint_doi}"
                
            # If PMID is not found in the structured data, try to extract it from the response text
            if pmid is None:
                extracted_pmid, _ = extract_pmid_pmcid(response_text)
                if extracted_pmid:
                    pmid = extracted_pmid
                    logger.info(f"Extracted PMID {pmid} from response text")
        else:
            # Treat result as a string and extract values using regex
            response_text = result if result is not None else ""
            pmid, pmcid = extract_pmid_pmcid(response_text)
            preprint_doi = extract_preprint_doi(response_text)
        
        # Normalize DOIs by removing version suffixes (e.g., v1, v2)
        if preprint_doi:
            preprint_doi_normalized = re.sub(r'v\d+$', '', preprint_doi)
        else:
            preprint_doi_normalized = None
            
        if test_case.expected_preprint_doi:
            expected_preprint_doi_normalized = re.sub(r'v\d+$', '', test_case.expected_preprint_doi)
        else:
            expected_preprint_doi_normalized = None
        
        # Check if the results match the expected values
        # For cases where no publication is expected, all values should be None
        if test_case.expected_pmid is None and test_case.expected_pmcid is None and test_case.expected_preprint_doi is None:
            # Success if no publication or preprint was found
            pmid_correct = pmid is None
            pmcid_correct = pmcid is None
            preprint_doi_correct = preprint_doi is None
        else:
            # For cases where a publication or preprint is expected
            pmid_correct = pmid == test_case.expected_pmid if test_case.expected_pmid else True
            pmcid_correct = pmcid == test_case.expected_pmcid if test_case.expected_pmcid else True
            preprint_doi_correct = preprint_doi_normalized == expected_preprint_doi_normalized if expected_preprint_doi_normalized else True
        
        # Store the results
        results = {
            "success": pmid_correct and pmcid_correct and preprint_doi_correct,
            "found_pmid": pmid,
            "found_pmcid": pmcid,
            "found_preprint_doi": preprint_doi,
            "found_preprint_doi_normalized": preprint_doi_normalized,
            "expected_pmid": test_case.expected_pmid,
            "expected_pmcid": test_case.expected_pmcid,
            "expected_preprint_doi": test_case.expected_preprint_doi,
            "expected_preprint_doi_normalized": expected_preprint_doi_normalized,
            "pmid_correct": pmid_correct,
            "pmcid_correct": pmcid_correct,
            "preprint_doi_correct": preprint_doi_correct,
            "response": response_text,
            "execution_time": end_time - start_time
        }
        
        logger.info(f"Results for {accessions_str}: PMID={pmid}, PMCID={pmcid}, Preprint DOI={preprint_doi}")
        logger.info(f"Success: {pmid_correct and pmcid_correct and preprint_doi_correct}")
        
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
    # Configure logging
    configure_logging()
    
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
        success = results.get("success", False)
        
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
                "expected_preprint_doi": test_case.expected_preprint_doi,
                "description": test_case.description
            },
            "results": results,
            "success": success
        }
    
    return evaluation_results

async def main():
    """Run the evaluation and print the results."""
    # Configure logging
    configure_logging()
    
    # Set email and API key for Entrez
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
    # Check if a specific test case was requested
    if len(sys.argv) > 1:
        test_case_name = sys.argv[1]
        print(f"Command-line argument received: {test_case_name}")
        
        # Find the requested test case
        selected_test_cases = [tc for tc in TEST_CASES if tc.name == test_case_name]
        print(f"Found {len(selected_test_cases)} matching test cases")
        
        if not selected_test_cases:
            print(f"Test case '{test_case_name}' not found. Available test cases:")
            for tc in TEST_CASES:
                print(f"  - {tc.name}")
            return False
        
        print(f"Running only test case: {test_case_name}")
        results = await evaluate_publications_agent(selected_test_cases)
    else:
        # Run all test cases
        print("Running all test cases")
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
    # Configure logging
    configure_logging()
    
    # Run the evaluation
    success = asyncio.run(main())
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 