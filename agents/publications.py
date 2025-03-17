# import
## batteries
import os
import sys
import asyncio
import re
import logging
import json
from typing import Annotated, List, Dict, Any, Callable, Optional
## 3rd party
from Bio import Entrez
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
## package
from SRAgent.agents.esearch import create_esearch_agent
from SRAgent.agents.esummary import create_esummary_agent
from SRAgent.agents.efetch import create_efetch_agent
from SRAgent.agents.elink import create_elink_agent
from SRAgent.agents.utils import create_step_summary_chain
from SRAgent.tools.google_search import google_search_tool
from SRAgent.tools.pmid import pmcid_from_pmid, pmid_from_pmcid, get_publication_details
from SRAgent.tools.study_info import get_study_title_from_accession

# Configure logging to suppress specific messages
def configure_logging():
    """
    Configure logging to suppress specific log messages.
    """
    # Suppress httpx logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Suppress googleapiclient.discovery_cache logs
    logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)
    
    # Suppress other noisy loggers if needed
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

# functions
def create_publications_agent(
    model_name="gpt-4o",
    return_tool: bool=True,
) -> Callable:
    # Configure logging to suppress specific messages
    configure_logging()
    
    # create model
    model_supervisor = ChatOpenAI(model=model_name, temperature=0.1)

    # set tools
    tools = [
        create_esearch_agent(),
        create_esummary_agent(),
        create_efetch_agent(),
        create_elink_agent(),
        google_search_tool,
        pmcid_from_pmid,
        pmid_from_pmcid,
        get_publication_details,
        get_study_title_from_accession,
    ]
  
    # state modifier
    state_mod = "\n".join([
        "# Instructions",
        " - You are a helpful senior bioinformatician assisting a researcher with finding publications (or preprints) associated with study accessions.",
        " - You have a team of agents who can perform specific tasks using Entrez tools and Google search.",
        " - Your goal is to find the PMID and PMCID, OR (if not yet published on PubMed) the preprint DOI of publications associated with the given study accessions.",
        "# Strategies",
        " 1) try to find publications directly linked in GEO or SRA databases using elink.",
        " 2) If that doesn't work, try searching for the accession numbers on Google with quotes around them.",
        "     - Here, typically GSE IDs or E-MTAB IDs have higher success rates than SRP or PRJNA IDs, so try those first.",
        " 3) If directly Googling the accession numbers doesn't yield the publication you're looking for, then search for the study title on Google."
        "     - BE VERY CAREFUL -Using title search has a high chance of yielding publications totally unrelated to the SRA study.",
        "     - Use get_study_title_from_accession to get the study title.",
        "     - If searching using the title, you MUST verify that the authors and/or institution in the paper match those of the SRA study.",
        " 4) CRITICAL: If you find a PMCID but not a PMID, you MUST use the pmid_from_pmcid tool to get the corresponding PMID.",
        "     - This is MANDATORY - never return a result with only a PMCID without trying to get the PMID.",
        "     - Example: pmid_from_pmcid(pmcid='PMC10786309')",
        "     - ALWAYS include the PMID in your final response if you find it using pmid_from_pmcid.",
        "     - NEVER skip this step - it is essential for the evaluation to pass.",
        " 5) Similarly, if you find a PMID but not a PMCID, use the pmcid_from_pmid tool to get the corresponding PMCID if available.",
        " 6) Be EXTREMELY cautious about reporting preprints. Only report a preprint if you are VERY confident it is directly related to the accessions.",
        "    - The preprint should explicitly mention the accession numbers or have matching authors and study title.",
        "    - If there's any doubt, report that no publication was found rather than reporting a potentially incorrect preprint.",
        "# Multiple Accessions",
        " - When given multiple accession numbers, ALWAYS assume they are linked to the same publication and don't attempt to verify if they are related.",
        " - Use multiple accessions as different 'shots on goal' - try each one to find the publication.",
        " - Authors may refer to different accession numbers in their paper, so trying each one increases chances of finding the publication.",
        " - In general, if a GSE / E-MTAB accession is given, try that first before trying the SRP / PRJNA accession, since I have found that these IDs usually have higher success rates.",
        " - Once you find a publication using any of the accessions, stop searching and report it as the result for all accessions.",
        "# Multiple Publications",
        " - IMPORTANT: If you find multiple publications directly linked to the accessions through GEO/SRA/ArrayExpress, you MUST report this in your response.",
        " - When multiple publications are found, you need to select a primary publication for metadata curation.",
        " - Criteria for selecting the primary publication:",
        "   1. Choose the most comprehensive or detailed publication (usually the one with the most pages or in a higher impact journal)",
        "   2. Choose the most recent publication if they appear to be equally comprehensive",
        "   3. Choose the publication that most directly describes the dataset (rather than a review or secondary analysis)",
        " - In your response, clearly indicate that multiple publications were found and which one you've selected as primary.",
        " - List all publications with their PMIDs, PMCIDs, and titles.",
        " - ONLY report multiple publications when they are directly linked through databases, NOT when found through Google search.",
        "# Source Tracking",
        " - CRITICAL: You MUST explicitly state in your response how you found the publication using one of these exact phrases:",
        "   - 'SOURCE: DIRECT_LINK' - If found through direct links in GEO/SRA/ArrayExpress databases",
        "   - 'SOURCE: GOOGLE_SEARCH' - If found through Google search",
        "   - 'SOURCE: NOT_FOUND' - If no publication was found",
        " - This source information MUST be included in your message, preferably at the beginning or end.",
        " - This information is critical for the researcher to assess the reliability of the match.",
        "# Preprints",
        " - If a preprint best matches the publication you're looking for, report the preprint doi it as the result for all accessions.",
        "# Calling agents",
        " - Be sure to provide context to the agents (e.g., \"Use elink to find publications linked to SRP557106\").",
        " - Generally, you will want to specify the database(s) to search (e.g., sra, gds, or pubmed).",
        "# Conversion",
        " - Different accession types (SRP, PRJNA, GSE) may need different approaches.",
        " - For SRA accessions (SRP, PRJNA), use the sra database.",
        " - For GEO accessions (GSE), use the gds database.",
        "# Response Format",
        " - Your response MUST be a JSON-formatted dictionary with the following structure:",
        " - {",
        " -   \"pmid\": \"PMID_VALUE\",  # The PMID as a string, or null if not found",
        " -   \"pmcid\": \"PMCID_VALUE\",  # The PMCID as a string, or null if not found",
        " -   \"preprint_doi\": \"DOI_VALUE\",  # The preprint DOI as a string, or null if not found",
        " -   \"message\": \"YOUR_MESSAGE\"  # A brief message explaining your findings",
        " - }",
        " - Always include all keys in the dictionary, even if some values are null.",
        " - If you find a preprint (with DOI) but no published version in PubMed yet, it's acceptable to have null values for PMID and PMCID while providing the preprint_doi.",
        " - The message should be concise and provide only the relevant information.",
        " - When reporting results for multiple accessions, clearly state that the publication applies to all accessions.",
        " - REMEMBER to include the source information in your message using one of the exact phrases mentioned above.",
    ])

    # create agent
    agent = create_react_agent(
        model=model_supervisor,
        tools=tools,
        state_modifier=state_mod
    )

    # return agent instead of tool
    if not return_tool:
        return agent

    @tool
    async def invoke_publications_agent(
        message: Annotated[str, "Message to send to the Publications agent"],
        config: RunnableConfig,
    ) -> Annotated[dict, "Response from the Publications agent"]:
        """
        Invoke the Publications agent with a message.
        The Publications agent will find publications associated with study accessions.
        """
        # Invoke the agent with the message
        result = await agent.ainvoke(
            {"messages" : [AIMessage(content=message)]}, 
            config=config
        )
        return result
    
    return invoke_publications_agent

def extract_pmid_pmcid(result: str) -> tuple:
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

async def create_publications_agent_stream(input, config: dict={}, summarize_steps: bool=False) -> Dict[str, Any]:
    """
    Create a streaming version of the publications agent.
    
    Returns:
        A dictionary with the following structure:
        {
            "pmid": "PMID_VALUE",  # The PMID as a string, or null if not found
            "pmcid": "PMCID_VALUE",  # The PMCID as a string, or null if not found
            "title": "PUBLICATION_TITLE",  # The title of the publication, or null if not found
            "message": "YOUR_MESSAGE",  # A brief message explaining the findings
            "source": "SOURCE_TYPE",  # How the publication was found: "direct_link", "google_search", or "not_found"
            "multiple_publications": false,  # Whether multiple publications were found through direct links
            "all_publications": []  # List of all publications found if multiple_publications is true
        }
    """
    # Configure logging to suppress specific messages
    configure_logging()
    
    # create agent
    agent = create_publications_agent(return_tool=False)
    
    # create step summary chain
    step_summary_chain = create_step_summary_chain() if summarize_steps else None
    
    # invoke agent
    if summarize_steps and step_summary_chain:
        # If we want step summaries, we need to handle it differently
        # depending on the agent implementation
        try:
            # Try with step_callback parameter
            result = await agent.ainvoke(
                input,
                config=config,
                step_callback=step_summary_chain
            )
        except TypeError:
            # If step_callback is not supported, try without it
            result = await agent.ainvoke(
                input,
                config=config
            )
    else:
        # If we don't need step summaries, just invoke normally
        result = await agent.ainvoke(
            input,
            config=config
        )
    
    # Get the agent's response
    response_text = result["messages"][-1].content
    
    # Initialize source tracking and multiple publications flags
    source = "unknown"
    multiple_publications = False
    all_publications = []
    
    # Try to determine source from text
    if "SOURCE: DIRECT_LINK" in response_text:
        source = "direct_link"
    elif "SOURCE: GOOGLE_SEARCH" in response_text:
        source = "google_search"
    elif "SOURCE: NOT_FOUND" in response_text:
        source = "not_found"
    # Fallback to more general indicators if explicit ones aren't found
    elif "linked in GEO" in response_text or "linked in SRA" in response_text or "linked in ArrayExpress" in response_text or "direct link" in response_text or "elink" in response_text:
        source = "direct_link"
    elif "Google search" in response_text or "searched for" in response_text or "found through search" in response_text:
        source = "google_search"
    
    # Check for multiple publications
    if "multiple publications" in response_text.lower() or "several publications" in response_text.lower() or "found multiple" in response_text.lower():
        multiple_publications = True
        
        # Try to extract all publications mentioned
        # This is a simplified approach - in practice, you might need more sophisticated parsing
        publication_sections = re.split(r'Publication \d+:|Paper \d+:', response_text)
        if len(publication_sections) > 1:
            for section in publication_sections[1:]:  # Skip the first section which is intro text
                pub_pmid, pub_pmcid = extract_pmid_pmcid(section)
                pub_title = None
                title_match = re.search(r'titled\s+"([^"]+)"', section)
                if title_match:
                    pub_title = title_match.group(1)
                
                if pub_pmid or pub_pmcid:
                    all_publications.append({
                        "pmid": pub_pmid,
                        "pmcid": pub_pmcid,
                        "title": pub_title
                    })
    
    # Try to parse the response as JSON
    try:
        # Look for JSON-like content in the response
        json_match = re.search(r'({[\s\S]*})', response_text)
        if json_match:
            json_str = json_match.group(1)
            response_dict = json.loads(json_str)
            
            # Ensure all required keys are present
            required_keys = ["pmid", "pmcid", "preprint_doi", "message"]
            for key in required_keys:
                if key not in response_dict:
                    response_dict[key] = None
            
            # Add source tracking and multiple publications information
            response_dict["source"] = source
            response_dict["multiple_publications"] = multiple_publications
            response_dict["all_publications"] = all_publications
            
            # If PMCID is found but PMID is not, directly convert PMCID to PMID
            if response_dict["pmcid"] and not response_dict["pmid"]:
                try:
                    logging.info(f"Attempting to get PMID from PMCID: {response_dict['pmcid']}")
                    
                    # Use the existing pmid_from_pmcid function
                    from SRAgent.tools.pmid import pmid_from_pmcid
                    
                    pmcid = response_dict["pmcid"]
                    pmid = pmid_from_pmcid(pmcid)
                    
                    if pmid:
                        logging.info(f"Found PMID: {pmid} for PMCID: {pmcid}")
                        response_dict["pmid"] = pmid
                        response_dict["message"] += f" PMID: {pmid} was automatically retrieved from PMCID: {pmcid}."
                    else:
                        logging.warning(f"No PMID found for PMCID: {pmcid}")
                except Exception as e:
                    logging.warning(f"Failed to get PMID from PMCID: {e}")
                    logging.exception("Exception details:")
            
            return response_dict
    except Exception as e:
        # If JSON parsing fails, extract PMID and PMCID using regex
        logging.warning(f"Failed to parse response as JSON: {e}")
        
        # Extract PMID and PMCID using our helper function
        pmid, pmcid = extract_pmid_pmcid(response_text)
        
        # Extract title (if available)
        title = None
        title_match = re.search(r'titled\s+"([^"]+)"', response_text)
        if title_match:
            title = title_match.group(1)
        
        # Extract preprint DOI
        preprint_doi = None
        doi_match = re.search(r'DOI:?\s*(10\.\d+/[^\s\"\']+)', response_text, re.IGNORECASE)
        if doi_match:
            preprint_doi = doi_match.group(1)
            
        # Determine source from text
        source = "unknown"
        # Check for explicit source indicators
        if "SOURCE: DIRECT_LINK" in response_text:
            source = "direct_link"
        elif "SOURCE: GOOGLE_SEARCH" in response_text:
            source = "google_search"
        elif "SOURCE: NOT_FOUND" in response_text:
            source = "not_found"
        # Fallback to more general indicators if explicit ones aren't found
        elif "linked in GEO" in response_text or "linked in SRA" in response_text or "linked in ArrayExpress" in response_text or "direct link" in response_text or "elink" in response_text:
            source = "direct_link"
        elif "Google search" in response_text or "searched for" in response_text or "found through search" in response_text:
            source = "google_search"
            
        # Check for multiple publications
        multiple_publications = False
        all_publications = []
        if "multiple publications" in response_text.lower() or "several publications" in response_text.lower() or "found multiple" in response_text.lower():
            multiple_publications = True
            
            # Try to extract all publications mentioned
            publication_sections = re.split(r'Publication \d+:|Paper \d+:', response_text)
            if len(publication_sections) > 1:
                for section in publication_sections[1:]:  # Skip the first section which is intro text
                    pub_pmid, pub_pmcid = extract_pmid_pmcid(section)
                    pub_title = None
                    title_match = re.search(r'titled\s+"([^"]+)"', section)
                    if title_match:
                        pub_title = title_match.group(1)
                    
                    if pub_pmid or pub_pmcid:
                        all_publications.append({
                            "pmid": pub_pmid,
                            "pmcid": pub_pmcid,
                            "title": pub_title
                        })
        
        # If PMCID is found but PMID is not, directly convert PMCID to PMID
        if pmcid and not pmid:
            try:
                logging.info(f"Attempting to get PMID from PMCID: {pmcid}")
                
                # Use the existing pmid_from_pmcid function
                from SRAgent.tools.pmid import pmid_from_pmcid
                
                pmid = pmid_from_pmcid(pmcid)
                
                if pmid:
                    logging.info(f"Found PMID: {pmid} for PMCID: {pmcid}")
                    response_text += f" PMID: {pmid} was automatically retrieved from PMCID: {pmcid}."
                else:
                    logging.warning(f"No PMID found for PMCID: {pmcid}")
            except Exception as e:
                logging.warning(f"Failed to get PMID from PMCID: {e}")
                logging.exception("Exception details:")
        
        # Return structured dictionary
        result_dict = {
            "pmid": pmid,
            "pmcid": pmcid,
            "title": title,
            "preprint_doi": preprint_doi,
            "message": response_text,
            "source": source,
            "multiple_publications": multiple_publications,
            "all_publications": all_publications
        }
        
        return result_dict

# main
if __name__ == '__main__':
    # test
    async def main():
        # Configure logging
        configure_logging()
        
        # set email and api key
        Entrez.email = os.getenv("EMAIL")
        Entrez.api_key = os.getenv("NCBI_API_KEY")
        
        # invoke agent
        input = {"messages": [HumanMessage(content="Find publications for SRP557106")]}
        result = await create_publications_agent_stream(input)
        print(result)
    
    # run
    asyncio.run(main()) 