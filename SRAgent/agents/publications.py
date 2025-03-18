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
from SRAgent.tools.pmid import pmcid_from_pmid, pmid_from_pmcid, get_publication_details, pmid_from_title_tool
from SRAgent.tools.study_info import get_study_title_from_accession
from SRAgent.tools.geo import get_pmid_from_geo

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
        pmid_from_title_tool,
        get_pmid_from_geo,
    ]
  
    # state modifier
    state_mod = "\n".join([
        "# Instructions",
        " - You are a helpful senior bioinformatician assisting a researcher with finding publications (or preprints) associated with study accessions.",
        " - You have a team of agents who can perform specific tasks using Entrez tools and Google search.",
        " - Your goal is to find the PMID and PMCID, OR (if not yet published on PubMed) the preprint DOI of publications associated with the given study accessions.",
        "# Strategies",
        " 1) For GSE accessions, ALWAYS try to extract the PMID directly from the GEO page first using get_pmid_from_geo. This is the most reliable method for GSE accessions.",
        " 2) If direct GEO PMID retrieval doesn't work, try to find publications directly linked in GEO/ArrayExpress or SRA databases using elink.",
        " 3) If that doesn't work, try searching for the accession numbers on Google with quotes around them.",
        "     - Here, typically GSE IDs or E-MTAB IDs have higher success rates than SRP or PRJNA IDs, so try those first.",
        " 4) If directly Googling the accession numbers doesn't yield the publication you're looking for, then search for the study title on Google."
        "     - BE VERY CAREFUL -Using title search has a high chance of yielding publications totally unrelated to the SRA study.",
        "     - Use get_study_title_from_accession to get the study title.",
        "     - If searching using the title, you MUST verify that the authors and/or institution in the paper match those of the SRA study.",
        " 5) IMPORTANT: If you find a publication title through any method (Google search, database, etc.) but do not have a PMID or PMCID, you MUST use the pmid_from_title_tool to search for the publication in PubMed.",
        "     - NEVER tell the user to check a journal website or database themselves.",
        "     - ALWAYS try to find the PMID yourself using the title.",
        "     - This is critical for providing complete information to the user.",
        " 6) Once you find a PMID, you can use the pmcid_from_pmid tool or pmid_from_pmcid tool to get the corresponding PMCID if available.",
        " 7) Similarly, if you find a PMCID only, use the  tool to get the corresponding PMID.",
        "# Multiple Accessions",
        " - When given multiple accession numbers, ALWAYS assume they are linked to the same publication and don't attempt to verify if they are related.",
        " - Use multiple accessions as different 'shots on goal' - try each one to find the publication.",
        " - Authors may refer to different accession numbers in their paper, so trying each one increases chances of finding the publication.",
        " - In general, if a GSE / E-MTAB accession is given, try that first before trying the SRP / PRJNA accession, since I have found that these IDs usually have higher success rates.",
        " - Once you find a publication using any of the accessions, stop searching and report it as the result for all accessions.",
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
        " - If you find a publication title but no PMID or PMCID, you MUST use the pmid_from_title_tool to search for the publication in PubMed before responding.",
        " - NEVER return a response with null PMID/PMCID values when you have found a publication title, unless you've conclusively proven it doesn't exist in PubMed.",
        " - It is NOT acceptable to tell the user to check a journal website or database. You must provide complete information.",
        " - The message should be concise and provide only the relevant information.",
        " - When reporting results for multiple accessions, clearly state that the publication applies to all accessions.",
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

async def create_publications_agent_stream(input, config: dict={}, summarize_steps: bool=False) -> Dict[str, Any]:
    """
    Create a streaming version of the publications agent.
    
    Returns:
        A dictionary with the following structure:
        {
            "pmid": "PMID_VALUE",  # The PMID as a string, or null if not found
            "pmcid": "PMCID_VALUE",  # The PMCID as a string, or null if not found
            "title": "PUBLICATION_TITLE",  # The title of the publication, or null if not found
            "message": "YOUR_MESSAGE"  # A brief message explaining the findings
        }
    """
    # Configure logging to suppress specific messages
    configure_logging()
    
    # create agent
    agent = create_publications_agent(return_tool=False)
    
    # create step summary chain
    step_summary_chain = create_step_summary_chain() if summarize_steps else None
    
    try:
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
    
        # Try to parse the response as JSON
        try:
            # Look for JSON-like content in the response
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                response_dict = json.loads(json_str)
                
                # Ensure all required keys are present
                required_keys = ["pmid", "pmcid", "preprint_doi", "title", "message"]
                for key in required_keys:
                    if key not in response_dict:
                        response_dict[key] = None
                
                # Add source tracking and multiple publications information
                response_dict["source"] = source
                response_dict["multiple_publications"] = multiple_publications
                response_dict["all_publications"] = all_publications
                
                return response_dict
        except Exception as e:
            # If JSON parsing fails, extract PMID and PMCID using regex
            logging.warning(f"Failed to parse response as JSON: {e}")
        
        # Extract PMID
        pmid = None
        pmid_patterns = [
            r"PMID:?\s*(\d+)",
            r"PMID\s+(\d+)",
            r"PMID[:\s]*(\d+)",
            r"PubMed ID:?\s*(\d+)",
            r"PubMed\s+ID:?\s*(\d+)",
            r"\*\*PMID:\*\*\s*(\d+)",
            r"\*\*PMID\*\*:?\s*(\d+)",
            r"- \*\*PMID:\*\*\s*(\d+)",
            r"- \*\*PMID\*\*:?\s*(\d+)",
        ]
        
        for pattern in pmid_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                pmid = match.group(1)
                break
        
        # If we still haven't found the PMID, try a more general approach
        if pmid is None:
            general_pmid_pattern = r"PMID.*?(\d+)"
            match = re.search(general_pmid_pattern, response_text, re.IGNORECASE)
            if match:
                pmid = match.group(1)
        
        # Extract PMCID
        pmcid = None
        pmcid_patterns = [
            r"PMCID:?\s*(PMC\d+)",
            r"PMCID\s+(PMC\d+)",
            r"PMCID[:\s]*(PMC\d+)",
            r"PMC:?\s*(\d+)",
            r"PMC\s+ID:?\s*(PMC\d+)",
            r"PMC\s+ID:?\s*(\d+)",
            r"\*\*PMCID:\*\*\s*(PMC\d+)",
            r"\*\*PMCID\*\*:?\s*(PMC\d+)",
            r"- \*\*PMCID:\*\*\s*(PMC\d+)",
            r"- \*\*PMCID\*\*:?\s*(PMC\d+)",
        ]
        
        for pattern in pmcid_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                pmcid = match.group(1)
                # Add PMC prefix if it's just a number
                if not pmcid.startswith("PMC"):
                    pmcid = f"PMC{pmcid}"
                break
        
        # If we still haven't found the PMCID, try a more general approach
        if pmcid is None:
            general_pmcid_pattern = r"PMC.*?(\d+)"
            match = re.search(general_pmcid_pattern, response_text, re.IGNORECASE)
            if match:
                pmcid = f"PMC{match.group(1)}"
        
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
        
        # Return structured dictionary
        return {
            "pmid": pmid,
            "pmcid": pmcid,
            "preprint_doi": preprint_doi,
            "title": title,
            "message": response_text,
            "source": source,
            "multiple_publications": multiple_publications,
            "all_publications": all_publications
        }
    except Exception as e:
        logging.error(f"Error in create_publications_agent_stream: {e}")
        # Return a default dictionary in case of errors
        return {
            "pmid": None,
            "pmcid": None,
            "preprint_doi": None, 
            "title": None,
            "message": f"Error processing publication data: {str(e)}",
            "source": "error",
            "multiple_publications": False,
            "all_publications": []
        }

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
        print(json.dumps(result, indent=2))
    
    # run
    asyncio.run(main()) 