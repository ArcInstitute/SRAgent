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
    
    # Try to parse the response as JSON
    try:
        # Look for JSON-like content in the response
        json_match = re.search(r'({[\s\S]*})', response_text)
        if json_match:
            json_str = json_match.group(1)
            response_dict = json.loads(json_str)
            
            # Ensure all required keys are present
            required_keys = ["pmid", "pmcid", "title", "message"]
            for key in required_keys:
                if key not in response_dict:
                    response_dict[key] = None
            
            # If PMCID is found but PMID is not, try to get PMID from PMCID
            if response_dict["pmcid"] and not response_dict["pmid"]:
                try:
                    logging.info(f"Attempting to get PMID from PMCID: {response_dict['pmcid']}")
                    from Bio import Entrez
                    Entrez.email = os.getenv("EMAIL", "user@example.com")
                    Entrez.api_key = os.getenv("NCBI_API_KEY")
                    
                    pmcid = response_dict["pmcid"]
                    # Ensure the PMCID has the 'PMC' prefix
                    if not pmcid.startswith('PMC'):
                        pmcid = f'PMC{pmcid}'
                    
                    logging.info(f"Using Entrez to convert PMCID {pmcid} to PMID")
                    # Use the Entrez elink utility to convert PMC ID to PMID
                    handle = Entrez.elink(dbfrom="pmc", db="pubmed", linkname="pmc_pubmed", id=pmcid.replace('PMC', ''))
                    record = Entrez.read(handle)
                    handle.close()
                    
                    logging.info(f"Entrez response: {record}")
                    
                    # Extract the PMID from the response
                    if record and record[0]['LinkSetDb'] and record[0]['LinkSetDb'][0]['Link']:
                        pmid = record[0]['LinkSetDb'][0]['Link'][0]['Id']
                        logging.info(f"Found PMID: {pmid} for PMCID: {pmcid}")
                        response_dict["pmid"] = pmid
                        response_dict["message"] += f" PMID: {pmid} was automatically retrieved from PMCID: {pmcid}."
                except Exception as e:
                    logging.warning(f"Failed to get PMID from PMCID: {e}")
                    logging.exception("Exception details:")
            
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
        
        # If PMCID is found but PMID is not, try to get PMID from PMCID
        if pmcid and not pmid:
            try:
                logging.info(f"Attempting to get PMID from PMCID: {pmcid}")
                from Bio import Entrez
                Entrez.email = os.getenv("EMAIL", "user@example.com")
                Entrez.api_key = os.getenv("NCBI_API_KEY")
                
                # Ensure the PMCID has the 'PMC' prefix
                if not pmcid.startswith('PMC'):
                    pmcid = f'PMC{pmcid}'
                
                logging.info(f"Using Entrez to convert PMCID {pmcid} to PMID")
                # Use the Entrez elink utility to convert PMC ID to PMID
                handle = Entrez.elink(dbfrom="pmc", db="pubmed", linkname="pmc_pubmed", id=pmcid.replace('PMC', ''))
                record = Entrez.read(handle)
                handle.close()
                
                logging.info(f"Entrez response: {record}")
                
                # Extract the PMID from the response
                if record and record[0]['LinkSetDb'] and record[0]['LinkSetDb'][0]['Link']:
                    pmid = record[0]['LinkSetDb'][0]['Link'][0]['Id']
                    logging.info(f"Found PMID: {pmid} for PMCID: {pmcid}")
                    response_text += f" PMID: {pmid} was automatically retrieved from PMCID: {pmcid}."
                else:
                    logging.warning(f"No PMID found for PMCID: {pmcid} in Entrez response")
            except Exception as e:
                logging.warning(f"Failed to get PMID from PMCID: {e}")
                logging.exception("Exception details:")
        
        # Extract title (if available)
        title = None
        title_match = re.search(r'titled\s+"([^"]+)"', response_text)
        if title_match:
            title = title_match.group(1)
        
        # Return structured dictionary
        return {
            "pmid": pmid,
            "pmcid": pmcid,
            "title": title,
            "message": response_text
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
        print(result)
    
    # run
    asyncio.run(main()) 