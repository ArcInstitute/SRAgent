# import
## batteries
import os
import sys
import re
from typing import Annotated, Dict, Any, Optional
## 3rd party
from langchain_core.tools import tool
from Bio import Entrez
## package
from SRAgent.tools.utils import determine_database

@tool
def get_study_title_from_accession(
    accession: Annotated[str, "Accession number (e.g., SRP557106, PRJNA1210001, GSE63525)"]
) -> Annotated[str, "Study title and additional metadata"]:
    """
    Get the title and basic metadata of a study given its accession number.
    This is useful when you need to search for publications using the study title
    rather than the accession number.
    
    The function automatically determines the appropriate database (SRA, GEO, etc.)
    based on the accession format.
    """
    try:
        # Set email for Entrez
        if not Entrez.email:
            Entrez.email = os.getenv("EMAIL", "your.email@example.com")
        
        # Determine the database based on the accession format
        database = determine_database(accession)
        if not database:
            return f"Error: Could not determine database for accession {accession}"
        
        # For PRJNA accessions, try bioproject first, then sra if that fails
        original_database = database
        
        # Fetch the study information
        handle = Entrez.esearch(db=database, term=accession)
        record = Entrez.read(handle)
        handle.close()
        
        # If no results and it's a bioproject accession, try sra database
        if not record["IdList"] and database == "bioproject":
            database = "sra"
            handle = Entrez.esearch(db=database, term=accession)
            record = Entrez.read(handle)
            handle.close()
        
        if not record["IdList"]:
            # Try one more approach - for PRJNA accessions, try removing the PRJNA prefix
            if accession.startswith("PRJNA"):
                numeric_id = accession[5:]  # Remove "PRJNA"
                handle = Entrez.esearch(db="bioproject", term=numeric_id)
                record = Entrez.read(handle)
                handle.close()
                database = "bioproject"
                
                # If still no results, try sra with the numeric ID
                if not record["IdList"]:
                    handle = Entrez.esearch(db="sra", term=numeric_id)
                    record = Entrez.read(handle)
                    handle.close()
                    database = "sra"
            
            # If still no results, return error
            if not record["IdList"]:
                return f"Error: No records found for accession {accession} in database {original_database} or alternatives"
        
        # Get the summary of the study
        handle = Entrez.esummary(db=database, id=record["IdList"][0])
        summary = Entrez.read(handle)
        handle.close()
        
        # Extract relevant information based on the database
        if database == "sra":
            if isinstance(summary, dict) and "DocumentSummarySet" in summary:
                # Handle newer Entrez API format
                documents = summary["DocumentSummarySet"]["DocumentSummary"]
                if documents:
                    study_info = documents[0]
                    title = study_info.get("Title", "No title available")
                    exp_type = study_info.get("ExpType", "Unknown experiment type")
                    organism = study_info.get("Organism", "Unknown organism")
                    center = study_info.get("Center", "Unknown center")
                    
                    return (f"Study Title: {title}\n"
                            f"Experiment Type: {exp_type}\n"
                            f"Organism: {organism}\n"
                            f"Center/Institution: {center}\n"
                            f"Accession: {accession}")
            elif isinstance(summary, list) and len(summary) > 0:
                # Handle older Entrez API format
                study_info = summary[0]
                title = study_info.get("Title", "No title available")
                exp_type = study_info.get("ExpType", "Unknown experiment type")
                organism = study_info.get("Organism", "Unknown organism")
                center = study_info.get("Center", "Unknown center")
                
                return (f"Study Title: {title}\n"
                        f"Experiment Type: {exp_type}\n"
                        f"Organism: {organism}\n"
                        f"Center/Institution: {center}\n"
                        f"Accession: {accession}")
            else:
                return f"Error: Could not parse summary for accession {accession} in database {database}"
                
        elif database == "gds":
            if isinstance(summary, dict) and "DocumentSummarySet" in summary:
                # Handle newer Entrez API format
                documents = summary["DocumentSummarySet"]["DocumentSummary"]
                if documents:
                    study_info = documents[0]
                    title = study_info.get("title", "No title available")
                    summary_text = study_info.get("summary", "No summary available")
                    organism = study_info.get("taxon", "Unknown organism")
                    
                    return (f"Study Title: {title}\n"
                            f"Summary: {summary_text}\n"
                            f"Organism: {organism}\n"
                            f"Accession: {accession}")
            elif isinstance(summary, list) and len(summary) > 0:
                # Handle older Entrez API format
                study_info = summary[0]
                title = study_info.get("title", "No title available")
                summary_text = study_info.get("summary", "No summary available")
                organism = study_info.get("taxon", "Unknown organism")
                
                return (f"Study Title: {title}\n"
                        f"Summary: {summary_text}\n"
                        f"Organism: {organism}\n"
                        f"Accession: {accession}")
            else:
                return f"Error: Could not parse summary for accession {accession} in database {database}"
                
        elif database == "bioproject":
            if isinstance(summary, dict) and "DocumentSummarySet" in summary:
                # Handle newer Entrez API format
                documents = summary["DocumentSummarySet"]["DocumentSummary"]
                if documents:
                    study_info = documents[0]
                    title = study_info.get("Project_Title", "No title available")
                    description = study_info.get("Project_Description", "No description available")
                    org_name = study_info.get("Project_Organism_Name", "Unknown organism")
                    
                    return (f"Study Title: {title}\n"
                            f"Description: {description}\n"
                            f"Organism: {org_name}\n"
                            f"Accession: {accession}")
            elif isinstance(summary, list) and len(summary) > 0:
                # Handle older Entrez API format
                study_info = summary[0]
                title = study_info.get("Project_Title", "No title available")
                description = study_info.get("Project_Description", "No description available")
                org_name = study_info.get("Project_Organism_Name", "Unknown organism")
                
                return (f"Study Title: {title}\n"
                        f"Description: {description}\n"
                        f"Organism: {org_name}\n"
                        f"Accession: {accession}")
            else:
                return f"Error: Could not parse summary for accession {accession} in database {database}"
        
        else:
            # For other databases, try to extract common fields
            if isinstance(summary, dict) and "DocumentSummarySet" in summary:
                # Handle newer Entrez API format
                documents = summary["DocumentSummarySet"]["DocumentSummary"]
                if documents:
                    study_info = documents[0]
                    # Try different possible field names for title
                    title_fields = ["Title", "title", "Project_Title", "name", "Name"]
                    title = "No title available"
                    for field in title_fields:
                        if field in study_info:
                            title = study_info[field]
                            break
                    
                    return (f"Study Title: {title}\n"
                            f"Database: {database}\n"
                            f"Accession: {accession}")
            elif isinstance(summary, list) and len(summary) > 0:
                # Handle older Entrez API format
                study_info = summary[0]
                # Try different possible field names for title
                title_fields = ["Title", "title", "Project_Title", "name", "Name"]
                title = "No title available"
                for field in title_fields:
                    if field in study_info:
                        title = study_info[field]
                        break
                
                return (f"Study Title: {title}\n"
                        f"Database: {database}\n"
                        f"Accession: {accession}")
            else:
                return f"Error: Could not parse summary for accession {accession} in database {database}"
                
    except Exception as e:
        return f"Error retrieving study title: {str(e)}"

# Create a utility function to determine the database based on accession format
def determine_database(accession: str) -> Optional[str]:
    """
    Determine the appropriate Entrez database based on the accession format.
    """
    # SRA accessions
    if re.match(r"^SRR\d+$", accession) or re.match(r"^ERR\d+$", accession) or re.match(r"^DRR\d+$", accession):
        return "sra"
    elif re.match(r"^SRX\d+$", accession) or re.match(r"^ERX\d+$", accession) or re.match(r"^DRX\d+$", accession):
        return "sra"
    elif re.match(r"^SRP\d+$", accession) or re.match(r"^ERP\d+$", accession) or re.match(r"^DRP\d+$", accession):
        return "sra"
    elif re.match(r"^PRJNA\d+$", accession) or re.match(r"^PRJEB\d+$", accession) or re.match(r"^PRJDB\d+$", accession):
        return "bioproject"
    
    # GEO accessions
    elif re.match(r"^GSE\d+$", accession):
        return "gds"
    elif re.match(r"^GSM\d+$", accession):
        return "gds"
    elif re.match(r"^GPL\d+$", accession):
        return "gds"
    
    # If no match, try to guess based on prefix
    elif accession.startswith("SRA"):
        return "sra"
    elif accession.startswith("GEO"):
        return "gds"
    elif accession.startswith("PRJ"):
        return "bioproject"
    
    # Default to sra if we can't determine
    return "sra"

# For testing
if __name__ == "__main__":
    # Set email for Entrez
    Entrez.email = os.getenv("EMAIL", "your.email@example.com")
    
    # Test with different accessions
    test_accessions = ["SRP557106", "PRJNA1210001", "GSE63525"]
    for acc in test_accessions:
        print(f"\nTesting accession: {acc}")
        result = get_study_title_from_accession(acc)
        print(result) 