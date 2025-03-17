@tool
def pmid_from_pmcid(pmcid: Annotated[str, "PMC ID to convert to PMID"]) -> Annotated[str, "PMID corresponding to the PMC ID"]:
    """
    Convert a PMC ID to a PMID.
    
    Args:
        pmcid: The PMC ID to convert, e.g., 'PMC1234567' or '1234567'
    
    Returns:
        The PMID corresponding to the PMC ID, or an error message if not found.
    """
    # Ensure the PMCID has the 'PMC' prefix
    if not pmcid.startswith('PMC'):
        pmcid = f'PMC{pmcid}'
    
    try:
        # Use the Entrez elink utility to convert PMC ID to PMID
        handle = Entrez.elink(dbfrom="pmc", db="pubmed", linkname="pmc_pubmed", id=pmcid.replace('PMC', ''))
        record = Entrez.read(handle)
        handle.close()
        
        # Extract the PMID from the response
        if record and record[0]['LinkSetDb'] and record[0]['LinkSetDb'][0]['Link']:
            pmid = record[0]['LinkSetDb'][0]['Link'][0]['Id']
            return f"PMID {pmid} corresponds to {pmcid}"
        else:
            return f"No PMID found for {pmcid}"
    except Exception as e:
        return f"Error converting {pmcid} to PMID: {str(e)}" 