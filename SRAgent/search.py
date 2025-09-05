# import
from datetime import datetime


def construct_query(
    start_date: str, end_date: str, search_terms: list = None, organism: str = "human"
) -> str:
    """
    Create an Entrez query string with date range and optional search term.
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        search_term: Optional search term
    Returns:
        Entrez query string
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_range = f"{start.strftime('%Y/%m/%d')}:{end.strftime('%Y/%m/%d')}[PDAT]"

    if search_terms is None:
        search_terms = [
            "single cell RNA sequencing",
            "single cell RNA-seq",
            "single cell transcriptomics",
        ]

    if search_terms:
        # add quotes to search terms
        search_terms = [f'"{term}"' for term in search_terms]
        search_terms = " OR ".join(search_terms)
        query = f"({search_terms}) AND {date_range}"
    else:
        query = date_range
    query += f' AND "{organism}"[Organism]'
    return query
