# import
## batteries
import os
import asyncio
from Bio import Entrez
from langchain_core.messages import HumanMessage
from SRAgent.cli.utils import CustomFormatter
from SRAgent.workflows.tissue_ontology import create_tissue_ontology_workflow
from SRAgent.agents.display import create_agent_stream, display_final_results

# functions
def tissue_ontology_parser(subparsers):
    help = 'Tissue Ontology: categorize tissue descriptions using the Uberon ontology.'
    desc = """
    # Example prompts:
    1. "Categorize the following tissue: brain"
    2. "What is the Uberon ID for hippocampus?"
    3. "Tissues: lung, heart, liver"
    4. "Find the ontology term for the thin layer of epithelial cells lining the alveoli in lungs"
    5. "What is the Uberon classification for skeletal muscle tissue?"
    """
    sub_parser = subparsers.add_parser(
        'tissue-ontology', help=help, description=desc, formatter_class=CustomFormatter
    )
    sub_parser.set_defaults(func=tissue_ontology_main)
    sub_parser.add_argument('prompt', type=str, help='Tissue description(s) to categorize')
    sub_parser.add_argument('--max-concurrency', type=int, default=3, 
                            help='Maximum number of concurrent processes')
    sub_parser.add_argument('--recursion-limit', type=int, default=40,
                            help='Maximum recursion limit')
    
def tissue_ontology_main(args):
    """
    Main function for invoking the tissue ontology workflow
    """
    # set email and api key
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")

    # invoke workflow with streaming
    config = {
        "max_concurrency": args.max_concurrency,
        "recursion_limit": args.recursion_limit
    }
    input = {"messages": [HumanMessage(content=args.prompt)]}
    results = asyncio.run(
        create_agent_stream(
            input, create_tissue_ontology_workflow, config, 
            summarize_steps=not args.no_summaries,
            no_progress=args.no_progress
        )
    )
    
    # Display final results with rich formatting
    display_final_results(results, "Uberon Tissue Classifications 🧠 🫁 🫀")

# main
if __name__ == '__main__':
    pass
