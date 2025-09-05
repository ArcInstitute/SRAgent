# import
## batteries
import os
import asyncio
from Bio import Entrez
from langchain_core.messages import HumanMessage
from SRAgent.cli.utils import CustomFormatter
from SRAgent.workflows.disease_ontology import create_disease_ontology_workflow
from SRAgent.workflows.graph_utils import handle_write_graph_option
from SRAgent.agents.display import create_agent_stream, display_final_results

# functions
def disease_ontology_parser(subparsers):
    help = 'Disease Ontology: categorize disease descriptions using the MONDO/PATO ontology.'
    desc = """
    # Example prompts:
    1. "Categorize the following disease: heart disorder"
    2. "What is the MONDO ontology ID for congestive heart failure?"
    3. "Diseases: heart neoplasm, bursitis"
    """
    sub_parser = subparsers.add_parser(
        'disease-ontology', help=help, description=desc, formatter_class=CustomFormatter
    )
    sub_parser.set_defaults(func=disease_ontology_main)
    sub_parser.add_argument('prompt', type=str, help='Disease description(s) to categorize')
    sub_parser.add_argument('--max-concurrency', type=int, default=3, 
                            help='Maximum number of concurrent processes')
    sub_parser.add_argument('--recursion-limit', type=int, default=40,
                            help='Maximum recursion limit')
    sub_parser.add_argument(
        '--tenant', type=str, default=os.getenv("DYNACONF", 'prod'),
        choices=['prod', 'test', 'claude'],
        help='Settings environment (also sets DYNACONF). Use "claude" to run with Claude defaults.'
    )
    sub_parser.add_argument(
        '--write-graph', type=str, metavar='FILE', default=None,
        help='Write the workflow graph to a file and exit (supports .png, .svg, .pdf, .mermaid formats)'
    )
    
def disease_ontology_main(args):
    """
    Main function for invoking the disease ontology workflow
    """
    # set tenant early so model selection reflects it
    if getattr(args, 'tenant', None):
        os.environ["DYNACONF"] = args.tenant
    # set email and api key
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
    # handle write-graph option
    if args.write_graph:
        handle_write_graph_option(create_disease_ontology_workflow, args.write_graph)
        return

    # invoke workflow with streaming
    config = {
        "max_concurrency": args.max_concurrency,
        "recursion_limit": args.recursion_limit
    }
    input = {"messages": [HumanMessage(content=args.prompt)]}
    results = asyncio.run(
        create_agent_stream(
            input, create_disease_ontology_workflow, config, 
            summarize_steps=not args.no_summaries,
            no_progress=args.no_progress
        )
    )
    
    # Display final results with rich formatting
    display_final_results(results, "🧬 Disease Ontologies 🧬")

# main
if __name__ == '__main__':
    pass
