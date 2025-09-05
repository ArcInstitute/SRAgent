# import
## batteries
import os
import asyncio
from Bio import Entrez
from langchain_core.messages import HumanMessage
from SRAgent.cli.utils import CustomFormatter
from SRAgent.agents.sragent import create_sragent_agent
from SRAgent.workflows.graph_utils import handle_write_graph_option
from SRAgent.agents.display import create_agent_stream, display_final_results

# functions
def sragent_parser(subparsers):
    help = 'SRAgent: high-level agent for working with sequence data.'
    desc = """
    # Example prompts:
    1. "Convert GSE121737 to SRX accessions"
    2. "Is SRX25994842 Illumina sequence data, 10X Genomics data, and which organism?"
    3. "List the collaborators for the SRX20554853 dataset"
    4. "Obtain all SRR accessions for SRX20554853"
    5. "Is SRP309720 paired-end sequencing data?"
    """
    sub_parser = subparsers.add_parser(
        'sragent', help=help, description=desc, formatter_class=CustomFormatter
    )
    sub_parser.set_defaults(func=sragent_main)
    sub_parser.add_argument('prompt', type=str, help='Prompt for the agent') 
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

def sragent_main(args):
    """
    Main function for invoking the sragent agent
    """
    # set tenant early so model selection reflects it
    if getattr(args, 'tenant', None):
        os.environ["DYNACONF"] = args.tenant
    # set email and api key
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
    # handle write-graph option
    if args.write_graph:
        handle_write_graph_option(create_sragent_agent, args.write_graph)
        return

    # invoke agent with streaming
    config = {
        "max_concurrency" : args.max_concurrency,
        "recursion_limit": args.recursion_limit
    }
    input = {"messages": [HumanMessage(content=args.prompt)]}
    results = asyncio.run(
        create_agent_stream(
            input, create_sragent_agent, config, 
            summarize_steps=not args.no_summaries,
            no_progress=args.no_progress
        )
    )
    
    # Display final results with rich formatting
    display_final_results(results)

# main
if __name__ == '__main__':
    pass
