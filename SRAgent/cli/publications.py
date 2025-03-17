# import
## batteries
import os
import asyncio
from Bio import Entrez
from langchain_core.messages import HumanMessage
from SRAgent.cli.utils import CustomFormatter
from SRAgent.agents.publications import create_publications_agent
from SRAgent.agents.utils import create_agent_stream

# functions
def publications_agent_parser(subparsers):
    help = 'Publications Agent: find publications associated with study accessions.'
    desc = """
    # Example prompts:
    1. "Find publications for SRP557106"
    2. "Find publications for PRJNA1210001"
    3. "Find publications for GSE196830"
    4. "Find publications for SRP557106, PRJNA1210001, and GSE196830"
    """
    sub_parser = subparsers.add_parser(
        'publications', help=help, description=desc, formatter_class=CustomFormatter
    )
    sub_parser.set_defaults(func=publications_agent_main)
    sub_parser.add_argument('prompt', type=str, help='Prompt for the agent')    
    sub_parser.add_argument('--no-summaries', action='store_true', default=False,
                            help='No LLM summaries')
    sub_parser.add_argument('--max-concurrency', type=int, default=3, 
                            help='Maximum number of concurrent processes')
    sub_parser.add_argument('--recursion-limit', type=int, default=40,
                            help='Maximum recursion limit')
    
def publications_agent_main(args):
    """
    Main function for invoking the publications agent
    """
    # set email and api key
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")

    # invoke agent with streaming
    config = {
        "max_concurrency" : args.max_concurrency,
        "recursion_limit": args.recursion_limit
    }
    input = {"messages": [HumanMessage(content=args.prompt)]}
    results = asyncio.run(
        create_agent_stream(
            input, create_publications_agent, config, summarize_steps=not args.no_summaries
        )
    )
    print(results)
            

# main
if __name__ == '__main__':
    pass 