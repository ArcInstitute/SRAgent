# import
## batteries
import os
import asyncio
from Bio import Entrez
from langchain_core.messages import HumanMessage
from SRAgent.cli.utils import CustomFormatter
from SRAgent.agents.entrez import create_entrez_agent
from SRAgent.workflows.graph_utils import handle_write_graph_option
from SRAgent.agents.display import create_agent_stream, display_final_results


# functions
def entrez_agent_parser(subparsers):
    help = "Entrez Agent: general agent for working with Entrez databases."
    desc = """
    # Example prompts:
    1. "Convert GSE121737 to SRX accessions"
    2. "Obtain any available publications for GSE196830"
    3. "Obtain the SRR accessions for SRX4967527"
    4. "Is SRR8147022 paired-end Illumina data?"
    5. "Is SRP309720 10X Genomics data?"
    """
    sub_parser = subparsers.add_parser(
        "entrez", help=help, description=desc, formatter_class=CustomFormatter
    )
    sub_parser.set_defaults(func=entrez_agent_main)
    sub_parser.add_argument("prompt", type=str, help="Prompt for the agent")
    sub_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="Maximum number of concurrent processes",
    )
    sub_parser.add_argument(
        "--recursion-limit", type=int, default=40, help="Maximum recursion limit"
    )
    sub_parser.add_argument(
        "--tenant",
        type=str,
        default=os.getenv("DYNACONF", "prod"),
        choices=["prod", "test", "claude"],
        help='Settings environment (also sets DYNACONF). Use "claude" to run with Claude defaults.',
    )
    sub_parser.add_argument(
        "--write-graph",
        type=str,
        metavar="FILE",
        default=None,
        help="Write the workflow graph to a file and exit (supports .png, .svg, .pdf, .mermaid formats)",
    )


def entrez_agent_main(args):
    """
    Main function for invoking the entrez agent
    """
    # set tenant early so model selection reflects it
    if getattr(args, "tenant", None):
        os.environ["DYNACONF"] = args.tenant

    # set email and api key
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")

    # handle write-graph option
    if args.write_graph:
        handle_write_graph_option(create_entrez_agent, args.write_graph)
        return

    # invoke agent with streaming
    config = {
        "max_concurrency": args.max_concurrency,
        "recursion_limit": args.recursion_limit,
    }
    input = {"messages": [HumanMessage(content=args.prompt)]}
    results = asyncio.run(
        create_agent_stream(
            input,
            create_entrez_agent,
            config,
            summarize_steps=not args.no_summaries,
            no_progress=args.no_progress,
        )
    )

    # Display final results with rich formatting
    display_final_results(results)


# main
if __name__ == "__main__":
    pass
