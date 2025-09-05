# import
## batteries
import os
import asyncio

## 3rd party
import pandas as pd
from Bio import Entrez
from langchain_core.messages import HumanMessage

## package
from SRAgent.cli.utils import CustomFormatter
from SRAgent.workflows.metadata import get_metadata_items, create_metadata_graph
from SRAgent.workflows.graph_utils import handle_write_graph_option
from SRAgent.agents.display import create_step_summary_chain


# functions
def metadata_agent_parser(subparsers):
    help = "Metadata Agent: Obtain metadata for specific SRX accessions"
    desc = """
    """
    sub_parser = subparsers.add_parser(
        "metadata", help=help, description=desc, formatter_class=CustomFormatter
    )
    sub_parser.set_defaults(func=metadata_agent_main)
    sub_parser.add_argument(
        "srx_accession_csv",
        type=str,
        help="CSV of entrez_id,srx_accession. Headers required",
    )
    sub_parser.add_argument(
        "--database",
        type=str,
        default="sra",
        choices=["gds", "sra"],
        help="Entrez database origin of the Entrez IDs",
    )
    sub_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=6,
        help="Maximum number of concurrent processes",
    )
    sub_parser.add_argument(
        "--recursion-limit", type=int, default=200, help="Maximum recursion limit"
    )
    sub_parser.add_argument(
        "--max-parallel",
        type=int,
        default=2,
        help="Maximum parallel processing of SRX accessions",
    )
    sub_parser.add_argument(
        "--no-srr",
        action="store_true",
        default=False,
        help="Do NOT upload SRR accessions to scBaseCount SQL database",
    )
    sub_parser.add_argument(
        "--use-database",
        action="store_true",
        default=False,
        help="Add the results to the scBaseCount SQL database",
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


async def _process_single_srx(
    entrez_srx, database, graph, step_summary_chain, config: dict, no_summaries: bool
):
    """Process a single entrez_id"""
    # format input for the graph
    metadata_items = "\n".join([f" - {x}" for x in get_metadata_items().values()])
    prompt = "\n".join(
        [
            "# Instructions",
            "For the SRA experiment accession {SRX_accession}, find the following dataset metadata:",
            metadata_items,
            "# Notes",
            " - Try to confirm any questionable metadata values with two data sources",
        ]
    )
    input = {
        "entrez_id": entrez_srx[0],
        "SRX": entrez_srx[1],
        "database": database,
        "messages": [HumanMessage(prompt.format(SRX_accession=entrez_srx[1]))],
    }

    # call the graph
    final_state = None
    i = 0
    async for step in graph.astream(input, config=config):
        i += 1
        final_state = step
        if no_summaries:
            nodes = ",".join(list(step.keys()))
            print(f"[{entrez_srx[0]}] Step {i}: {nodes}")
        else:
            msg = await step_summary_chain.ainvoke({"step": step})
            print(f"[{entrez_srx[0]}] Step {i}: {msg.content}")

    if final_state:
        print(f"#-- Final results for Entrez ID {entrez_srx[0]} --#")
        try:
            print(final_state["final_state_node"]["messages"][-1].content)
        except KeyError:
            print("Processing skipped")
    print("#---------------------------------------------#")


async def _metadata_agent_main(args):
    """
    Main function for invoking the metadata agent
    """
    # set tenant
    if args.tenant:
        os.environ["DYNACONF"] = args.tenant

    # set email and api key
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")

    # handle write-graph option
    if args.write_graph:
        handle_write_graph_option(create_metadata_graph, args.write_graph)
        return

    # create supervisor agent
    graph = create_metadata_graph()
    step_summary_chain = create_step_summary_chain()

    # invoke agent
    config = {
        "max_concurrency": args.max_concurrency,
        "recursion_limit": args.recursion_limit,
        "configurable": {"use_database": args.use_database, "no_srr": args.no_srr},
    }

    # read in entrez_id and srx_accession
    entrez_srx = pd.read_csv(args.srx_accession_csv, comment="#").to_records(
        index=False
    )

    # Create semaphore to limit concurrent processing
    semaphore = asyncio.Semaphore(args.max_parallel)

    async def _process_with_semaphore(entrez_id):
        async with semaphore:
            await _process_single_srx(
                entrez_id,
                args.database,
                graph,
                step_summary_chain,
                config,
                args.no_summaries,
            )

    # Create tasks for each entrez_id
    tasks = [_process_with_semaphore(x) for x in entrez_srx]

    # Run tasks concurrently with limited concurrency
    await asyncio.gather(*tasks)


def metadata_agent_main(args):
    asyncio.run(_metadata_agent_main(args))


# main
if __name__ == "__main__":
    pass
