# import
## batteries
import os
import sys
import asyncio
import argparse
from typing import List
from datetime import datetime, timedelta

## 3rd party
from Bio import Entrez
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

## package
from SRAgent.cli.utils import CustomFormatter
from SRAgent.workflows.find_datasets import create_find_datasets_graph
from SRAgent.workflows.graph_utils import handle_write_graph_option
from SRAgent.agents.display import create_step_summary_chain
from SRAgent.tools.utils import set_entrez_access
from SRAgent.organisms import OrganismEnum

# organism parameters
human_mouse = [OrganismEnum.HUMAN.name.lower(), OrganismEnum.MOUSE.name.lower()]
other_orgs = [
    org.name.lower().replace("_", " ")
    for org in OrganismEnum
    if org not in [OrganismEnum.HUMAN, OrganismEnum.MOUSE]
]


# functions
def find_datasets_parser(subparsers):
    help = "Obtain datasets and process each via the srx-info workflow"
    desc = """
    Example message: "Obtain recent single cell RNA-seq datasets in the SRA database"
    """
    sub_parser = subparsers.add_parser(
        "find-datasets", help=help, description=desc, formatter_class=CustomFormatter
    )
    sub_parser.set_defaults(func=find_datasets_main)
    sub_parser.add_argument(
        "message", type=str, help="Message to instruct the agent. See the Description"
    )
    sub_parser.add_argument(
        "--max-datasets",
        type=int,
        default=5,
        help="Maximum number of datasets to find and process",
    )
    sub_parser.add_argument(
        "--min-date",
        type=str,
        default=(datetime.now() - timedelta(days=365 * 10)).strftime("%Y/%m/%d"),
        help="Oldest date to search for datasets",
    )
    sub_parser.add_argument(
        "--max-date",
        type=str,
        default=datetime.now().strftime("%Y/%m/%d"),
        help="Newest date to search for datasets",
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
        "-o",
        "--organisms",
        type=str,
        nargs="+",
        default=human_mouse,
        choices=human_mouse + other_orgs + ["human-mouse", "other-orgs"],
        help='Organisms to search for. Use "human-mouse" or "other-orgs" to select human/mouse or all other organisms',
    )
    sub_parser.add_argument(
        "--use-database",
        action="store_true",
        default=False,
        help="Use the scBaseCount database to screen out existing datasets for adding the newly found datasets",
    )
    sub_parser.add_argument(
        "--tenant",
        type=str,
        default=os.getenv("DYNACONF", "prod"),
        choices=["prod", "test", "claude"],
        help='Settings environment (also sets DYNACONF). Use "claude" to run with Claude defaults.',
    )
    sub_parser.add_argument(
        "--reprocess-existing",
        action="store_true",
        default=False,
        help="Reprocess existing Entrez IDs in the scBaseCount database instead of ignoring existing (assuming --use-database)",
    )
    sub_parser.add_argument(
        "--write-graph",
        type=str,
        metavar="FILE",
        default=None,
        help="Write the workflow graph to a file and exit (supports .png, .svg, .pdf, .mermaid formats)",
    )


async def _find_datasets_main(args):
    """
    Main function for invoking the find-datasets workflow
    """
    # set tenant
    if args.tenant:
        os.environ["DYNACONF"] = args.tenant

    # set email and api key
    set_entrez_access()

    # handle write-graph option
    if args.write_graph:
        handle_write_graph_option(create_find_datasets_graph, args.write_graph)
        return

    # create supervisor agent
    graph = create_find_datasets_graph()

    if not args.no_summaries:
        step_summary_chain = create_step_summary_chain()

    # format organisms
    if "human-mouse" in args.organisms:
        args.organisms.remove("human-mouse")
        args.organisms += human_mouse
    if "other-orgs" in args.organisms:
        args.organisms.remove("other-orgs")
        args.organisms += other_orgs
    args.organisms = sorted(list(set(args.organisms)))

    # set graph config
    config = {
        "max_concurrency": args.max_concurrency,
        "recursion_limit": args.recursion_limit,
        "configurable": {
            "organisms": args.organisms,
            "max_datasets": args.max_datasets,
            "use_database": args.use_database,
            "reprocess_existing": args.reprocess_existing,
            "min_date": args.min_date,
            "max_date": args.max_date,
        },
    }
    input = {"messages": [HumanMessage(content=args.message)]}

    # call the graph
    final_state = None
    i = 0
    async for step in graph.astream(input, config=config):
        i += 1
        final_state = step
        if args.no_summaries:
            nodes = ",".join(list(step.keys()))
            print(f"Step {i}: {nodes}")
        else:
            msg = await step_summary_chain.ainvoke({"step": step})
            print(f"Step {i}: {msg.content}")

    # print final results
    if final_state:
        print(f"\n#-- Final results --#")
        try:
            for msg in final_state["final_state_node"]["messages"]:
                try:
                    print(msg.content)
                except AttributeError:
                    if isinstance(msg, list):
                        for x in msg:
                            print(x)
                    else:
                        print(msg)
        except KeyError:
            print("Processing skipped")


def find_datasets_main(args):
    asyncio.run(_find_datasets_main(args))


# main
if __name__ == "__main__":
    pass
