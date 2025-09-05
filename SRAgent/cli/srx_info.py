# import
## batteries
import os
import sys
import asyncio
import argparse
import pandas as pd
from typing import List, Optional, Callable
## 3rd party
from Bio import Entrez
## package
from SRAgent.cli.utils import CustomFormatter
from SRAgent.workflows.srx_info import create_SRX_info_graph
from SRAgent.workflows.graph_utils import handle_write_graph_option
from SRAgent.agents.display import create_step_summary_chain
from SRAgent.db.connect import db_connect 
from SRAgent.db.get import db_get_srx_records

# functions
def SRX_info_agent_parser(subparsers):
    help = 'SRX_info Agent: Obtain metadata for SRA experiments.'
    desc = """
    """
    sub_parser = subparsers.add_parser(
        'srx-info', help=help, description=desc, formatter_class=CustomFormatter
    )
    sub_parser.set_defaults(func=SRX_info_agent_main)
    sub_parser.add_argument(
        'entrez_ids', type=str, nargs='+', help='>=1 dataset Entrez IDs to query (or a csv file with a column named "entrez_id")'
    )    
    sub_parser.add_argument(
        '--database', type=str, default='sra', choices=['gds', 'sra'], 
        help='Entrez database origin of the Entrez IDs'
    )
    sub_parser.add_argument(
        '--max-concurrency', type=int, default=6, help='Maximum number of concurrent processes'
    )
    sub_parser.add_argument(
        '--recursion-limit', type=int, default=200, help='Maximum recursion limit'
    )
    sub_parser.add_argument(
        '--max-parallel', type=int, default=3, help='Maximum parallel processing of entrez ids'
    )
    sub_parser.add_argument(
        '--use-database', action='store_true', default=False, 
        help='Add the results to the SRAgent SQL database'
    )
    sub_parser.add_argument(
        '--tenant', type=str, default=os.getenv("DYNACONF", 'prod'),
        choices=['prod', 'test', 'claude'],
        help='Settings environment (also sets DYNACONF). Use "claude" to run with Claude defaults.'

    )
    sub_parser.add_argument(
        '--reprocess-existing', action='store_true', default=False, 
        help='Reprocess existing Entrez IDs in the SRAgent database instead of ignoring existing (assumning --use-database)'
    )
    sub_parser.add_argument(
        '--write-graph', type=str, metavar='FILE', default=None,
        help='Write the workflow graph to a file and exit (supports .png, .svg, .pdf, .mermaid formats)'
    )

async def _process_single_entrez_id(
    entrez_id, database, graph, 
    step_summary_chain: Optional[Callable], 
    config: dict, no_summaries: bool
):
    """
    Process a single entrez_id.
    Args:
        entrez_id: The Entrez ID to process.
        database: The database to use.
        graph: The graph to use.
        step_summary_chain: The step summary chain to use.
        config: The config to use.
        no_summaries: Whether to print summaries.
    """
    input = {
        "entrez_id": entrez_id, 
        "database": database,
    }
    final_state = None
    i = 0
    async for step in graph.astream(input, config=config):
        i += 1
        final_state = step
        if step_summary_chain:
            msg = await step_summary_chain.ainvoke({"step": step})
            print(f"[{entrez_id}] Step {i}: {msg.content}")
        else:
            nodes = ",".join(list(step.keys()))
            print(f"[{entrez_id}] Step {i}: {nodes}")

    if final_state:
        print(f"#-- Final results for Entrez ID {entrez_id} --#")
        try:
            print(final_state["final_state_node"]["messages"][-1].content)
        except KeyError:
            print("Processing skipped")
    print("#---------------------------------------------#")

async def _SRX_info_agent_main(args):
    """
    Main function for invoking the srx-info agent
    """
    # filter entrez_ids
    if args.use_database and not args.reprocess_existing:
        existing_ids = set()
        with db_connect() as conn:
            existing_ids = set(db_get_srx_records(conn))
        args.entrez_ids = [x for x in args.entrez_ids if x not in existing_ids]
        if len(args.entrez_ids) == 0:
            print("All Entrez IDs are already in the metadata database.", file=sys.stderr)
            return 0

    # set email and api key
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")

    # create supervisor agent
    graph = create_SRX_info_graph()
    if not args.no_summaries:
        step_summary_chain = create_step_summary_chain()
    else:
        step_summary_chain = None

    # invoke agent
    config = {
        "max_concurrency": args.max_concurrency,
        "recursion_limit": args.recursion_limit,
        "configurable": {
            "use_database": args.use_database,
            "reprocess_existing": args.reprocess_existing,
        }
    }

    # Create semaphore to limit concurrent processing
    semaphore = asyncio.Semaphore(args.max_parallel)

    async def _process_with_semaphore(entrez_id):
        async with semaphore:
            await _process_single_entrez_id(
                entrez_id,
                args.database,
                graph,
                step_summary_chain,
                config,
                args.no_summaries,
            )

    # Create tasks for each entrez_id
    tasks = [_process_with_semaphore(entrez_id) for entrez_id in args.entrez_ids]
    
    # Run tasks concurrently with limited concurrency
    await asyncio.gather(*tasks)

def SRX_info_agent_main(args):
    # set tenant
    if args.tenant:
        os.environ["DYNACONF"] = args.tenant
    
    # handle write-graph option
    if args.write_graph:
        handle_write_graph_option(create_SRX_info_graph, args.write_graph)
        return

    # if entrez_ids is a csv, read in the entrez_ids
    if args.entrez_ids[0].endswith(".csv") and os.path.exists(args.entrez_ids[0]):
        df = pd.read_csv(args.entrez_ids[0])
        if "entrez_id" not in df.columns:
            print("'entrez_id' column not found in the csv file", file=sys.stderr)
            return 1
        args.entrez_ids = df["entrez_id"].unique().astype(str).tolist()
        
    # filter to non-integer entrez_ids
    problem_entrez_ids = [x for x in args.entrez_ids if not x.isnumeric()]
    if problem_entrez_ids:
        print("Invalid Entrez IDs found:", file=sys.stderr)
        for x in problem_entrez_ids:
            print(x, file=sys.stderr)
        return 1
    # run agent
    asyncio.run(_SRX_info_agent_main(args))

# main
if __name__ == '__main__':
    pass
