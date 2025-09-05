#!/usr/bin/env python
# import
import os
import sys
import asyncio
import operator
from typing import List, Dict, Any, Annotated, TypedDict, Sequence, Callable

## 3rd party
from dotenv import load_dotenv
import pandas as pd
from Bio import Entrez
from pydantic import BaseModel
from langgraph.types import Send
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables.config import RunnableConfig

## package
from SRAgent.agents.utils import set_model
from SRAgent.agents.find_datasets import create_find_datasets_agent
from SRAgent.workflows.srx_info import create_SRX_info_graph
from SRAgent.db.connect import db_connect
from SRAgent.db.upsert import db_upsert
from SRAgent.db.get import db_get_entrez_ids


# state
class GraphState(TypedDict):
    """
    Shared state of the agents in the graph
    """

    messages: Annotated[Sequence[BaseMessage], operator.add]
    entrez_ids: Annotated[List[int], "List of dataset Entrez IDs"]
    database: Annotated[str, operator.add]  # Database name (e.g., 'sra', 'gds')


# nodes
def create_find_datasets_node():
    # create the agent
    agent = create_find_datasets_agent()

    # create the node function
    async def invoke_find_datasets_agent_node(
        state: GraphState,
        config: RunnableConfig,
    ) -> Dict[str, Any]:
        """Invoke the find_datasets agent to get datasets to process"""
        # call the agent
        response = await agent.ainvoke(
            {"message": state["messages"][-1].content}, config=config
        )
        # return the last message in the response
        return {
            "messages": [response["messages"][-1]],
        }

    return invoke_find_datasets_agent_node


## entrez IDs extraction
class EntrezInfo(BaseModel):
    entrez_ids: List[int]
    database: str


def create_get_entrez_ids_node() -> Callable:
    model = set_model(agent_name="get_entrez_ids")

    async def invoke_get_entrez_ids_node(
        state: GraphState, config: RunnableConfig
    ) -> Dict[str, Any]:
        """
        Structured data extraction of Entrez IDs from message
        """
        # create prompt
        message = state["messages"][-1].content
        prompt = "\n".join(
            [
                "You are a helpful assistant for a bioinformatics researcher.",
                "# Tasks",
                " - Extract Entrez IDs (e.g., 19007785 or 27176348) from the message below.",
                "    - If you cannot find any Entrez IDs, do not provide any accessions.",
                "    - Entrez IDs may be referred to as 'database IDs' or 'accession numbers'.",
                " - Extract the database name (e.g., GEO, SRA, etc.)",
                "   - If you cannot find the database name, do not provide any database name.",
                "   - GEO should be formatted as 'gds'"
                "   - SRA should be formatted as 'sra'",
                "#-- START OF MESSAGE --#",
                message,
                "#-- END OF MESSAGE --#",
            ]
        )

        # invoke model with structured output; try 3 times to get valid output
        entrez_ids = []
        max_retries = 3
        database = ""
        for i in range(max_retries):
            try:
                response = await model.with_structured_output(
                    EntrezInfo, strict=True
                ).ainvoke(prompt)
                entrez_ids = response.entrez_ids
                database = str(response.database).lower()
                if database in ["sra", "gds"]:
                    break
            except Exception as e:
                if (
                    "OpenAIRefusalError" in str(type(e).__name__)
                    and i < max_retries - 1
                ):
                    print(
                        f"OpenAI refused to extract Entrez IDs (attempt {i + 1}), retrying...",
                        file=sys.stderr,
                    )
                    prompt += "\nIf no valid Entrez IDs or database are found, return empty values."
                    continue
                else:
                    # For final attempt or other errors, use empty values
                    print(f"Error extracting Entrez IDs: {str(e)}", file=sys.stderr)
                    entrez_ids = []
                    database = ""
                    break

        ## if no valid database, return no entrez IDs
        if database not in ["sra", "gds"]:
            return {"entrez_ids": [], "database": ""}

        # entrez ID check
        ## filter out entrez IDs that are already in the database
        if config.get("configurable", {}).get("use_database"):
            with db_connect() as conn:
                existing_ids = db_get_entrez_ids(conn=conn, database=database)
                entrez_ids = [x for x in entrez_ids if x not in existing_ids]

        # cap number of entrez IDs to max_datasets in config
        max_datasets = config.get("configurable", {}).get("max_datasets")
        if max_datasets and max_datasets > 0 and len(entrez_ids) > max_datasets:
            entrez_ids = entrez_ids[:max_datasets]

        ## update the database
        if len(entrez_ids) > 0 and config.get("configurable", {}).get("use_database"):
            df = pd.DataFrame(
                {
                    "entrez_id": entrez_ids,
                    "database": database,
                    "notes": "New dataset found by Find-Datasets agent",
                }
            )
            if config.get("configurable", {}).get("use_database"):
                with db_connect() as conn:
                    db_upsert(df, "srx_metadata", conn=conn)

        # return the extracted values
        return {"entrez_ids": entrez_ids, "database": database}

    return invoke_get_entrez_ids_node


def continue_to_srx_info(
    state: GraphState, config: RunnableConfig
) -> List[Dict[str, Any]]:
    """
    Parallel invoke of the srx_info graph
    """
    ## submit each SRX accession to the metadata graph
    responses = []
    for entrez_id in state["entrez_ids"]:
        input = {
            "database": state["database"],
            "entrez_id": entrez_id,
        }
        responses.append(Send("srx_info_node", input))
    return responses


def final_state(state: GraphState) -> Dict[str, Any]:
    """
    Return the final state of the graph
    Args:
        state: The final state of the graph
    Returns:
        The final state of the graph
    """
    # filter to messages that contain the SRX accession
    messages = []
    for msg in state["messages"]:
        try:
            msg = [msg.content]
        except AttributeError:
            msg = [x.content for x in msg]
        for x in msg:
            if x.startswith("# SRX accession: "):
                messages.append(x)
    # filter to unique messages
    messages = list(set(messages))
    # final message
    if len(messages) == 0:
        message = "No novel SRX accessions found."
    else:
        message = "\n".join(messages)
    return {"messages": [AIMessage(content=message)]}


def create_find_datasets_graph():
    # -- graph --#
    workflow = StateGraph(GraphState)

    # nodes
    workflow.add_node("search_datasets_node", create_find_datasets_node())
    workflow.add_node("get_entrez_ids_node", create_get_entrez_ids_node())
    workflow.add_node("srx_info_node", create_SRX_info_graph())
    workflow.add_node("final_state_node", final_state)

    # edges
    workflow.add_edge(START, "search_datasets_node")
    workflow.add_edge("search_datasets_node", "get_entrez_ids_node")
    workflow.add_conditional_edges(
        "get_entrez_ids_node", continue_to_srx_info, ["srx_info_node"]
    )
    workflow.add_edge("srx_info_node", "final_state_node")
    workflow.add_edge("final_state_node", END)

    # compile the graph
    graph = workflow.compile()
    return graph


# main
if __name__ == "__main__":
    from Bio import Entrez

    # -- setup --#
    from dotenv import load_dotenv

    load_dotenv(override=True)
    Entrez.email = os.getenv("EMAIL")

    # -- graph --#
    async def main():
        msg = "Obtain recent single cell RNA-seq datasets in the SRA database"
        input = {"messages": [HumanMessage(content=msg)]}
        config = {
            "max_concurrency": 4,
            "recursion_limit": 200,
            "configurable": {"organisms": ["rat"]},
        }
        graph = create_find_datasets_graph()
        async for step in graph.astream(input, config=config):
            print(step)

    asyncio.run(main())
