#!/usr/bin/env python
# import
import os
import asyncio
import operator
from typing import List, Dict, Any, Tuple, Annotated, TypedDict, Sequence, Callable
## 3rd party
from dotenv import load_dotenv
import pandas as pd
from Bio import Entrez
from pydantic import BaseModel
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
## package
from SRAgent.agents.find_datasets import create_find_datasets_agent
from SRAgent.workflows.srx_info import create_SRX_info_graph

# state
class GraphState(TypedDict):
    """
    Shared state of the agents in the graph
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    entrez_ids: Annotated[List[int], "List of dataset Entrez IDs"]

# nodes
def create_find_datasets_node():
    # create the agent
    agent = create_find_datasets_agent()

    # create the node function
    async def invoke_find_datasets_agent_node(state: GraphState) -> Dict[str, Any]:
        """Invoke the find_datasets agent to get datasets to process"""
        # call the agent
        response = await agent.ainvoke({"message": state["messages"][-1].content})
        # return the last message in the response
        return {
            "messages" : [response["messages"][-1]],
        }
    return invoke_find_datasets_agent_node

## entrez IDs extraction
class EntrezInfo(BaseModel):
    entrez_ids: List[int]
    database: str

def create_get_entrez_ids_node() -> Callable:
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    async def invoke_get_entrez_ids_node(state: GraphState):
        """
        Structured data extraction
        """
        # create prompt
        message = state["messages"][-1].content
        prompt = "\n".join([
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
            "#-- END OF MESSAGE --#"
        ])
        # invoke model with structured output
        response = await model.with_structured_output(EntrezInfo, strict=True).ainvoke(prompt)
        return {
            "entrez_ids" : response.entrez_ids,
            "database" : str(response.database).lower()
        }
    return invoke_get_entrez_ids_node

def continue_to_srx_info(state: GraphState) -> List[Dict[str, Any]]:
    """
    Parallel invoke of the srx_info graph
    """
    ## submit each SRX accession to the metadata graph
    responses = []
    for entrez_id in state["entrez_ids"]:
        input = {
            "database": state["database"],
            "entrez_id": entrez_id
        }
        responses.append(Send("srx_info_node", input))
    return responses

def final_state(state: GraphState) -> Dict[str, Any]:
    """
    Final state of the graph
    """
    return {"messages" : state["messages"]}

def create_find_datasets_graph():
    #-- graph --#
    workflow = StateGraph(GraphState)

    # nodes
    workflow.add_node("search_datasets_node", create_find_datasets_node())
    workflow.add_node("get_entrez_ids_node", create_get_entrez_ids_node())
    workflow.add_node("srx_info_node", create_SRX_info_graph())
    workflow.add_node("final_state_node", final_state)

    # edges
    workflow.add_edge(START, "search_datasets_node")
    workflow.add_edge("search_datasets_node", "get_entrez_ids_node")
    workflow.add_conditional_edges("get_entrez_ids_node", continue_to_srx_info, ["srx_info_node"])
    workflow.add_edge("srx_info_node", "final_state_node")
    workflow.add_edge("final_state_node", END)

    # compile the graph
    graph = workflow.compile()
    return graph

# main
if __name__ == "__main__":
    from functools import partial
    from Bio import Entrez

    #-- setup --#
    from dotenv import load_dotenv
    load_dotenv()
    Entrez.email = os.getenv("EMAIL")

    #-- graph --#
    async def main():
        msg = "Obtain recent single cell RNA-seq datasets in the SRA database"
        input = {"messages" : [HumanMessage(content=msg)]}
        config = {"max_concurrency" : 8, "recursion_limit": 200}
        graph = create_find_datasets_graph()
        async for step in graph.astream(input, config=config):
            print(step)
    asyncio.run(main())