# import 
import os
import re
import asyncio
import operator
from enum import Enum
from typing import Annotated, List, Sequence, TypedDict, Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
## package
from SRAgent.agents.utils import set_model


# state
class GraphState(TypedDict):
    """
    Shared state of the agents in the graph
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    tissue_level1: Annotated[str, "Tissue level 1"]
    tissue_level2: Annotated[str, "Tissue level 2"]


def extract_accessions(message: str) -> List[str]:
    """
    Extract SRX and ERX accessions from text using regex
    Args:
        message: Text message to extract accessions from
    Returns:
        List of unique SRX and ERX accessions
    """
    # Find all matches in the message
    accessions = re.findall(r'(?:SRX|ERX)[0-9]{4,}+', message)
    # Return unique accessions
    return list(set(accessions))

def create_get_accessions_node() -> Callable:
    model = set_model(agent_name="accessions")
    async def invoke_get_accessions_node(state: GraphState):
        """
        Structured data extraction
        """
        # try regex extraction
        accessions = extract_accessions(state["messages"][-1].content)
        if accessions:
            return {"SRX" : accessions}
        # fallback to model
        ## create prompt
        message = state["messages"][-1].content
        prompt = "\n".join([
            f"Extract SRX and ERX accessions (e.g., \"SRX123456\" or \"ERX223344\") from the message below.",
            "If you cannot find any SRX or ERX accessions (must have the \"SRX\" or \"ERX\" prefix), do not provide any accessions.",
            "#-- START OF MESSAGE --#",
            message,
            "#-- END OF MESSAGE --#"
        ])
        ## invoke model with structured output
        response = await model.with_structured_output(Acessions, strict=True).ainvoke(prompt)
        return {"SRX" : response.srx}
    return invoke_get_accessions_node

## router
class Choices(Enum):
    CONTINUE = "CONTINUE"
    STOP = "STOP"

class Choice(BaseModel):
    Choice: Choices
    Message: str

def create_router_node() -> Callable:
    """
    Router for the graph
    """
    model = set_model(agent_name="convert_router")

    async def invoke_router(
        state: GraphState
    ) -> Annotated[dict, "Response from the router"]:
        """
        Route the conversation to the appropriate tool based on the current state of the conversation.
        """
        # format accessions
        def format_accessions(accessions):
            if not accessions:
                return "No accessions found"
            return ", ".join(accessions)

        accesions = "\n".join([
            " - SRX: " + format_accessions(state["SRX"]),
            "\n"
        ])

        # create prompt
        prompt = ChatPromptTemplate.from_messages([
            # First add any static system message if needed
            ("system", "\n".join([
                "# Instructions",
                " - You determine whether Sequence Read Archive SRX accessions (e.g., SRX123456) have been obtained from the Entrez ID.",
                " - There should be at least one SRX accession.",
                " - ERX accessions are also valid.",
                " - If the accessions have been obtained, select STOP. If more information is needed, select CONTINUE.",
                " - If more information is needed (CONTINUE), provide one or two sentences of feedback on how to obtain the data (e.g., use esearch instead of efetch).",
            ])),
            ("system", "\nHere are the last few messages:"),
            MessagesPlaceholder(variable_name="history"),
            ("system", "\nHere are the extracted SRA accessions:\n" + accesions)
        ])
        formatted_prompt = prompt.format_messages(history=state["messages"][-4:])
        # call the model
        response = await model.with_structured_output(Choice, strict=True).ainvoke(formatted_prompt)
        # format the response
        return {
            "route": response.Choice.value,  
            "messages": [AIMessage(content=response.Message)], 
            "attempts": 1
        }
    
    return invoke_router

def route_interpret(state: GraphState) -> str:
    """
    Determine the route based on the current state of the conversation.
    """
    if state["attempts"] >= 2:
        return END
    return "convert_agent_node" if state["route"] == "CONTINUE" else END

def create_convert_graph() -> StateGraph:
    """
    Create a graph that converts Entrez IDs & non-SRA accessions to SRA accessions
    """
    workflow = StateGraph(GraphState)
    
    # nodes
    workflow.add_node("convert_agent_node", create_convert_agent_node())
    workflow.add_node("get_accessions_node", create_get_accessions_node())
    workflow.add_node("router_node", create_router_node())

    # edges
    workflow.add_edge(START, "convert_agent_node")
    workflow.add_edge("convert_agent_node", "get_accessions_node")
    workflow.add_edge("get_accessions_node", "router_node")
    workflow.add_conditional_edges("router_node", route_interpret)

    # compile the graph
    graph = workflow.compile()
    return graph

async def invoke_convert_graph(
    state: GraphState, 
    graph: StateGraph,
) -> Annotated[dict, "Response from the graph"]:
    """
    Invoke the graph to convert Entrez IDs & non-SRA accessions to SRA accessions
    """
    # filter state to just GraphState keys
    state_filt = {k: v for k, v in state.items() if k in graph.state_keys}
    response = await graph.ainvoke(state_filt)
    # filter to just the keys we want to return
    return {"SRX" : response["SRX"]}

# main
if __name__ == "__main__":
    from functools import partial
    from Bio import Entrez

    #-- setup --#
    from dotenv import load_dotenv
    load_dotenv(override=True)
    Entrez.email = os.getenv("EMAIL")

    #-- graph --#
    async def main():
        entrez_id = "34748561"
        #entrez_id = "30749595"
        #entrez_id = "307495950000"
        msg = f"Obtain all SRX and ERX accessions for the Entrez ID {entrez_id}"
        input = {"messages" : [HumanMessage(content=msg)], "entrez_id" : entrez_id}
        config = {"max_concurrency" : 3, "recursion_limit": 30}
        graph = create_convert_graph()
        async for step in graph.astream(input, config=config):
            print(step)
    asyncio.run(main())
    exit();

