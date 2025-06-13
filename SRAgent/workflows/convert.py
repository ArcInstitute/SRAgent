# import 
import os
import re
import sys
import asyncio
import operator
from enum import Enum
from typing import Annotated, List, Dict, Tuple, Optional, Union, Any, Sequence, TypedDict, Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
## package
from SRAgent.agents.utils import set_model
from SRAgent.agents.sragent import create_sragent_agent
from SRAgent.workflows.utils import entrez_id_to_srx
from SRAgent.tools.utils import structured_output_with_retry

# state
class GraphState(TypedDict):
    """
    Shared state of the agents in the graph
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    entrez_id: Annotated[str, "Entrez ID"]
    SRP: Annotated[List[str], operator.add]
    SRX: Annotated[List[str], operator.add]
    SRR: Annotated[List[str], operator.add]
    route: Annotated[str, "Route choice"]
    attempts: Annotated[int, operator.add]

## convert agent
def create_convert_agent_node() -> Callable:
    convert_agent = create_sragent_agent()
    async def invoke_convert_agent_node(state: GraphState) -> Dict[str, List[str]]:
        """
        Invoke the Entrez convert agent to obtain SRA accessions
        """
        # attempt to convert entrez ID via entrez_id_to_srx
        srx = await entrez_id_to_srx(str(state["entrez_id"]))
        if srx:
            srx = ", ".join(set(srx))
            return {"messages" : [AIMessage(content=f"Obtained accessions: {srx}")]}
        # fallback to convert agent
        response = await convert_agent.ainvoke({"messages" : state["messages"]})
        return {"messages" : [response["messages"][-1]]}
    return invoke_convert_agent_node

## accessions extraction
class Acessions(BaseModel):
    srx: List[str]

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
        # Try to extract accessions with retries on refusal
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ## invoke model with structured output
                #response = await model.with_structured_output(Acessions, strict=True).ainvoke(prompt)
                response = await structured_output_with_retry(model, Acessions, prompt, max_retries=3)
                return {"SRX" : response.srx}
            except Exception as e:
                if "OpenAIRefusalError" in str(type(e).__name__) and attempt < max_retries - 1:
                    print(f"OpenAI refused to extract accessions (attempt {attempt + 1}), retrying...", file=sys.stderr)
                    prompt += "\nIf no valid SRX/ERX accessions are found, return an empty list."
                    continue
                else:
                    # For final attempt or other errors, return empty list
                    print(f"Error extracting accessions: {str(e)}", file=sys.stderr)
                    return {"SRX" : []}
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
                " - If one or more accessions have been obtained, state \"STOP\". If more information is needed, state \"CONTINUE\".",
                " - If more information is needed (\"CONTINUE\"), provide one or two sentences of feedback on how to obtain the data (e.g., use esearch instead of efetch).",
            ])),
            ("system", "\nHere are the last few messages:"),
            MessagesPlaceholder(variable_name="history"),
            ("system", "\nHere are the extracted SRA accessions:\n" + accesions)
        ])
        formatted_prompt = prompt.format_messages(history=state["messages"][-4:])
        # call the model
        #response = await model.with_structured_output(Choice, strict=True).ainvoke(formatted_prompt)
        response = await structured_output_with_retry(model, Choice, formatted_prompt, max_retries=3)
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
    # filter to just distinct SRX accessions
    return {"SRX" : list(set(response["SRX"]))}

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

    # save graph image
    # from SRAgent.utils import save_graph_image
    # save_graph_image(graph)
    
    ## invoke with graph object directly provided
    #invoke_convert_graph = partial(invoke_convert_graph, graph=graph)
    #print(invoke_convert_graph(input))
    

    #-- nodes --#
    state = {
        "entrez_id" : "34748561",
        "messages" : []
    }
    # entrez_agent_node = create_entrez_agent_node(state)

    state = {
        "messages" : [
            HumanMessage(
                content=" - **Study Accession**: SRP526682\n- **Sample Accession**: SRS22358714- **Run Accession**: SRR30256648\n**Experiment Accession**: SRX4967528"
            )
        ]
    }
    # create_get_accessions_node(state)
