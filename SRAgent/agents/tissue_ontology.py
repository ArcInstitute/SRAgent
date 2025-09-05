# import
## batteries
import asyncio
from typing import Annotated, Optional, Callable

## 3rd party
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai.chat_models.base import OpenAIRefusalError

## package
from SRAgent.agents.utils import set_model
from SRAgent.tools.tissue_ontology import (
    query_vector_db,
    get_neighbors,
    query_uberon_ols,
)


# classes
class UBERON_ID(BaseModel):
    id: str = Field(
        description="The selected Uberon ID (UBERON:XXXXXXX) or 'No suitable ontology term found'"
    )


# functions
def create_tissue_ontology_agent(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    return_tool: bool = True,
) -> Callable:
    # create model
    model = set_model(
        model_name=model_name,
        agent_name="tissue_ontology",
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )

    # set tools
    tools = [
        query_vector_db,
        get_neighbors,
        query_uberon_ols,
    ]

    # state modifier
    state_mod = "\n".join(
        [
            "# Introduction",
            " - You are a helpful senior bioinformatician assisting a researcher with a task involving classifying a tissue.",
            " - You will be provided with a free text description of the tissue.",
            " - Your task is to categorize the tissue based on the Uberon ontology.",
            " - You must find the single most suitable Uberon ontology term that best describes the tissue description.",
            " - You have a set of tools that can help you with this task.",
            "# Tool summary",
            " - query_vector_db: Perform a semantic search on a vector database to find Uberon terms related to the target tissue. The database contains a collection of tissue descriptions and their corresponding Uberon terms.",
            " - get_neighbors: Get the neighbors of a given Uberon term in the Uberon ontology. Useful for finding adjacent terms in the ontology.",
            " - query_uberon_ols: Query the Ontology Lookup Service (OLS) for Uberon terms matching the search term.",
            "# Workflow",
            " Step 1: Use the query_vector_db tool to find the most similar Uberon terms.",
            "   - ALWAYS use the query_vector_db tool to find the most similar Uberon terms.",
            " Step 2: Use the get_neighbors tool on the Uberon terms returned in Step 1 to help find the most suitable term.",
            "   - ALWAYS use the get_neighbors tool to explore more the terms adjacent to the terms returned in Step 1.",
            " Step 3: Repeat steps 1 and 2 until you are confident in the most suitable term.",
            "   - ALWAYS perform between 1 and 3 iterations to find the most suitable term.",
            " Step 4: If you are uncertain about which term to select, use the query_uberon_ols tool to help find the most suitable term.",
            "# Notes",
            ' - There is no valid Uberon ontology term for "tumor" or "cancer". You must be provided with the tissue context for the cancer/tumor.',
            '   - For example, UBERON:1000010 (mole) should only be used if the tissue description includes "mole" and/or "skin".',
            '   - If the tissue context was not provided, return "No suitable ontology term found"; DO NOT return an Uberon term.',
            "# Response",
            " - Provide the most suitable Uberon ontology ID (UBERON:XXXXXXX) that best describes the tissue description.",
            ' - If a suitable term is not found, provide "No suitable ontology term found".',
        ]
    )
    # create agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=state_mod,
        response_format=UBERON_ID,
    )

    # return agent instead of tool
    if not return_tool:
        return agent

    # create tool
    @tool
    async def invoke_tissue_ontology_agent(
        tissue_description: Annotated[str, "Tissue description to annotate"],
        config: RunnableConfig,
    ) -> Annotated[
        dict,
        "Response from the Tissue Ontology agent with the most suitable Uberon term, or 'No suitable ontology term found' if one cannot be found",
    ]:
        """
        Invoke the Tissue Ontology agent with a message.
        The Tissue Ontology agent will annotate a tissue description with the most suitable Uberon term, if one can be found.
        """
        try:
            # Invoke the agent with the message
            messages = [HumanMessage(content=tissue_description)]
            result = await agent.ainvoke({"messages": messages}, config=config)
            msg = f"Tissue ontology term ID: {result['structured_response'].id}"
        except OpenAIRefusalError:
            msg = "No suitable ontology term found"
        return {"messages": [AIMessage(content=msg, name="tissue_ontology_agent")]}

    return invoke_tissue_ontology_agent


# main
if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv

    load_dotenv(override=True)

    async def main():
        # create entrez agent
        agent = create_tissue_ontology_agent(return_tool=False)

        # Example 1: Simple tissue example
        print("\n=== Example 1: Simple tissue example ===")
        msg = "Categorize the following tissue: brain cortex"
        input = {"messages": [HumanMessage(content=msg)]}
        result = await agent.ainvoke(input)
        print(result["messages"][-1].content)

        # Example 2: More specific tissue example
        # print("\n=== Example 2: More specific tissue example ===")
        # msg = "Categorize the following tissue: hippocampus"
        # input = {"messages": [HumanMessage(content=msg)]}
        # result = await agent.ainvoke(input)
        # print(f"Result for 'hippocampus': {result['structured_response'].id}")

    asyncio.run(main())
