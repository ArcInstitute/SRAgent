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
from SRAgent.tools.disease_ontology import (
    query_vector_db,
    get_neighbors,
    query_mondo_ols,
)


# classes
class MONDO_ID(BaseModel):
    id: str = Field(
        description="The selected MONDO ID (MONDO:XXXXXXX) or 'No suitable ontology term found'"
    )


# functions
def create_disease_ontology_agent(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    return_tool: bool = True,
) -> Callable:
    # create model
    model = set_model(
        model_name=model_name,
        agent_name="disease_ontology",
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )

    # set tools
    tools = [
        query_vector_db,
        get_neighbors,
        query_mondo_ols,
    ]

    # state modifier
    state_mod = "\n".join(
        [
            "# Introduction",
            " - You are a helpful senior bioinformatician assisting a researcher with a task involving classifying a disease.",
            " - You will be provided with a free text description of the disease.",
            " - Your task is to categorize the disease based on the MONDO/PATO ontology.",
            " - You must find the single most suitable MONDO/PATO ontology term that best describes the disease description.",
            " - You have a set of tools that can help you with this task.",
            "# Tool summary",
            " - query_vector_db: Perform a semantic search on a vector database to find MONDO/PATO terms related to the target disease. The database contains a collection of disease descriptions and their corresponding MONDO/PATO terms.",
            " - get_neighbors: Get the neighbors of a given MONDO/PATO term in the MONDO/PATO ontology. Useful for finding adjacent terms in the ontology.",
            " - query_mondo_ols: Query the Ontology Lookup Service (OLS) for MONDO/PATO terms matching the search term.",
            "# Workflow",
            " Step 1: Use the query_vector_db tool to find the most similar MONDO/PATO terms.",
            "   - ALWAYS use the query_vector_db tool to find the most similar MONDO/PATO terms.",
            " Step 2: Use the get_neighbors tool on the MONDO/PATO terms returned in Step 1 to help find the most suitable term.",
            "   - ALWAYS use the get_neighbors tool to explore more the terms adjacent to the terms returned in Step 1.",
            " Step 3: Repeat steps 1 and 2 until you are confident in the most suitable term.",
            "   - ALWAYS perform between 1 and 3 iterations to find the most suitable term.",
            " Step 4: If you are uncertain about which term to select, use the query_mondo_ols tool to help find the most suitable term.",
            "   - Note: The query_mondo_ols tool is only useful for finding MONDO terms. It is not useful for finding PATO terms.",
            "# Response",
            " - Provide the most suitable MONDO/PATO ontology ID (MONDO:XXXXXXX or PATO:XXXXXXX) that best describes the disease description.",
            ' - If a suitable term is not found, provide "No suitable ontology term found".',
        ]
    )
    # create agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=state_mod,
        response_format=MONDO_ID,
    )

    # return agent instead of tool
    if not return_tool:
        return agent

    # create tool
    @tool
    async def invoke_disease_ontology_agent(
        disease_description: Annotated[str, "Disease description to annotate"],
        config: RunnableConfig,
    ) -> Annotated[
        dict,
        "Response from the Disease Ontology agent with the most suitable MONDO/PATO term, or 'No suitable ontology term found' if one cannot be found",
    ]:
        """
        Invoke the Disease Ontology agent with a message.
        The Disease Ontology agent will annotate a disease description with the most suitable MONDO/PATO term, if one can be found.
        """
        try:
            # Invoke the agent with the message
            messages = [HumanMessage(content=disease_description)]
            result = await agent.ainvoke({"messages": messages}, config=config)
            msg = f"Disease ontology term ID: {result['structured_response'].id}"
        except OpenAIRefusalError:
            msg = "No suitable ontology term found"
        return {"messages": [AIMessage(content=msg, name="disease_ontology_agent")]}

    return invoke_disease_ontology_agent


# main
if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv

    load_dotenv(override=True)

    async def main():
        # create entrez agent
        agent = create_disease_ontology_agent(return_tool=False)

        # Example 1: Simple disease example
        # print("\n=== Example 1: Simple disease example ===")
        # msg = "Categorize the following disease: heart disorder"
        # input = {"messages": [HumanMessage(content=msg)]}
        # result = await agent.ainvoke(input)
        # print(result['messages'][-1].content)

        # Example 2: More specific disease example
        # print("\n=== Example 2: More specific disease example ===")
        # msg = "Categorize the following disease: congestive heart failure"
        # input = {"messages": [HumanMessage(content=msg)]}
        # result = await agent.ainvoke(input)
        # print(f"Results: {result['structured_response'].id}")

    asyncio.run(main())
