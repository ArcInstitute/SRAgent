# import
## batteries
import os
import asyncio
from typing import Annotated, Callable, Optional

## 3rd party
from Bio import Entrez
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage

## package
from SRAgent.agents.utils import set_model
from SRAgent.tools.esearch import esearch_scrna
from SRAgent.tools.entrez_db import which_entrez_databases


# functions
def create_find_datasets_agent(model_name: Optional[str] = None) -> Callable:
    """
    Create an agent that uses esearch to find datasets.
    Args:
        model_name: Override model name from settings
    Returns:
        Callable: Function to invoke the agent
    """
    model = set_model(model_name=model_name, agent_name="find_datasets")
    agent = create_react_agent(
        model=model,
        tools=[esearch_scrna],
        prompt="\n".join(
            [
                "# Instructions",
                " - You are an expert in bioinformatics and you are working on a project to find information about a specific dataset.",
                " - Based on the task provided by your supervisor, use Entrez esearch to help complete the task.",
                "# Strategy",
                " - If your initial search does not return any results, try different search terms or databases.",
                " - If the first attempt fails, you MUST make at least one more attempt to find datasets.",
                "# Response",
                " - You will return Entrez IDs.",
                " - Be sure to state which database you searched (e.g., GEO, SRA).",
                " - Provide a concise summary of your findings; use lists when possible; do not include helpful wording.",
            ]
        ),
    )

    @tool
    async def invoke_find_datasets_agent(
        message: Annotated[str, "Message to the find_datasets agent"],
        config: RunnableConfig,
    ) -> Annotated[str, "Response from the find_datasets agent"]:
        """
        Invoke the find_datasets agent to perform a task.
        """
        # Invoke the agent with the message
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=message)]}, config=config
        )
        return {
            "messages": [
                AIMessage(
                    content=result["messages"][-1].content, name="find_datasets_agent"
                )
            ]
        }

    return invoke_find_datasets_agent


if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv

    load_dotenv()
    Entrez.email = os.getenv("EMAIL")

    # test agent
    async def main():
        agent = create_find_datasets_agent()
        input = {"message": "Find single cell datasets"}
        result = await agent.ainvoke(input)
        print(result)

    asyncio.run(main())
