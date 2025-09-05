# import
## batteries
import os
import asyncio
from typing import Annotated, Callable, Optional

## 3rd party
from Bio import Entrez
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

## package
from SRAgent.agents.utils import set_model
from SRAgent.tools.esummary import esummary
from SRAgent.tools.entrez_db import which_entrez_databases


# functions
def create_esummary_agent(model_name: Optional[str] = None) -> Callable:
    """
    Create an agent that uses Entrez esummary to help complete a task.
    Args:
        model_name: Override model name from settings
    Returns:
        Configured agent instance
    """
    model = set_model(model_name=model_name, agent_name="esummary")
    agent = create_react_agent(
        model=model,
        tools=[esummary, which_entrez_databases],
        prompt="\n".join(
            [
                "# Instructions",
                " - You are an expert in bioinformatics and you are working on a project to find information about a specific dataset.",
                " - Based on the task provided by your supervisor, use Entrez esearch to help complete the task.",
                " - If you are unsure of which database to query (e.g., sra, gds, or pubmed), you can use which_entrez_databases to determine which databases contain the Entrez ID.",
                "# Response",
                " - Base your response on the evidence you found; do not infer information.",
                " - Provide a concise summary of your findings; use lists when possible; do not include helpful wording.",
            ]
        ),
    )

    @tool
    async def invoke_esummary_agent(
        message: Annotated[str, "Message to the esummary agent"],
    ) -> Annotated[str, "Response from the esummary agent"]:
        """
        Invoke the esummary agent to perform a task.
        """
        # Invoke the agent with the message
        result = await agent.ainvoke({"messages": [HumanMessage(content=message)]})
        return {
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="esummary_agent")
            ]
        }

    return invoke_esummary_agent


if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv

    load_dotenv()
    Entrez.email = os.getenv("EMAIL")

    # test esummary agent
    async def main():
        agent = create_esummary_agent()
        input = {"message": "Find information about 27978912."}
        result = await agent.ainvoke(input)
        print(result)

    asyncio.run(main())
