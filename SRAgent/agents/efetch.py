# import
## batteries
import os
import asyncio
from pprint import pprint
from typing import Annotated, Optional, Callable

## 3rd party
from dotenv import load_dotenv
from Bio import Entrez
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

## package
from SRAgent.agents.utils import set_model
from SRAgent.tools.efetch import efetch
from SRAgent.tools.entrez_db import which_entrez_databases


def create_efetch_agent(model_name: Optional[str] = None) -> Callable:
    """
    Create an agent that uses Entrez efetch to help complete a task.
    Args:
        model_name: Override model name from settings
    Returns:
        Configured agent instance
    """
    model = set_model(model_name=model_name)
    agent = create_react_agent(
        model=model,
        tools=[efetch, which_entrez_databases],
        prompt="\n".join(
            [
                "# Instructions",
                " - You are an expert in bioinformatics and you are working on a project to find information about a specific dataset.",
                " - Based on the task provided by your supervisor, use Entrez efetch to help complete the task.",
                " - If you are unsure of which database to query (sra or gds), you can use which_entrez_databases to determine which databases contain the Entrez ID.",
                "# Response",
                " - Base your response on the evidence you found; do not infer information.",
                " - Provide a concise summary of your findings; use lists when possible; do not include helpful wording.",
            ]
        ),
    )

    @tool
    async def invoke_efetch_agent(
        message: Annotated[str, "Message to the efetch agent"],
    ) -> Annotated[str, "Response from the efetch agent"]:
        """
        Invoke the efetch agent to run Entrez efetch queries.
        """
        # Invoke the agent with the message
        result = await agent.ainvoke({"messages": [HumanMessage(content=message)]})
        return {
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="efetch_agent")
            ]
        }

    return invoke_efetch_agent


if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv

    load_dotenv()
    Entrez.email = os.getenv("EMAIL")

    # test agent
    async def main():
        agent = create_efetch_agent()
        input = {"message": "I need to find information about 35966237."}
        result = await agent.ainvoke(input)
        pprint(result)

    asyncio.run(main())
