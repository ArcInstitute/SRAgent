# import
## batteries
import os
import asyncio
from typing import Annotated, Callable

## 3rd party
from Bio import Entrez
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig

## package
from SRAgent.agents.utils import set_model
from SRAgent.tools.esearch import esearch


# functions
def create_esearch_agent(model_name: str = None) -> Callable:
    """
    Create an agent that uses Entrez esearch to help complete a task.
    Args:
        model_name: Override model name from settings
    Returns:
        Configured agent instance
    """
    model = set_model(model_name=model_name, agent_name="esearch")
    agent = create_react_agent(
        model=model,
        tools=[esearch],
        prompt="\n".join(
            [
                "# Instructions",
                " - You are an expert in bioinformatics and you are working on a project to find information about a specific dataset.",
                " - Based on the task provided by your supervisor, use Entrez esearch to help complete the task.",
                " - If the sra or gds database does not return findings, try the other database.",
                "# Response",
                " - Base your response on the evidence you found; do not infer information.",
                " - Provide a concise summary of your findings; use lists when possible; do not include helpful wording.",
            ]
        ),
    )

    @tool
    async def invoke_esearch_agent(
        message: Annotated[str, "Message to the esearch agent"], config: RunnableConfig
    ) -> Annotated[str, "Response from the esearch agent"]:
        """
        Invoke the esearch agent to perform a task.
        """
        # Invoke the agent with the message
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=message)]}, config=config
        )
        return {
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="esearch_agent")
            ]
        }

    return invoke_esearch_agent


if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv

    load_dotenv()
    Entrez.email = os.getenv("EMAIL")

    # test agent
    async def main():
        agent = create_esearch_agent()
        input = {
            "message": "Find rat single-cell RNA-seq datasets.",
        }
        config = {"configurable": {"organisms": ["human", "mouse", "rat", "dog"]}}
        result = await agent.ainvoke(input, config=config)
        print(result)

    asyncio.run(main())
