# import
## batteries
import sys
import asyncio
from typing import Annotated, List, Optional, Callable

## 3rd party
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai.chat_models.base import OpenAIRefusalError

## package
from SRAgent.agents.utils import set_model
from SRAgent.agents.disease_ontology import create_disease_ontology_agent


# classes
class MONDO_ID(BaseModel):
    id: str = Field(
        description="The MONDO/PATO term ID (MONDO:XXXXXXX or PATO:XXXXXXX) for the disease description or 'No suitable ontology term found' if no term is found."
    )


class MONDO_IDS(BaseModel):
    ids: List[MONDO_ID] = Field(
        description="The MONDO/PATO term IDs (MONDO:XXXXXXX or PATO:XXXXXXX) for each disease description, if available."
    )


# functions
def create_disease_ontology_workflow(
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
        create_disease_ontology_agent(
            model_name=model_name,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        ),
    ]

    # state modifier
    state_mod = "\n".join(
        [
            "# Introduction",
            " - You are a helpful senior bioinformatician assisting a researcher with a task involving classifying one or more diseases.",
            " - You will be provided with a free text description of the diseases.",
            " - Your task is to categorize the diseases based on the MONDO/PATO ontology.",
            " - You must find the single most suitable MONDO/PATO ontology term that best describes the disease description.",
            " - You have a set of tools that can help you with this task.",
            "# Tool summary",
            " - create_disease_ontology_agent: Use this tool to find the most suitable MONDO/PATO ontology term that best describes the disease description.",
            "# Workflow",
            " 1. Identify each unique disease description in the input.",
            "   - For example, 'brain cortex; eye lens; aortic valve;' should be split into the following separate descriptions:",
            "     - brain cortex",
            "     - eye lens",
            "     - aortic valve",
            ' 2. For each description (e.g., "heart disorder"), use the create_disease_ontology_agent tool to find the most suitable MONDO/PATO ontology term.',
            "   - You MUST use the create_disease_ontology_agent tool for EACH disease description."
            "# Response",
            " - Provide the MONDO/PATO ontology IDs (MONDO:XXXXXXX or PATO:XXXXXXX) that describe each disease description, if they are available.",
        ]
    )
    # create agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=state_mod,
        response_format=MONDO_IDS,
    )

    # create tool
    @tool
    async def invoke_disease_ontology_workflow(
        messages: Annotated[
            List[BaseMessage], "Messages to send to the Disease Ontology agent"
        ],
        config: RunnableConfig,
    ) -> Annotated[dict, "Response from the Disease Ontology agent"]:
        """
        Invoke the Disease Ontology agent with a message.
        The Disease Ontology agent will annotate each disease description with the most suitable MONDO/PATO term,
        or "No suitable ontology term found" if no term is found.
        """
        try:
            # The React agent expects messages in this format
            response = await agent.ainvoke({"messages": messages}, config=config)
            # filter out ids that do not start with "MONDO:" or "PATO:"
            ids = [
                x.id
                for x in response["structured_response"].ids
                if x.id.startswith("MONDO:") or x.id.startswith("PATO:")
            ]
            return ids
        except OpenAIRefusalError as e:
            # Handle cases where the model refuses to generate a response
            print(
                f"OpenAI refused to generate disease ontology: {str(e)}",
                file=sys.stderr,
            )
            return []

    return invoke_disease_ontology_workflow


# main
if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv

    load_dotenv(override=True)

    async def main():
        # create workflow
        workflow = create_disease_ontology_workflow()

        # Example 1: Complex disease description example
        print("\n=== Example 1: Complex disease description example ===")
        # msg = "Categorize the following diseases: the thin layer of epithelial cells lining the alveoli in lungs; brain cortex; eye lens"
        # msg = "Diseases: congestive heart failure, heart neoplasm"
        msg = "Diseases: bursitis, diverticulitis"
        input = {"messages": [HumanMessage(content=msg)]}
        results = await workflow.ainvoke(input)
        print(results)

    # run
    asyncio.run(main())
