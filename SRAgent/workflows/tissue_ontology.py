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
from SRAgent.agents.tissue_ontology import create_tissue_ontology_agent


# classes
class UBERON_ID(BaseModel):
    id: str = Field(
        description="The Uberon term ID (UBERON:XXXXXXX) for the tissue description or 'No suitable ontology term found' if no term is found."
    )


class UBERON_IDS(BaseModel):
    ids: List[UBERON_ID] = Field(
        description="The Uberon term IDs (UBERON:XXXXXXX) for each tissue description, if available."
    )


# functions
def create_tissue_ontology_workflow(
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
        create_tissue_ontology_agent(
            model_name=model_name,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        ),
    ]

    # state modifier
    state_mod = "\n".join(
        [
            "# Introduction",
            " - You are a helpful senior bioinformatician assisting a researcher with a task involving classifying one or more tissue.",
            " - You will be provided with a free text description of the tissues.",
            " - Your task is to categorize the tissues based on the Uberon ontology.",
            " - You must find the single most suitable Uberon ontology term that best describes the tissue description.",
            " - You have a set of tools that can help you with this task.",
            "# Tool summary",
            " - create_tissue_ontology_agent: Use this tool to find the most suitable Uberon ontology term that best describes the tissue description.",
            "# Workflow",
            " 1. Identify each unique tissue description in the input.",
            "   - For example, 'brain cortex; eye lens; aortic valve;' should be split into the following separate descriptions:",
            "     - brain cortex",
            "     - eye lens",
            "     - aortic valve",
            ' 2. For each description (e.g., "brain cortex"), use the create_tissue_ontology_agent tool to find the most suitable Uberon ontology term.',
            "   - You MUST use the create_tissue_ontology_agent tool for EACH tissue description."
            "# Notes",
            ' - There is no valid Uberon ontology term for "tumor" or "cancer".',
            "   - You must be provided with the tissue context for the tumor/cancer.",
            '   - For example, "tumor of of the skin" is valid, but "tumor" is not valid.',
            "   - Only call the create_tissue_ontology_agent tool when you have the tissue context for tumor/cancer.",
            "# Response",
            " - Provide the Uberon ontology IDs (UBERON:XXXXXXX) that describe each tissue description, if they are available.",
        ]
    )
    # create agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=state_mod,
        response_format=UBERON_IDS,
    )

    # create tool
    @tool
    async def invoke_tissue_ontology_workflow(
        messages: Annotated[
            List[BaseMessage], "Messages to send to the Tissue Ontology agent"
        ],
        config: RunnableConfig,
    ) -> Annotated[dict, "Response from the Tissue Ontology agent"]:
        """
        Invoke the Tissue Ontology agent with a message.
        The Tissue Ontology agent will annotate each tissue description with the most suitable Uberon term,
        or "No suitable ontology term found" if no term is found.
        """
        try:
            # The React agent expects messages in this format
            response = await agent.ainvoke({"messages": messages}, config=config)
            # filter out ids that do not start with "UBERON:"
            ids = [
                x.id
                for x in response["structured_response"].ids
                if x.id.startswith("UBERON:")
            ]
            return ids
        except OpenAIRefusalError as e:
            # Handle cases where the model refuses to generate a response
            # This typically happens when tissue description is too vague (e.g., just "tumor")
            print(
                f"OpenAI refused to generate tissue ontology: {str(e)}", file=sys.stderr
            )
            return []

    return invoke_tissue_ontology_workflow


# main
if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv

    load_dotenv(override=True)

    async def main():
        # create workflow
        workflow = create_tissue_ontology_workflow()

        # Example 1: Complex tissue description example
        print("\n=== Example 1: Complex tissue description example ===")
        # msg = "Categorize the following tissues: the thin layer of epithelial cells lining the alveoli in lungs; brain cortex; eye lens"
        # msg = "Tissues: lung,tumor,adjacent non-affected lung"
        msg = "Tissues: tumor"
        input = {"messages": [HumanMessage(content=msg)]}
        results = await workflow.ainvoke(input)
        print(results)

    # run
    asyncio.run(main())
