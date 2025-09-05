# import
## batteries
import asyncio
from typing import Annotated, Callable, Optional

## 3rd party
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

## package
from SRAgent.agents.utils import set_model
from SRAgent.tools.sequences import sra_stat, fastq_dump


# functions
def create_sequences_agent(model_name: Optional[str] = None) -> Callable:
    """
    Create an agent to call the sequence-based tools.
    Args:
        model_name: Override model name from settings
    Returns:
        Configured agent instance
    """
    model = set_model(model_name=model_name, agent_name="sequences")
    agent = create_react_agent(
        model=model,
        tools=[sra_stat, fastq_dump],
        prompt="\n".join(
            [
                "# Instructions",
                " - You are an expert in bioinformatics and you are working on a project to find information about a specific dataset.",
                " - Based on the task provided by your supervisor, use fastq-dump and sra-stat to help complete the task.",
                " - You can investige the sequence data (fastq files) associated with GEO and/or SRA accessions.",
                " - Use sra-stat to obtain sequence data information (e.g., number of spots and bases) associated with GEO and/or SRA accessions.",
                " - fastq-dump is useful for quickly checking the fastq files of SRR accessions (e.g., is the data actually paired-end?).",
                " - Note: fastq-dump only works with SRR accessions; use the convert tool to first convert Entrez IDs and accessions to SRR accessions.",
                "# Response",
                " - Base your response on the evidence you found; do not infer information.",
                " - Provide a concise summary of your findings; use lists when possible; do not include helpful wording.",
            ]
        ),
    )

    @tool
    async def invoke_sequences_agent(
        message: Annotated[str, "Message to the sequences agent"],
    ) -> Annotated[str, "Response from the sequences agent"]:
        """
        Invoke the sequences agent to run the sra-stat and fastq-dump tools
        and provide information the sequence data (fastq files).
        """
        # Invoke the agent with the message
        result = await agent.ainvoke({"messages": [HumanMessage(content=message)]})
        return {
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="sequence_agent")
            ]
        }

    return invoke_sequences_agent


if __name__ == "__main__":
    load_dotenv()

    # Create the sequences agent
    async def main():
        agent = create_sequences_agent()
        input = {"message": "I need information about the SRA accession SRR18029850."}
        result = await agent.ainvoke(input)
        print(result)

    asyncio.run(main())
