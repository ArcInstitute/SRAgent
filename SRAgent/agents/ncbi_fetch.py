# import
## batteries
import asyncio
from typing import Annotated, Callable

## 3rd party
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

## package
from SRAgent.agents.utils import set_model
from SRAgent.tools.ncbi_fetch import (
    fetch_geo_record,
    fetch_ncbi_record,
    fetch_pubmed_record,
    fetch_biosample_record,
    fetch_bioproject_record,
)


def create_ncbi_fetch_agent(model_name: str = None) -> Callable:
    """
    Create an agent that queries the NCBI website
    """
    model = set_model(model_name=model_name, agent_name="ncbi_fetch")
    agent = create_react_agent(
        model=model,
        tools=[
            fetch_geo_record,
            fetch_ncbi_record,
            fetch_pubmed_record,
            fetch_biosample_record,
            fetch_bioproject_record,
        ],
        prompt="\n".join(
            [
                "# Instructions"
                " - You are an expert in bioinformatics and you are working on a project to find information about a specific dataset.",
                " - You will use tools that directly request data from the NCBI website.",
                " - You can query with both Entrez IDs and accessions (e.g. SRA, GEO, PubMed, Biosample, BioProject).",
                "# Strategy",
                " - Always try multiple tools if the first tool does not return the information you need.",
                " - Be sure to provide all important information each each tool, such as accessions, databases, or metadata fields.",
                "# Notes",
                " - Bulk RNA-seq is NOT the same as single-cell RNA-seq (scRNA-seq); be sure to distinguish between them.",
                "   - If you do not find evidence of single cell, do not assume it is scRNA-seq.",
                "# Response",
                " - Base your response on the evidence you found; do not infer information.",
                " - Provide a concise summary of your findings.",
                " - Use lists when possible.",
                " - Do not include helpful wording like 'Here is the information I found'.",
            ]
        ),
    )

    @tool
    async def invoke_ncbi_fetch_agent(
        message: Annotated[str, "Message to the ncbi-fetch agent"],
    ) -> Annotated[str, "Response from the ncbi-fetch agent"]:
        """
        Invoke the ncbi-fetch agent to query the NCBI website for information
        on Entrez IDs, SRA accessions, and GEO accessions.
        """
        # Invoke the agent with the message
        result = await agent.ainvoke({"messages": [HumanMessage(content=message)]})
        return {
            "messages": [
                AIMessage(
                    content=result["messages"][-1].content, name="ncbi-fetch_agent"
                )
            ]
        }

    return invoke_ncbi_fetch_agent


# main
if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv

    load_dotenv(override=True)

    # test agent
    async def main():
        invoke_ncbi_fetch_agent = create_ncbi_fetch_agent()
        # message = "Fetch information for Entrez ID 35447314"
        # message = "Fetch information for Entrez ID 200277303. The accession is associated with the gds database"
        # message = "Fetch information for Biosample ID SAMN39619157"
        # message = "Fetch information for BioProject ID PRJNA1026523"
        message = "Fetch BioProject PRJNA1143898 entry and state the library preparation method (e.g., 10X Genomics)."
        result = await invoke_ncbi_fetch_agent.ainvoke(message)
        print(result)

    asyncio.run(main())
