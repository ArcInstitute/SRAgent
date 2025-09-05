# import
## batteries
import os
import asyncio
from typing import Annotated, Callable

## 3rd party
from Bio import Entrez
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage

## package
from SRAgent.agents.utils import set_model
from SRAgent.agents.esearch import create_esearch_agent
from SRAgent.agents.esummary import create_esummary_agent
from SRAgent.agents.elink import create_elink_agent
from SRAgent.tools.ncbi_fetch import fetch_geo_record, fetch_ncbi_record


# functions
def create_entrez_convert_agent(
    model_name: str = None,
    return_tool: bool = True,
) -> Callable:
    # create model
    model_supervisor = set_model(model_name=model_name, agent_name="entrez_convert")

    # set tools
    tools = [
        create_esearch_agent(),
        create_esummary_agent(),
        create_elink_agent(),
        fetch_geo_record,
        fetch_ncbi_record,
    ]

    # state modifier
    state_mod = "\n".join(
        [
            "You are a helpful senior bioinformatician assisting a researcher with a task involving Entrez databases.",
            "You have a team of agents who can perform specific tasks using Entrez tools.",
            "Your goal is to convert Entrez IDs to SRA or ENA accessions.",
            "Provide guidance to the agents to help them complete the task successfully.",
            "# Strategy",
            ' - Be sure to provide context to the agents (e.g., "Use esearch to find SRA accessions associated with Entrez ID 123456.").',
            " - Generally, you will want to specify the database(s) to search (e.g., sra, gds, or pubmed).",
            " - If there are dozens of records, batch the IDs and call the agent multiple times to avoid rate limits and token count limits.",
            "# Execution Rules",
            " - Continue sending tasks to your agents until you successfully complete the task.",
            " - Be very concise; provide simple lists when possible; do not include unnecessary wording.",
            "# Response Format",
            " - Be sure to provide the final conversion results (e.g., SRA accessions).",
            " - Write your output as plain text instead of markdown.",
            "#Example workflows",
            "## Task: Convert the SRA Entrez ID 123456 to SRA accessions",
            "  1. fetch NCBI record: fetch the NCBI record for the Entrez ID 123456",
            "  2. esearch agent: esearch of the SRA accessions to obtain SRA accessions",
            "  3. esummary agent: esummary of the Entrez IDs to confirm the SRA accessions",
            "## Task: Convert the GEO Entrez ID 123456 to SRA accessions",
            "  1. fetch GEO record: fetch the GEO record for the Entrez ID 123456",
            "  2. esearch agent: esearch of the GEO accessions to obtain GEO accessions",
            "  3. elink agent: elink of the GEO accessions to obtain SRA accessions",
        ]
    )

    # create agent
    agent = create_react_agent(model=model_supervisor, tools=tools, prompt=state_mod)

    # return agent instead of tool
    if not return_tool:
        return agent

    @tool
    async def invoke_entrez_convert_agent(
        message: Annotated[str, "Message to send to the Entrez Convert agent"],
    ) -> Annotated[dict, "Response from the Entrez Convert agent"]:
        """
        Invoke the Entrez Convert agent with a message.
        The Entrez agent will convert Entrez IDs to SRA or ENA accessions.
        """
        # Invoke the agent with the message
        result = await agent.ainvoke({"messages": [AIMessage(content=message)]})
        return {
            "messages": [
                AIMessage(
                    content=result["messages"][-1].content, name="entrez_convert_agent"
                )
            ]
        }

    return invoke_entrez_convert_agent


# main
if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv

    load_dotenv()
    Entrez.email = os.getenv("EMAIL")

    # test
    async def main():
        agent = create_entrez_convert_agent()
        input = {"message": "Convert 35087715 to SRA accessions"}
        result = await agent.ainvoke(input)
        print(result)

    asyncio.run(main())
