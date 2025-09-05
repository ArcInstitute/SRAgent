# import
## batteries
import os
import asyncio
from typing import Annotated, List, Optional, Callable

## 3rd party
from Bio import Entrez
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

## package
from SRAgent.agents.utils import set_model
from SRAgent.agents.entrez import create_entrez_agent
from SRAgent.agents.ncbi_fetch import create_ncbi_fetch_agent
from SRAgent.agents.bigquery import create_bigquery_agent
from SRAgent.agents.sequences import create_sequences_agent


# functions
def create_sragent_agent(
    model_name: Optional[str] = None,
    return_tool: bool = True,
) -> Callable:
    # create model
    model = set_model(model_name=model_name, agent_name="sragent")

    # set tools
    tools = [
        create_entrez_agent(),
        create_ncbi_fetch_agent(),
        create_bigquery_agent(),
        create_sequences_agent(),
    ]

    # state modifier
    state_mod = "\n".join(
        [
            "# Introduction",
            " - You are a helpful senior bioinformatician assisting a researcher with a task involving National Center for Biotechnology Information (NCBI) databases.",
            " - You have a team of agents who can perform specific tasks using tools.",
            " - Your role is to coordinate these agents effectively to complete tasks, even if initial attempts fail.",
            "# Agents",
            "## Entrez Agent",
            " - The Entrez agent can perform tasks using the NCBI Entrez tools:",
            "   - Search for datasets in the NCBI SRA database.",
            "   - Retrieve metadata for datasets.",
            "   - Assess sequence data for datasets.",
            "   - Link records across databases (e.g., SRA to PubMed, or SRA to GEO).",
            " - Generally, this agent works with Entrez IDs as input.",
            "## NCBI Fetch Agent",
            " - The NCBI Fetch agent can fetch data directly from the NCBI website (html).",
            " - The agent can use any set of Entrez IDs and/or SRA, ENA, GEO, PubMed, Biosample, or BioProject accessions as input.",
            " - The agent is general purpose, given the large range of input types it can handle.",
            "## BigQuery Agent",
            " - The BigQuery agent can use the SRA BigQuery tool to find metadata on SRA and ENA datasets.",
            " - The agent can also be used to convert accessions among the SRA (and ENA) hierarchy: studies (SRP) → experiments (SRX) → runs (SRR).",
            "## Sequences Agent",
            " - The Sequences agent can fetch sequences and statistics on sequences (e.g., number of bases) from the NCBI database.",
            " - This agent is useful for questions about sequence data (e.g., number of bases, sequence length, etc.).",
            "# Strategy",
            " - Always try multiple approaches if the first attempt fails.",
            " - Consider the following patterns:",
            "   - If searching for metadata, first try the BigQuery agent, and then try the NCBI Fetch and Entrez agents.",
            "   - For accession conversions, try the BigQuery agent first, and then try the Entrez agent.",
            "   - For sequence data, combine the Sequences agent with metadata from other agents.",
            " - Keep track of what you have tried and what information you still need.",
            " - If one agent returns partial information, use another agent to fill in the gaps.",
            " - If unsure about any information, cross-validate by calling multiple agents.",
            "# Execution rules",
            " - Be sure to provide ALL important information to each agent, such as accessions, databases, or metadata fields.",
            " - Do not simplify metadata fields; ask the agent for the EXACT metadata field.",
            '   - For example, state "Which 10X Genomics library preparation technology?" to the agent instead of just "10X Genomics data".',
            " - Provide the agent with concise, clear directions.",
            " - After each agent call, briefly analyze the agent's response:",
            "   - What information was obtained?",
            "   - Should I verify any questionable information with another data source?",
            "   - What information is still missing?",
            "   - Which agent(s) should be tried next?",
            "# Notes",
            " - Bulk RNA-seq is NOT the same as single-cell RNA-seq (scRNA-seq); be sure to distinguish between them.",
            ' - "single cell" typically refers to whole-cell sequencing; "nucleus" is usually stated if single nucleus sequencing.',
            "# Final response",
            " - Be concise; use lists when possible.",
            " - Do not use markdown headers or other complex formatting.",
            " - In final response, combine all gathered information.",
        ]
    )
    # create agent
    agent = create_react_agent(model=model, tools=tools, prompt=state_mod)

    # return agent instead of tool
    if not return_tool:
        return agent

    # create tool
    @tool
    async def invoke_sragent_agent(
        messages: Annotated[List[BaseMessage], "Messages to send to the SRAgent agent"],
        config: RunnableConfig,
    ) -> Annotated[dict, "Response from the SRAgent agent"]:
        """
        Invoke the SRAgent agent with a message.
        The SRAgent agent will perform a task using Entrez and other tools.
        """
        # Invoke the agent with the message
        result = await agent.ainvoke({"messages": messages}, config=config)
        return {
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="sragent_agent")
            ]
        }

    return invoke_sragent_agent


# main
if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv

    load_dotenv(override=True)
    Entrez.email = os.getenv("EMAIL")

    async def main():
        # create entrez agent
        agent = create_sragent_agent()

        # invoke agent
        config = {"configurable": {"organisms": ["mouse", "rat"]}}
        # msg = "Find rat scRNA-seq datasets in the SRA database"
        # msg = "Convert GSE121737 to SRX accessions"
        # msg = "Is SRX20554853 paired-end Illumina data?"
        # msg = "What is the tissue description and tissue ontology for SRX20554853?"
        msg = "Obtain all SRR accessions for SRX20554853"
        # msg = "List the collaborators for the SRX20554853 dataset"
        # msg = "\n".join([
        #     "For the SRA accession SRX25994842, find the following information:"
        #     " - Is the dataset Illumina sequence data?",
        #     " - Is the dataset single cell RNA-seq data?",
        #     " - Is the dataset paired-end sequencing data?",
        #     " - Is the dataset 10X Genomics data?",
        #     " - Which 10X Genomics technology?",
        #     " - Which organism was sequenced?"
        # ])
        input = {"messages": [HumanMessage(content=msg)]}
        result = await agent.ainvoke(input, config=config)
        print(result)

    asyncio.run(main())
