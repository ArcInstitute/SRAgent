# import
## batteries
import os
import sys
import asyncio
from typing import Annotated, Callable

## 3rd party
from Bio import Entrez
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig

## package
from SRAgent.agents.utils import set_model
from SRAgent.agents.esearch import create_esearch_agent
from SRAgent.agents.esummary import create_esummary_agent
from SRAgent.agents.efetch import create_efetch_agent
from SRAgent.agents.elink import create_elink_agent
from SRAgent.agents.display import create_step_summary_chain


# functions
def create_entrez_agent(
    model_name=None,
    return_tool: bool = True,
) -> Callable:
    # create model
    model_supervisor = set_model(model_name=model_name, agent_name="entrez")

    # set tools
    tools = [
        create_esearch_agent(),
        create_esummary_agent(),
        create_efetch_agent(),
        create_elink_agent(),
    ]

    # state modifier
    state_mod = "\n".join(
        [
            "# Instructions"
            " - You are a helpful senior bioinformatician assisting a researcher with a task involving Entrez databases.",
            " - You have a team of agents who can perform specific tasks using Entrez tools.",
            " - Provide guidance to the agents to help them complete the task successfully.",
            "# Strategies",
            " - Generally, start with the esearch agent to find Entrez records, then use the efetch agent to get detailed information.",
            " - Use the esummary agent to obtain summary information on an Entrez record.",
            " - Use the elink agent to navigate between databases to find related records (e.g., GEO to SRA).",
            " - After each agent call, briefly analyze the agent's response:",
            "   - What information was obtained?",
            "   - Should I verify any questionable information with another data source?",
            "   - What information is still missing?",
            "   - Which agent(s) should be tried next?",
            "# Calling agents",
            ' - Be sure to provide context to the agents (e.g., "Use efetch to determine whether SRX4967527 is Illumina data.").'
            " - Generally, you will want to specify the database(s) to search (e.g., sra, gds, or pubmed).",
            " - If there are dozens of records, batch the IDs and call the agent multiple times to avoid rate limits and token count limits.",
            "# Conversion",
            " - If the task involves accessions instead of Entrez IDs, you may need to convert them to Entrez IDs first.",
            " - For example, convert SRX4967527 to the corresponding Entrez ID via eSearch of the SRA database.",
            "# Notes",
            " - Bulk RNA-seq is NOT the same as single-cell RNA-seq (scRNA-seq); be sure to distinguish between them.",
            "   - If you do not find evidence of single cell, do not assume it is scRNA-seq.",
            "   - A single layout does not imply single-cell data.",
            "# Response",
            " - Base your response on the evidence you found; do not infer information.",
            " - Be very concise; provide simple lists when possible; do not include unnecessary wording.",
            " - Write your output as plain text instead of markdown.",
            "# Example workflows",
            "## Task: Convert GSE123456 to SRX, SRP, or SRR accessions",
            "  1. esearch agent: eSearch of the GSE accession to obtain Entrez IDs",
            "  2. esummary agent: eSummary of the Entrez IDs to get the SRX accessions",
            "## Task: Obtain the SRR accessions for SRX4967527",
            "  1. esearch agent: eSearch of the SRX accession to obtain the Entrez ID",
            "  2. efetch agent: eFetch of the Entrez ID to obtain the SRR accessions",
            "## Task: Is SRP309720 paired-end Illumina 10X Genomics data?",
            "  1. esearch agent: eSearch of SRP accession obtain the Entrez IDs",
            "  2. efetch agent: eFetch of the Entrez IDs to get the library preparation information, including the 10X Genomics library prep method (e.g., 3 prime GEX)",
            "## Task: Obtain the SRA study accessions for the Entrez ID 36098095",
            "  1. efetch agent: eFetch of the Entrez ID to obtain the SRA accessions",
        ]
    )

    # create agent
    agent = create_react_agent(model=model_supervisor, tools=tools, prompt=state_mod)

    # return agent instead of tool
    if not return_tool:
        return agent

    @tool
    async def invoke_entrez_agent(
        message: Annotated[str, "Message to send to the Entrez agent"],
        config: RunnableConfig,
    ) -> Annotated[dict, "Response from the Entrez agent"]:
        """
        Invoke the Entrez agent with a message.
        The Entrez agent will perform a task using Entrez tools.
        """
        # Invoke the agent with the message
        result = await agent.ainvoke(
            {"messages": [AIMessage(content=message)]}, config=config
        )
        return {
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="entrez_agent")
            ]
        }

    return invoke_entrez_agent


async def create_entrez_agent_stream(
    input, config: dict = {}, summarize_steps: bool = False
) -> str:
    """
    Create an Entrez agent and stream the steps.
    Args:
        input: Input message to the agent.
        config: Configuration for the agent.
        summarize_steps: Whether to summarize the steps.
    Returns:
        The final step message.
    """
    # create entrez agent
    agent = create_entrez_agent(return_tool=False)

    # create step summary chain
    step_summary_chain = create_step_summary_chain() if summarize_steps else None

    # invoke agent
    step_cnt = 0
    final_step = ""
    async for step in agent.astream(input, stream_mode="values", config=config):
        step_cnt += 1
        final_step = step
        # summarize step
        if step_summary_chain:
            msg = step_summary_chain.invoke({"step": step})
            print(f"Step {step_cnt}: {msg.content}", file=sys.stderr)
        else:
            print(f"Step {step_cnt}: {step}", file=sys.stderr)
    try:
        final_step = final_step["agent"]["messages"][-1].content
    except KeyError:
        final_step = final_step["messages"][-1].content
    return final_step


# main
if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv

    load_dotenv(override=True)
    Entrez.email = os.getenv("EMAIL1")
    Entrez.api_key = os.getenv("NCBI_API_KEY1")

    # create entrez agent
    async def main():
        agent = create_entrez_agent()

        # invoke agent
        config = {"configurable": {"organisms": ["mouse", "rat"]}}
        # config = {"configurable": {"organisms": ["human"]}}
        # input = {"message": "Find rat single cell RNA-seq datasets in the SRA database"}
        # input = {"message": "Convert GSE121737 to SRX accessions"}
        input = {"message": "Is SRX20554853 paired-end Illumina data?"}
        # input = {"message": "List the collaborators for the SRX20554853 dataset"}
        # input = {"message": "How many bases per run in SRX20554853?"}
        result = await agent.ainvoke(input, config=config)
        print(result)

    asyncio.run(main())
