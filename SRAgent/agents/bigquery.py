# import
## batteries
import asyncio
from typing import Annotated, Any, Callable, Optional
## 3rd party
from google.cloud import bigquery
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
## package
from SRAgent.agents.utils import set_model
from SRAgent.tools.bigquery import create_get_study_experiment_run, create_get_study_metadata, create_get_experiment_metadata, create_get_run_metadata
from SRAgent.agents.entrez_convert import create_entrez_convert_agent

# functions
def create_bigquery_agent(model_name: Optional[str]=None) -> Callable:
    # create model
    model = set_model(model_name=model_name, agent_name="bigquery")

    # init client
    client = bigquery.Client()

    # set tools
    tools = [
        create_get_study_experiment_run(client),
        create_get_study_metadata(client),
        create_get_experiment_metadata(client),
        create_get_run_metadata(client),
        create_entrez_convert_agent()
    ]
  
    # state modifier
    state_mod = "\n".join([
        # Role and Purpose
        "You are an expert bioinformatician specialized in querying the Sequence Read Archive (SRA) database.",
        "Your purpose is to retrieve and analyze metadata across SRA's hierarchical structure: studies (SRP) → experiments (SRX) → runs (SRR).",
        "# Tool Capabilities",
        " 1. get_study_experiment_run: Retrieves study, experiment, and run accessions",
        " 2. get_study_metadata: Retrieves study and associated experiment accessions",
        " 3. get_experiment_metadata: Retrieves experiment details and associated run accessions",
        " 4. get_run_metadata: Retrieves detailed run-level information",
        " 5. entrez_convert: Converts Entrez IDs to SRA accessions",
        "# Tool usage guidelines",
        " - If your are provided with an Entrez ID, use the entrez_convert tool to convert it to SRA or ENA accessions.",
        "   - IMPORTANT: All get_*_metadata tools require SRA or ENA accessions",
        " - Use the get_study_experiment_run tool to convert accessions between study, experiment, and run levels.",
        " - Use the get_*_metadata tools to retrieve metadata for a specific accession type.",
        " - Chain the tools as needed to gather the necessary information for a given study, experiment, or run.",
        " - Be sure to provide all important information to each tool, such as accessions, databases, or metadata fields.",
        "# Warnings",
        " - Bulk RNA-seq is NOT the same as single-cell RNA-seq (scRNA-seq); be sure to distinguish between them.",
        "   - If you do not find evidence of single cell, do not assume it is scRNA-seq.",
        "   - A \"single layout\" does not imply single-cell data.",
        "# Response guidelines",
        " - If the query mentions one accession type but asks about another, automatically perform the necessary conversions",
        " - Chain multiple tool calls when needed to gather the necessary information for the task",
        " - If you receive an error, explain the error clearly and suggest alternatives",
        " - If you are instructed to perform a conversion, be sure to provide the final converted value (e.g., SRX accession)",
        "# Output format",
        " - Keep responses concise and structured",
        " - Present metadata as key-value pairs",
        " - Group related information",
        " - Include accession IDs in outputs",
        " - Do not use markdown formatting",
    ])

    # create agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        state_modifier=state_mod
    )
    @tool
    async def invoke_bigquery_agent(
        message: Annotated[str, "Message to send to the BigQuery agent"],
    ) -> Annotated[dict, "Response from the BigQuery agent"]:
        """
        Invoke the BigQuery agent with a message.
        The BigQuery agent will search the SRA database with BigQuery.
        """
        # Invoke the agent with the message
        result = await agent.ainvoke({"messages" : [AIMessage(content=message)]})
        return {
            "messages": [AIMessage(content=result["messages"][-1].content, name="bigquery_agent")]
        }
    return invoke_bigquery_agent

if __name__ == "__main__":
    # setup
    from dotenv import load_dotenv
    load_dotenv()

    #msg = "Convert Entrez ID 36927723 (associated with the sra database) to the corresponding SRX or ERX accessions."
    msg = "Convert Entrez ID 36927723 (associated with the sra database) to the corresponding SRX or ERX accessions. Provide detailed conversion mapping between the Entrez ID and experiment accessions."

    # test agent
    async def main():
        bigquery_agent = create_bigquery_agent()
        input = {"message" : msg}
        result = await bigquery_agent.ainvoke(input)
        print(result)
    asyncio.run(main())

    # print(bigquery_agent.invoke({"message" : "Get study metadata for SRP548813"}))
    # print(bigquery_agent.invoke({"message" : "Get experiment metadata for SRP548813"}))
    # print(bigquery_agent.invoke({"message" : "Get the number of base pairs for all runs in SRP548813"}))
    # print(bigquery_agent.invoke({"message" : "Convert SRP548813 to SRR"}))

