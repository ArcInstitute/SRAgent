```mermaid

graph LR

    SRAgent_Orchestrator["SRAgent Orchestrator"]

    NCBI_Entrez_Agent["NCBI Entrez Agent"]

    Dataset_Discovery_Agent["Dataset Discovery Agent"]

    Tissue_Ontology_Agent["Tissue Ontology Agent"]

    BigQuery_Integration_Agent["BigQuery Integration Agent"]

    Sequence_Analysis_Agent["Sequence Analysis Agent"]

    Agent_Output_Formatter["Agent Output Formatter"]

    Agent_Utilities["Agent Utilities"]

    Workflow_Orchestrator["Workflow Orchestrator"]

    External_Service_Integration["External Service Integration"]

    Workflow_Orchestrator -- "uses" --> SRAgent_Orchestrator

    SRAgent_Orchestrator -- "uses" --> NCBI_Entrez_Agent

    SRAgent_Orchestrator -- "uses" --> Dataset_Discovery_Agent

    SRAgent_Orchestrator -- "uses" --> Tissue_Ontology_Agent

    SRAgent_Orchestrator -- "uses" --> BigQuery_Integration_Agent

    SRAgent_Orchestrator -- "uses" --> Sequence_Analysis_Agent

    NCBI_Entrez_Agent -- "uses" --> External_Service_Integration

    Dataset_Discovery_Agent -- "uses" --> External_Service_Integration

    Tissue_Ontology_Agent -- "uses" --> External_Service_Integration

    BigQuery_Integration_Agent -- "uses" --> External_Service_Integration

    Sequence_Analysis_Agent -- "uses" --> External_Service_Integration

    SRAgent_Orchestrator -- "uses" --> Agent_Utilities

    NCBI_Entrez_Agent -- "uses" --> Agent_Utilities

    Dataset_Discovery_Agent -- "uses" --> Agent_Utilities

    Tissue_Ontology_Agent -- "uses" --> Agent_Utilities

    BigQuery_Integration_Agent -- "uses" --> Agent_Utilities

    Sequence_Analysis_Agent -- "uses" --> Agent_Utilities

    Agent_Output_Formatter -- "uses" --> Agent_Utilities

    SRAgent_Orchestrator -- "uses" --> Agent_Output_Formatter

    NCBI_Entrez_Agent -- "uses" --> Agent_Output_Formatter

    Dataset_Discovery_Agent -- "uses" --> Agent_Output_Formatter

    Tissue_Ontology_Agent -- "uses" --> Agent_Output_Formatter

    BigQuery_Integration_Agent -- "uses" --> Agent_Output_Formatter

    Sequence_Analysis_Agent -- "uses" --> Agent_Output_Formatter

    Agent_Utilities -- "uses" --> Agent_Output_Formatter

    click Workflow_Orchestrator href "https://github.com/ArcInstitute/SRAgent/blob/main/.codeboarding//Workflow_Orchestrator.md" "Details"

    click External_Service_Integration href "https://github.com/ArcInstitute/SRAgent/blob/main/.codeboarding//External_Service_Integration.md" "Details"

```



[![CodeBoarding](https://img.shields.io/badge/Generated%20by-CodeBoarding-9cf?style=flat-square)](https://github.com/CodeBoarding/GeneratedOnBoardings)[![Demo](https://img.shields.io/badge/Try%20our-Demo-blue?style=flat-square)](https://www.codeboarding.org/demo)[![Contact](https://img.shields.io/badge/Contact%20us%20-%20contact@codeboarding.org-lightgrey?style=flat-square)](mailto:contact@codeboarding.org)



## Details



The Core AI Agents component is the intelligent backbone of the system, encapsulating the LLM-driven reasoning and decision-making logic for specialized bioinformatics tasks. It comprises various specialized agents, each designed to handle specific aspects of data curation and retrieval. These agents leverage External Service Integration tools to perform atomic operations and contribute to complex bioinformatics workflows. These components are fundamental because they embody the "Agent-Oriented Architecture" of the project. Each agent is a specialized, LLM-driven entity responsible for a distinct set of bioinformatics tasks, promoting modularity and reusability. Their interactions demonstrate the "Orchestration Pattern" where the SRAgent Orchestrator coordinates the activities of other agents, and the "Tool-Use/Function Calling" pattern as they all rely on External Service Integration to interact with the outside world. This clear separation of concerns makes the system scalable, maintainable, and adaptable to new bioinformatics challenges.



### SRAgent Orchestrator

The primary agent responsible for orchestrating high-level, SRA-specific bioinformatics workflows and coordinating tasks among other specialized agents. It acts as the central control point for SRA data curation and retrieval processes.





**Related Classes/Methods**:



- `SRAgent Orchestrator` (1:1)





### NCBI Entrez Agent

Manages all interactions with the NCBI Entrez suite of databases. This includes performing searches (ESearch), fetching detailed records (EFetch), linking related entries (ELink), summarizing results (ESummary), and converting data formats.





**Related Classes/Methods**:



- `NCBI Entrez Agent` (1:1)

- `NCBI Entrez Agent` (1:1)

- `NCBI Entrez Agent` (1:1)

- `NCBI Entrez Agent` (1:1)

- `NCBI Entrez Agent` (1:1)

- `NCBI Entrez Agent` (1:1)

- `NCBI Entrez Agent` (1:1)





### Dataset Discovery Agent

Specializes in identifying and locating relevant bioinformatics datasets based on user queries or internal logic. It leverages various tools to search for and filter datasets across different sources.





**Related Classes/Methods**:



- `Dataset Discovery Agent` (1:1)





### Tissue Ontology Agent

Handles the processing, interpretation, and application of tissue-related ontological information. This agent ensures consistency and accuracy in classifying and retrieving data based on tissue types.





**Related Classes/Methods**:



- `Tissue Ontology Agent` (1:1)





### BigQuery Integration Agent

Facilitates direct interaction with Google Cloud BigQuery for querying and retrieving large-scale bioinformatics datasets stored within the BigQuery environment.





**Related Classes/Methods**:



- `BigQuery Integration Agent` (1:1)





### Sequence Analysis Agent

Focuses on tasks related to the processing, manipulation, and basic analysis of biological sequence data (e.g., DNA, RNA, protein sequences).





**Related Classes/Methods**:



- `Sequence Analysis Agent` (1:1)





### Agent Output Formatter

Responsible for formatting and displaying the progress, intermediate results, and final outputs of the various agents in a user-friendly and consistent manner.





**Related Classes/Methods**:



- `Agent Output Formatter` (1:1)





### Agent Utilities

Provides a collection of common helper functions, reusable logic, and utility methods that support the operations of multiple agents within the subsystem.





**Related Classes/Methods**:



- `Agent Utilities` (1:1)





### Workflow Orchestrator [[Expand]](./Workflow_Orchestrator.md)

Manages overall bioinformatics workflows.





**Related Classes/Methods**: _None_



### External Service Integration [[Expand]](./External_Service_Integration.md)

Provides tools for interacting with external APIs and services.





**Related Classes/Methods**: _None_







### [FAQ](https://github.com/CodeBoarding/GeneratedOnBoardings/tree/main?tab=readme-ov-file#faq)