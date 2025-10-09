SRAgent
=======

Agentic workflows for obtaining data from the Sequence Read Archive.

# Manuscript

**scBaseCount: An AI agent-curated, uniformly processed, and continually expanding single cell data repository**.
Nicholas D Youngblut, Christopher Carpenter, Jaanak Prashar, Chiara Ricci-Tam, Rajesh Ilango, Noam Teyssier,
Silvana Konermann, Patrick Hsu, Alexander Dobin, David P Burke, Hani Goodarzi, Yusuf H Roohani.
bioRxiv 2025.02.27.640494; doi: [https://doi.org/10.1101/2025.02.27.640494](https://doi.org/10.1101/2025.02.27.640494)

# Install

## Option 1: Via conda

Create a conda environment [optional]: 

```bash
mamba create -n sragent-env -y python=3.12 sra-tools=3.1 \
  && conda activate sragent-env
```

Clone the repository:

```bash
git clone https://github.com/ArcInstitute/SRAgent.git \
  && cd SRAgent
```

## Option 2: Via UV

Create a uv venv:

```bash
uv venv
```

Source the venv:

```bash
source .venv/bin/activate
```

## Install the package

Install the package:
    
```bash
pip install .
```

For development, install the package with the `dev` optional dependencies:

```bash
pip install -e ".[dev]"
```

## Environmental variables

* `OPENAI_API_KEY` = API key for using the OpenAI API
  * **required** when using OpenAI models
  * See [Configuring models](#configuring-models) information on setting models
* `ANTHROPIC_API_KEY` = API key for using the Anthropic API
  * **required** when using Claude models
* `EMAIL` = email for using the Entrez API
  * optional, but **HIGHLY** recommended
* `NCBI_API_KEY` = API key for using the Entrez API
  * optional, increases rate limits
* `DYNACONF` = switch between "test", "prod", and "claude" environments
  * optional, default is "prod"
  * this affects the SQL database used and models selected
  * no database is used by default

# Testing

```bash
pip install pytest
```

```bash
pytest tests/
```

# Usage

## SQL database

Components of SRAgent can use an SQL database to store the results.

This was crucial for the scBaseCount project, in order to:
* track which datasets had been processed 
* quickly assess the progress of the project

However, for most users, the SQL database is not necessary.
SRAgent does not use the SQL database by default.

> Note: currently only a GCP Postgresql database is supported.

To set up the database, see [Setting up the SQL Database](#setting-up-the-sql-database).


## Entrez Agent

The lowest-level agent in the SRAgent hiearchy.
The agent can call various Entrez tools (`esearch`, `efetch`, `esummary`, and `elink`).
Usually, the SRAgent agent will be more useful, since it includes more tools, including calling the Entrez agent.

#### Example accession conversion:

```bash
SRAgent entrez "Convert GSE121737 to SRX accessions"
```

#### Example of summarizing a dataset:

```bash
SRAgent entrez "Summarize SRX4967527"
```

## SRAgent agent

A general tool for extracting data from the SRA database.
The tools available:

* Entrez agent (see above)
* SRA BigQuery
* scraping NCBI webpage HTML
* sra-stat and fastq-dump (directly assessing sequence data)

#### Example of summarizing a dataset

> using more tools than the Entrez agent

```bash
SRAgent sragent "Summarize SRX4967527"
```

#### Example of obtaining metadata for a specific SRX accession:

```bash
SRAgent sragent "Obtain any available publications for GSE196830"
```

#### Example of obtaining specific metadata fields for a dataset:

```bash
SRAgent sragent "Which 10X Genomics technology was used for ERX11887200?"
```

## Papers agent

Locate publications linked to SRA accessions and download the corresponding manuscripts.

* Input formats: single accession (SRX/PRJ/ERX/ERP) or a CSV with an accession column (`--accession-column` to override the default `accession`).
* Workflow: 
  * converts accessions to Entrez IDs
  * finds linked PubMed IDs
  * extracts DOIs
  * tries multiple sources to fetch manuscripts from the DOIs (preprint servers, CORE, Europe PMC, Unpaywall)
* Output: console summary of publications, DOIs, download status, and stored files under `--output-dir/<accession>`.
* CSV enrichment: when a CSV input is provided, an updated copy is written to `--output-dir/<input-filename>` with added `pubmed_id`, `doi`, and `download_path` columns.
* Helpful options: 
  * `--core-api-key`/`CORE_API_KEY` for CORE API
  * `--email`/`EMAIL` for Unpaywall
  * `--max-concurrency` for max concurrent tasks
  * `--recursion-limit` for max recursion depth
  * `--write-graph` for writing the workflow graph to a file

#### Examples

Download papers for a single accession:

```bash
SRAgent papers SRX4967527
```

Process a CSV and create an enriched output file:

```bash
SRAgent papers accessions.csv
```

## SRX-info agent

Obtain specific metadata for >=1 SRA dataset.

* Input: >=1 Entrez ID
* Output metadata fields:
  * SRX accession for the Entrez ID
  * SRR accessions for the SRX accession
  * Is the dataset Illumina sequence data?
  * Is the dataset single cell RNA-seq data?
  * Is the dataset paired-end sequencing data?
  * Which scRNA-seq library preparation technology?
  * If 10X Genomics, which particular 10X technologies?
  * Single nucleus or single cell RNA sequencing?
  * Which organism was sequenced?
  * Which tissue(s) were sequenced?
  * Corresponding tissue ontology ID(s)
  * Any disease information?
  * Any treatment/perturbation information?
  * Any cell line information?
* Workflow
  * The agent converts the Entrez IDs to SRX accessions
  * For each SRX accession, the agent obtains metadata
  * The agent consolidates the metadata into a single report

> As of now, the metadata fields are hard-coded into the agent.
> If you need alternative metadata fields, you will have to modify [metadata.py](./workflows/metadata.py).

#### Examples

A single SRA dataset:

```bash
SRAgent srx-info 25576380
```

Multiple SRA datasets:

```bash
SRAgent srx-info 36106630 32664033 27694586
```

Use the SQL database to filter out already-processed datasets:

```bash
SRAgent srx-info --use-database 18060880 27454880 27454942 27694586
```

## Metadata agent

Similar to the `SRX-info` agent, but you can provide SRX accessions directly, instead of Entrez IDs.
This saves compute time, since the agent does not need to convert the Entrez IDs to SRX accessions.

Provide a CSV of Entrez IDs and their associated SRX accessions to obtain metadata. 
Useful for when you already have the SRX accessions, instead of providing the Entrez IDs to `SRAgent srx-info`.

The CSV should have the header: `entrez_id,srx_accession`.

The metadata fields are the same as the `SRX-info` agent.

#### Examples

```bash
SRAgent metadata "entrez-id_srx-accession.csv"
```

## Tissue-ontology agent

An agent for categorizing tissue descriptions using the Uberon ontology. 
The agent helps identify the most suitable Uberon ontology term for a given tissue description.

* Input: Free text description of one or more tissues
* Output: Uberon ontology IDs (UBERON:XXXXXXX) for each tissue description
* Workflow
  * The agent processes each tissue description separately
  * For each description, it identifies the most suitable Uberon ontology term
  * The agent returns the corresponding Uberon ID for each tissue

#### Examples

Categorize a single tissue:

```bash
SRAgent tissue-ontology "Categorize the following tissue: brain"
```

Categorize multiple tissues:

```bash
SRAgent tissue-ontology "Tissues: lung, heart, liver"
```

Finding ontology terms for complex tissue descriptions:

```bash
SRAgent tissue-ontology "Find the ontology term for the thin layer of epithelial cells lining the alveoli in lungs"
```

## Disease-ontology agent

An agent for categorizing disease descriptions using the MONDO/PATO ontology. 
The agent helps identify the most suitable MONDO or PATO ontology term for a given disease description.

* Input: Free text description of one or more diseases
* Output: MONDO/PATO ontology IDs for each disease description
* Workflow
  * The agent processes each disease description separately
  * For each description, it identifies the most suitable MONDO or PATO ontology term
  * The agent returns the corresponding ontology ID for each disease

#### Examples

Categorize a single disease:

```bash
SRAgent disease-ontology "Categorize the following disease: heart disorder"
```

Finding ontology terms for specific diseases:

```bash
SRAgent disease-ontology "What is the MONDO ontology ID for congestive heart failure?"
```

Categorize multiple diseases:

```bash
SRAgent disease-ontology "Diseases: heart neoplasm, bursitis"
```

## find-datasets agent

A high-level agent for finding datasets in the SRA via `esearch` and then
processing them with the `SRX-info` agent. 

* Input: a search query
* Output: metadata fields for the datasets found (same as `SRX-info` agent)
* Workflow
  * The agent uses `esearch` to find datasets
  * The agent processes the datasets with the `SRX-info` agent
  * The agent consolidates the metadata into a single report

#### Examples

```bash
SRAgent find-datasets --max-datasets 2 \
  "Obtain recent single cell RNA-seq datasets in the SRA database"
```

#### Target specific organisms

```bash
SRAgent --no-summaries \
  find-datasets --max-datasets 1 --organisms pig -- \
    "Obtain recent single cell RNA-seq datasets in the SRA database"
```

<details>
  <summary><strong>Available organisms</strong></summary>

- **Mammals**
  - Human (*Homo sapiens*)
  - Mouse (*Mus musculus*)
  - Rat (*Rattus norvegicus*)
  - Macaque (*Macaca mulatta*)
  - Marmoset (*Callithrix jacchus*)
  - Horse (*Equus caballus*)
  - Dog (*Canis lupus*)
  - Bovine (*Bos taurus*)
  - Sheep (*Ovis aries*)
  - Pig (*Sus scrofa*)
  - Rabbit (*Oryctolagus cuniculus*)
  - Naked mole-rat (*Heterocephalus glaber*)
  - Chimpanzee (*Pan troglodytes*)
  - Gorilla (*Gorilla gorilla*)
  - Cat (*Felis catus*)
  - Bonobo (*Pan paniscus*)
  - Green monkey (*Chlorocebus aethiops*)
  - Gray short-tailed opposum (*Monodelphis domestica*)
  - Goat (*Capra hircus*)
  - Alpaca (*Vicugna pacos*)
  - Chinchilla (*Chinchilla lanigera*)
  - Domestic guinea pig (*Cavia porcellus*)
  - Golden hamster (*Mesocricetus auratus*)
  - Eurasian hedgehog (*Erinaceus europaeus*)
  - American mink (*Neovison vison*)
  - Sunda pangolin (*Manis javanica*)
  - Platypus (*Ornithorhynchus anatinus*)
  - Ferret (*Mustela putorius*)
  - Northern tree shrew (*Tupaia belangeri*)
- **Birds**
  - Chicken (*Gallus gallus*)
  - Zebrafinch (*Taeniopygia guttata*)
  - Goose (*Anser cygnoides*)
  - Duck (*Anas platyrhynchos*)
- **Reptiles**
  - Turtle (*Trachemys scripta*)
- **Amphibians**
  - Frog (*Xenopus tropicalis*)
  - Axolotl (*Ambystoma mexicanum*)
- **Fish**
  - Zebrafish (*Danio rerio*)
  - Salmon (*Salmo salar*)
  - Stickleback (*Gasterosteus aculeatus*)
- **Invertebrates**
  - Fruit fly (*Drosophila melanogaster*)
  - Roundworm (*Caenorhabditis elegans*)
  - Mosquito (*Anopheles gambiae*)
  - Blood fluke (*Schistosoma mansoni*)
- **Plants**
  - Thale cress (*Arabidopsis thaliana*)
  - Rice (*Oryza sativa*)
  - Tomato (*Solanum lycopersicum*)
  - Corn (*Zea mays*)
- **Microorganisms**
  - Metagenome
- **Other**
  - Other

</details>

#### Using an SQL database to store results

Using the `test` database:

```bash
SRAgent --no-summaries \
  find-datasets --use-database --tenant test \
    --max-datasets 1 --organisms rat -- \
    "Obtain recent single cell RNA-seq datasets in the SRA database"
```

## Workflow status

By default, most SRAgent sub-commands (e.g., `entrez` and `sragent`) use "fancy" CLI output formatting:

#### Progress bar

<img src="./assets/cli1.png" width="600">

#### Final results

<img src="./assets/cli2.png" width="600">

#### Parameters for controlling the CLI workflow status output

* `--no-progress`: Disable progress bar
* `--no-summaries`: Disable LLM summaries for each workflowstep

Example:

```bash
SRAgent --no-summaries --no-progress \
  entrez "Convert GSE121737 to SRX accessions"
```

# Configuring models

The models used by SRAgent are configured in the [settings.yml](./SRAgent/settings.yml) file.
Options for updating the settings:

## 1) Provide a new settings file

* Create a new settings yaml file
* Set the `DYNACONF_SETTINGS_PATH` environment variable to the path to the new settings file
  * e.g., `export DYNACONF_SETTINGS_PATH=/path/to/settings.yml`
* No need to (re)install the package. The settings will be loaded from the new file.

## 2) Update and install

* Clone the repository
* Update the [settings.yml](./SRAgent/settings.yml) file
* (Re)install the the package
  * e.g., `pip install .`

## Using Claude models

SRAgent supports using Anthropic's Claude models:

* Set the `ANTHROPIC_API_KEY` environment variable to your Anthropic API key
* Switch to the Claude environment using `export DYNACONF=claude`
  * Update the [settings.yml](./SRAgent/settings.yml) file, as needed
  * See [Configuring models](#configuring-models) information on setting model parameters
* Run SRAgent commands as usual

Claude models support different reasoning effort levels:
* `low`: 1024 thinking tokens (best for simple tasks)
* `medium`: 4096 thinking tokens (good balance)
* `high`: 16384 thinking tokens (best for complex reasoning)
* Anything else: Disables thinking tokens feature

Example:
```bash
export ANTHROPIC_API_KEY=your_api_key
export DYNACONF="claude"
SRAgent entrez "Convert GSE121737 to SRX accessions"
```

You can also customize the specific Claude model in settings.yml:
```yaml
claude:
  models:
    default: "claude-sonnet-4-5"  # Or any other Claude model version
  temperature:
    default: 0.1
  reasoning_effort:
    default: "medium"  # Set your preferred reasoning effort; use "" to disable
```

# Setting up the SQL database

* Create a GCP Postgresql database. See the [docs](https://cloud.google.com/sql?hl=en).
* Required secrets:
  * `GCP_SQL_DB_PASSWORD`
  * Store in the `.env` file or GCP Secret Manager
  * If using GCP Secret Manager, you must also provide:
    * `GOOGLE_APPLICATION_CREDENTIALS`
    * `GCP_PROJECT_ID`
* Update the [settings.py](./SRAgent/settings.yml) file with the database information.

# Evaluations

See the [eval.py](./scripts/eval.py) script for running evaluations.


# Contributing

Feel free to fork the repository and submit a pull request.
However, the top priority is to keep SRAgent functioning 
for the ongoing scBaseCount project.
