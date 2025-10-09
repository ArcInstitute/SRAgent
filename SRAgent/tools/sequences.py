# import
## batteries
from __future__ import annotations
import os
import time
import tempfile
from shutil import which
from typing import Annotated

## 3rd party
from langchain_core.tools import tool

## package
from SRAgent.tools.utils import run_cmd, truncate_values, xml2json


# functions
@tool
def fastq_dump(
    SRR_accessions: Annotated[
        list[str], "List of SRA run accessions (e.g., SRR1234567)"
    ],
    tries: Annotated[int, "Number of attempts to run fastq-dump"] = 3,
) -> str:
    """
    Use fastq-dump to download the first few lines from the fastq files of the given SRR accession.
    The tool is useful for quickly checking the fastq files of an SRR accession.
    """
    # check that fastq-dump is installed
    if not which("fastq-dump"):
        return "fastq-dump is not installed. Please install SRA Tools"

    # check if accession is valid
    incorrect_accessions = [x for x in SRR_accessions if not x.startswith("SRR")]
    if len(incorrect_accessions) > 0:
        acc_str = ", ".join(incorrect_accessions)
        return f"Invalid SRA accession numbers {acc_str}. Please provide >=1 valid SRR accession number."

    # create temp directory
    temp_dir = tempfile.TemporaryDirectory()

    # create command
    cmd = [
        "fastq-dump",
        "--outdir",
        temp_dir.name,
        "--split-files",
        "--maxSpotId",
        2,
    ] + SRR_accessions

    # run command
    for i in range(tries):
        return_code, output, error = run_cmd(cmd)
        if return_code == 0:
            break
        time.sleep(5 * (i + 1))
    if return_code != 0:
        return f"Error running fastq-dump: {error.decode('utf-8')}"

    # read in the files
    files = os.listdir(temp_dir.name)
    if len(files) == 0:
        return "No FASTQ files found."
    fastq_data = ""
    for file in files:
        file_name = os.path.basename(file)
        with open(os.path.join(temp_dir.name, file), "r") as f:
            fastq_data += f"#-- File: {file_name} --#\n"
            fastq_data += f.read() + "\n"

    # delete the temp directory
    temp_dir.cleanup()
    return str(fastq_data)


@tool
def sra_stat(
    accessions: Annotated[
        list[str],
        "List of GEO and/or SRA accessions (e.g., SRP359840, SRR1234567, or GSE12345)",
    ],
    tries: Annotated[int, "Number of attempts to run sra-stat"] = 3,
) -> str:
    """
    Run the sra-stat CLI command (SRA Tools) on a GEO or SRA accession.
    Use this tool to get information about all sequence data associated with the accession.
    """
    # check that sra-stat is installed
    if not which("sra-stat"):
        return "sra-stat is not installed. Please install SRA Tools"

    # check if accession is valid
    incorrect_accessions = [
        x for x in accessions if not x.startswith(("SRP", "SRX", "SRR", "GSE", "GSM"))
    ]
    if len(incorrect_accessions) > 0:
        acc_str = ", ".join(incorrect_accessions)
        return f"Invalid GEO/SRA accession numbers {acc_str}. Please provide >=1 valid GEO and/or SRA accession."

    # run sra-stat
    cmd = ["sra-stat", "--xml", "--quick"] + accessions

    # run command
    for i in range(tries):
        return_code, output, error = run_cmd(cmd)
        if return_code == 0:
            break
        time.sleep(5 * (i + 1))
    if return_code != 0:
        return f"Error running sra-stat: {error.decode('utf-8')}"

    # Decode the record if necessary
    if isinstance(output, bytes):
        try:
            output = output.decode("utf-8")
        except Exception as e:
            return f"Decoding error: {e}"

    # Truncate long values in the record
    output = truncate_values(output, max_length=1000)

    # convert to XML to JSON
    output = xml2json(output)
    return str(output)


if __name__ == "__main__":
    # fastq-dump
    input = {"SRR_accessions": ["SRR13112659", "SRR13112660"]}
    # input = {"SRR_accessions" : ["ERR12363157"]}
    print(fastq_dump.invoke(input))

    # sra-stat
    input = {"accessions": ["SRP359840", "GSE12345"]}
    # print(sra_stat.invoke(input))
