import asyncio
import pytest

from SRAgent.agents import papers


@pytest.mark.asyncio
async def test_process_accession_no_publications(monkeypatch):
    """Return early when the agent finds no publications."""

    class DummyAgent:
        async def ainvoke(self, *args, **kwargs):
            return {"structured_response": {"publications": []}}

    monkeypatch.setattr(papers, "create_papers_agent", lambda return_tool=False: DummyAgent())

    result = await papers.process_accession("SRX000001")

    assert result["accession"] == "SRX000001"
    assert result["pubmed_ids"] == []
    assert result["summary"] == "No publications found"


@pytest.mark.asyncio
async def test_process_accession_with_downloads(monkeypatch, tmp_path):
    """Ensure DOIs trigger downloads and summary reflects success."""

    class DummyAgent:
        async def ainvoke(self, *args, **kwargs):
            return {
                "structured_response": {
                    "publications": [
                        {"pubmed_id": "12345678", "doi": "10.1000/example"},
                        {"pubmed_id": "87654321", "doi": None},
                    ]
                }
            }

    async def fake_download_batch(dois, output_dir, api_key=None, email=None):
        fake_result = {}
        for pmid, doi in dois.items():
            fake_result[pmid] = {
                "status": "success",
                "doi": doi,
                "path": f"{output_dir}/{pmid}.pdf",
                "error": None,
            }
        return fake_result

    monkeypatch.setattr(papers, "create_papers_agent", lambda return_tool=False: DummyAgent())
    monkeypatch.setattr(papers, "_download_papers_batch", fake_download_batch)

    result = await papers.process_accession(
        "SRX000002",
        output_base_dir=str(tmp_path),
        api_key="core",
        email="user@example.org",
    )

    assert result["pubmed_ids"] == ["12345678", "87654321"]
    assert result["downloads"]["12345678"]["status"] == "success"
    assert result["downloads"]["12345678"]["path"].endswith("/12345678.pdf")
    # DOI without value should be skipped for download
    assert "87654321" not in result["downloads"]
    assert "downloaded 1/1" in result["summary"]
