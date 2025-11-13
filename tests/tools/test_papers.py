import pytest

from SRAgent.tools import papers


def test_download_paper_by_doi_preprint_success(monkeypatch, tmp_path):
    """Ensure preprint downloads short-circuit the fallback chain."""
    calls = {}

    def fake_preprint(doi: str, output_path: str):
        calls["preprint_args"] = (doi, output_path)
        return {"success": True, "message": "Downloaded", "source": "bioRxiv"}

    monkeypatch.setattr(papers, "_download_from_preprint_server", fake_preprint)

    result = papers.download_paper_by_doi(
        doi="10.1101/2025.02.27.640494",
        output_path=str(tmp_path / "paper.pdf"),
    )

    assert result == f"Successfully downloaded from bioRxiv to {tmp_path / 'paper.pdf'}"
    assert calls["preprint_args"][0] == "10.1101/2025.02.27.640494"


def test_download_paper_by_doi_all_sources_fail(monkeypatch, tmp_path):
    """Verify aggregated error reporting when every source fails."""

    def failing_preprint(doi: str, output_path: str):
        return {"success": False, "message": "not available"}

    monkeypatch.setattr(papers, "_download_from_preprint_server", failing_preprint)
    monkeypatch.setattr(papers, "_get_core_info", lambda doi, api_key=None: None)
    monkeypatch.setattr(papers, "_get_europepmc_info", lambda doi: None)
    monkeypatch.setattr(papers, "_get_unpaywall_info", lambda doi, email=None: None)
    monkeypatch.delenv("CORE_API_KEY", raising=False)

    message = papers.download_paper_by_doi(
        doi="10.48550/arXiv.12345",
        output_path=str(tmp_path / "paper.pdf"),
    )

    assert message.startswith("ERROR: Failed to download 10.48550/arXiv.12345")
    assert "Preprint server: not available" in message
    assert "CORE: Skipped" in message
    assert "Europe PMC: Article not found" in message
    assert "Unpaywall: Skipped" in message
