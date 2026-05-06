# sragent-mcp

An MCP (Model Context Protocol) server that exposes NCBI SRA / GEO dataset
discovery and metadata fetch as deterministic tools — for use from Claude
Desktop, Claude Code, Claude.ai connectors, Cursor, or any MCP client.

The MCP intentionally keeps the LLM out of the loop: tools are simple
deterministic Entrez E-utilities calls. Structured extraction (e.g. parsing
metadata XML into a typed schema) happens client-side in the host LLM, with
`get_rnaseq_metadata_schema` providing the target shape.

## Tools

| Tool | What it does |
|------|--------------|
| `find_rnaseq_datasets` | Entrez `esearch` over SRA/GEO. Optional `modality` filter (`bulk_rna_seq` excludes single-cell markers; `scrna_seq` tightens to RNA-seq strategy). Returns Entrez IDs. |
| `entrez_id_to_srx` | Resolve an Entrez ID to its SRX/ERX/DRX experiment accession(s). |
| `get_srx_runs` | List SRR/ERR/DRR run accessions for an SRX. |
| `get_srx_metadata_text` | Raw SRA experiment XML (or GEO esummary) for client-side extraction. Truncated at ~200 KB. |
| `get_rnaseq_metadata_schema` | JSON schema describing the metadata fields a host LLM should populate. |

Plus a resource: `schema://rnaseq-metadata`.

## Install (Python, stdio transport — for Claude Desktop / Claude Code)

```bash
# from the SRAgent repo root
cd mcp_server
uv venv && source .venv/bin/activate
uv pip install -e .
```

Or install directly from a fork without cloning:

```bash
uv pip install "git+https://github.com/ArcInstitute/SRAgent.git#subdirectory=mcp_server"
# or to use uvx:
uvx --from "git+https://github.com/ArcInstitute/SRAgent.git#subdirectory=mcp_server" sragent-mcp
```

Run it:

```bash
sragent-mcp                                    # stdio (default)
sragent-mcp --transport streamable-http --port 8000   # remote HTTP
```

### Environment variables

| Var | Purpose |
|-----|---------|
| `NCBI_API_KEY` | Optional. Raises NCBI E-utils rate limit from 3 → 10 req/s. Get one at https://account.ncbi.nlm.nih.gov/. |
| `ENTREZ_EMAIL` | Strongly recommended by NCBI. Falls back to `EMAIL`, then `anonymous@example.com`. |

### Configure for Claude Desktop

```jsonc
// ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "sragent": {
      "command": "/path/to/.venv/bin/sragent-mcp",
      "env": { "ENTREZ_EMAIL": "you@example.com", "NCBI_API_KEY": "..." }
    }
  }
}
```

### Configure for Claude Code

```bash
claude mcp add sragent -- /path/to/.venv/bin/sragent-mcp
```

## Deploy as a remote MCP (Supabase Edge Functions)

For Claude.ai connectors and other URL-based MCP clients, deploy the
TypeScript port as a Supabase Edge Function:

```bash
# from this directory
supabase functions deploy sragent-mcp --no-verify-jwt --project-ref YOUR_REF \
  --import-map supabase/import_map.json supabase/index.ts

# set NCBI credentials as function secrets
supabase secrets set NCBI_API_KEY=... ENTREZ_EMAIL=you@example.com \
  --project-ref YOUR_REF
```

Endpoint: `https://YOUR_REF.supabase.co/functions/v1/sragent-mcp`

You can also expose your local stdio server through a Cloudflare tunnel
(`cloudflared tunnel --url http://127.0.0.1:8000 --http-host-header
"127.0.0.1:8000"`) for a quick public test URL — that's ephemeral but
good for trying things out.

## Tests

```bash
pytest tests/                          # all tests
pytest tests/ -m "not live"            # offline-only (~1s)
```

## Design notes

* No SDK / single-file Edge Function port. The TypeScript hand-rolls the
  JSON-RPC streamable-HTTP protocol so the deployed function has no npm
  dependencies and a small bundle size.
* No nested LLM. All tools are deterministic Entrez calls. Structured
  extraction is the host LLM's job.
* Permissive organism handling. Common-name shortcuts (`human`, `mouse`,
  ...) are normalized to scientific names; arbitrary inputs pass through
  with a "not recognized" flag so users can search any species.
* Modality filters are opt-in. Without `modality=...` the search is
  agnostic — useful for SRAgent's primary scRNA-seq workflows as well as
  bulk-RNA-seq use cases.

## License

MIT — same as the parent SRAgent repository.
