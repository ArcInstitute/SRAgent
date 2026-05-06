// sragent-mcp — Supabase Edge Function (Deno)
//
// JSON-RPC over streamable-HTTP, no MCP SDK dependency. Mirrors the Python
// `sragent_mcp` package: same 5 tools, same Entrez query construction.
//
// SECRETS: NCBI credentials are read from Edge Function environment
// variables `NCBI_API_KEY` and `ENTREZ_EMAIL`. Set them via the Supabase
// dashboard (Edge Functions → Secrets) or via the `supabase secrets set`
// CLI. Never inline keys into this file.
//
// AUTH: deploy with `verify_jwt: false` if you want unauthenticated MCP
// clients (Claude Desktop, Claude.ai connectors) to be able to call it.
// The blast radius is limited to read-only NCBI E-utilities calls, but
// consider rate limits and your shared NCBI key carefully before exposing
// the URL widely.

const NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils";

const DEFAULT_QUERY_TERMS: string[] = [
  "RNA-seq", "RNA sequencing", "transcriptome sequencing", "mRNA-seq",
];

// ---- Normalization: organism (permissive — common-name shortcuts only) ----

const COMMON_NAME_TO_SCIENTIFIC: Record<string, string> = {
  "human": "Homo sapiens", "humans": "Homo sapiens",
  "h. sapiens": "Homo sapiens", "h sapiens": "Homo sapiens",
  "homo sapiens": "Homo sapiens",
  "mouse": "Mus musculus", "mus musculus": "Mus musculus",
  "rat": "Rattus norvegicus", "macaque": "Macaca mulatta",
  "marmoset": "Callithrix jacchus", "horse": "Equus caballus",
  "dog": "Canis lupus", "bovine": "Bos taurus", "cow": "Bos taurus",
  "sheep": "Ovis aries", "pig": "Sus scrofa",
  "rabbit": "Oryctolagus cuniculus",
  "chimpanzee": "Pan troglodytes", "gorilla": "Gorilla gorilla",
  "cat": "Felis catus", "bonobo": "Pan paniscus",
  "goat": "Capra hircus", "alpaca": "Vicugna pacos",
  "platypus": "Ornithorhynchus anatinus", "ferret": "Mustela putorius",
  "chicken": "Gallus gallus", "duck": "Anas platyrhynchos",
  "frog": "Xenopus tropicalis", "axolotl": "Ambystoma mexicanum",
  "zebrafish": "Danio rerio", "salmon": "Salmo salar",
  "fruit fly": "Drosophila melanogaster", "drosophila": "Drosophila melanogaster",
  "roundworm": "Caenorhabditis elegans", "c. elegans": "Caenorhabditis elegans",
  "mosquito": "Anopheles gambiae",
  "thale cress": "Arabidopsis thaliana", "arabidopsis": "Arabidopsis thaliana",
  "rice": "Oryza sativa", "tomato": "Solanum lycopersicum",
  "corn": "Zea mays", "maize": "Zea mays",
};
const SCIENTIFIC_NAME_RE = /^[A-Z][a-z]+(?:\s+[a-z]+){1,2}$/;

function normalizeOrganism(name: string): string {
  if (typeof name !== "string") {
    throw new TypeError(`organism must be a string, got ${typeof name}`);
  }
  const cleaned = name.trim();
  if (!cleaned) throw new Error("organism cannot be empty");
  const key = cleaned.toLowerCase();
  return COMMON_NAME_TO_SCIENTIFIC[key] ?? cleaned;
}

// ---- Normalization: date ----

const DATE_PARTS_RE = /^\s*(\d{4})(?:[-/.](\d{1,2}))?(?:[-/.](\d{1,2}))?\s*$/;
function lastDayOfMonth(y: number, m: number): number {
  const next = m === 12 ? new Date(y + 1, 0, 1) : new Date(y, m, 1);
  next.setDate(next.getDate() - 1);
  return next.getDate();
}
function normalizeDate(s: string, endOfPeriod = false): string {
  if (typeof s !== "string") throw new TypeError("date must be string");
  const m = DATE_PARTS_RE.exec(s);
  if (!m) throw new Error(`Could not parse date: ${JSON.stringify(s)}`);
  const y = parseInt(m[1], 10);
  const mo = m[2] === undefined ? (endOfPeriod ? 12 : 1) : parseInt(m[2], 10);
  if (mo < 1 || mo > 12) throw new Error(`Invalid month: ${mo}`);
  const d = m[3] === undefined ? (endOfPeriod ? lastDayOfMonth(y, mo) : 1) : parseInt(m[3], 10);
  if (d < 1 || d > 31) throw new Error(`Invalid day: ${d}`);
  return `${y.toString().padStart(4,"0")}/${mo.toString().padStart(2,"0")}/${d.toString().padStart(2,"0")}`;
}

// ---- Normalization: database ----

const DATABASE_ALIASES: Record<string, string> = {
  sra: "sra", gds: "gds", geo: "gds", gse: "gds", gpl: "gds",
};
function normalizeDatabase(db?: string | null): string {
  if (db === undefined || db === null || db === "") return "sra";
  const out = DATABASE_ALIASES[db.trim().toLowerCase()];
  if (out) return out;
  throw new Error(`database must be 'sra' or 'gds' (alias 'geo'); got '${db}'`);
}

// ---- Normalization: accession & entrez_id ----

const SRX_FULL_RE = /^(SRX|ERX|DRX)\d{4,}$/i;
const SRX_RE = /\b(SRX|ERX|DRX)\d{4,}\b/gi;
const SRR_RE = /\b(SRR|ERR|DRR)\d{4,}\b/gi;

function normalizeAccession(s: string): string {
  return s.trim().toUpperCase();
}
function isSrxAccession(s: string): boolean {
  return SRX_FULL_RE.test(normalizeAccession(s));
}
function normalizeEntrezId(eid: unknown): string {
  if (typeof eid === "number") {
    if (!Number.isInteger(eid) || eid < 0) throw new Error(`Entrez ID invalid: ${eid}`);
    return String(eid);
  }
  if (typeof eid === "string") {
    const c = eid.trim();
    if (!/^\d+$/.test(c)) throw new Error(`Entrez ID must be numeric: ${JSON.stringify(eid)}`);
    return c;
  }
  throw new TypeError(`Entrez ID must be int|string, got ${typeof eid}`);
}

// ---- Modality filters ----

interface ModalityFilter { extraAnds: string[]; exclusions: string[]; }
const MODALITY_FILTERS: Record<string, ModalityFilter> = {
  bulk_rna_seq: {
    extraAnds: [' AND "rna seq"[Strategy]'],
    exclusions: [
      '"single cell"[Title]', '"single-cell"[Title]',
      '"scRNA-seq"[Title]', '"snRNA-seq"[Title]',
      '"10X Genomics"[Title]', '"10x Genomics"[All Fields]',
      '"Smart-seq"[All Fields]', '"Drop-seq"[All Fields]',
      '"CEL-seq"[All Fields]',
    ],
  },
  scrna_seq: {
    extraAnds: [' AND "rna seq"[Strategy]'],
    exclusions: [],
  },
};

// ---- Schema (precomputed JSON schema for RNASeqMetadata) ----

const RNASEQ_SCHEMA = {
  $defs: {
    YesNo: { enum: ["yes","no","unsure"], type: "string", title: "YesNo" },
    Modality: { enum: ["bulk_rna_seq","scrna_seq","snrna_seq","other","unsure"], type: "string", title: "Modality" },
    LibrarySelectionEnum: { enum: ["poly-A_selection","ribo_depletion","total_RNA","cDNA","other","unsure","not_applicable"], type: "string", title: "LibrarySelectionEnum" },
    LibraryStrategyEnum: { enum: ["RNA-Seq","mRNA-Seq","ssRNA-seq","Total_RNA-Seq","other","unsure"], type: "string", title: "LibraryStrategyEnum" },
    StrandednessEnum: { enum: ["stranded_forward","stranded_reverse","unstranded","unsure","not_applicable"], type: "string", title: "StrandednessEnum" },
    ScRNALibPrepEnum: { enum: ["10x_Genomics","Smart-seq","Smart-seq2","Smart-seq3","Drop-seq","CEL-seq","CEL-seq2","indrops","Parse","BD_Rhapsody","other","not_applicable"], type: "string", title: "ScRNALibPrepEnum" },
    CellPrepEnum: { enum: ["single_cell","single_nucleus","unsure","not_applicable"], type: "string", title: "CellPrepEnum" },
  },
  type: "object",
  title: "RNASeqMetadata",
  description: "Per-experiment RNA-seq metadata fields a client should extract from SRA.",
  properties: {
    is_illumina: { $ref: "#/$defs/YesNo" },
    is_paired_end: { $ref: "#/$defs/YesNo" },
    modality: { $ref: "#/$defs/Modality" },
    organism: { type: "string" },
    instrument_model: { type: "string" },
    read_length: { type: "string" },
    library_strategy: { $ref: "#/$defs/LibraryStrategyEnum" },
    library_selection: { $ref: "#/$defs/LibrarySelectionEnum" },
    strandedness: { $ref: "#/$defs/StrandednessEnum" },
    scrna_lib_prep: { $ref: "#/$defs/ScRNALibPrepEnum" },
    cell_prep: { $ref: "#/$defs/CellPrepEnum" },
    tissue: { type: "string" },
    disease: { type: "string" },
    perturbation: { type: "string" },
    cell_line: { type: "string" },
    sample_condition: { type: "string" },
    srr_accessions: { type: "array", items: { type: "string" } },
  },
  required: [
    "is_illumina","is_paired_end","modality","organism","instrument_model","read_length",
    "library_strategy","library_selection","strandedness",
    "tissue","disease","perturbation","cell_line","sample_condition",
  ],
};

// ---- Entrez wrappers ----

function getEmail(): string {
  return Deno.env.get("ENTREZ_EMAIL") || Deno.env.get("EMAIL") || "anonymous@example.com";
}
function getApiKey(): string | undefined {
  return Deno.env.get("NCBI_API_KEY") || undefined;
}

function entrezUrl(endpoint: string, params: Record<string,string>): string {
  const u = new URL(`${NCBI_BASE}/${endpoint}.fcgi`);
  u.searchParams.set("email", getEmail());
  const k = getApiKey(); if (k) u.searchParams.set("api_key", k);
  for (const [pk, pv] of Object.entries(params)) u.searchParams.set(pk, pv);
  return u.toString();
}

interface BuildOpts {
  query_terms?: string[]; organisms?: string[];
  min_date?: string; max_date?: string;
  modality?: string | null;
  extra_exclusions?: string[];
}
function buildQuery(o: BuildOpts): string {
  const terms = o.query_terms?.length ? o.query_terms : DEFAULT_QUERY_TERMS;
  let q = `(${terms.map(t => `"${t}"`).join(" OR ")})`;
  if (o.min_date && o.max_date) {
    q += ` AND (${normalizeDate(o.min_date,false)}:${normalizeDate(o.max_date,true)}[PDAT])`;
  }
  if (o.organisms?.length) {
    const sci = o.organisms.filter(x=>x).map(normalizeOrganism);
    q += ` AND (${sci.map(n=>`"${n}"[Organism]`).join(" OR ")})`;
  }
  q += ' AND "public"[Access] AND "has data"[Properties]';
  q += ' AND "platform illumina"[Filter] AND "sra bioproject"[Filter]';
  const excl: string[] = [];
  if (o.modality) {
    const mf = MODALITY_FILTERS[o.modality];
    if (!mf) throw new Error(`Unknown modality '${o.modality}'`);
    for (const a of mf.extraAnds) q += a;
    excl.push(...mf.exclusions);
  }
  if (o.extra_exclusions) excl.push(...o.extra_exclusions);
  if (excl.length) q += ` NOT (${excl.join(" OR ")})`;
  return q;
}

async function esearchIds(args: BuildOpts & { max_ids?: number; database?: string }): Promise<string[]> {
  const db = normalizeDatabase(args.database);
  const term = buildQuery(args);
  const maxIds = Math.max(1, Math.min(args.max_ids ?? 25, 200));
  const url = entrezUrl("esearch", {
    db, term, retmax: String(maxIds), retmode: "json", sort: "pub+date",
  });
  const res = await fetch(url);
  if (!res.ok) throw new Error(`esearch HTTP ${res.status}`);
  const data = await res.json();
  return (data.esearchresult?.idlist ?? []).slice(0, maxIds);
}

async function efetchText(database: string, id: string, rettype="full", retmode="xml"): Promise<string> {
  const r = await fetch(entrezUrl("efetch", { db: database, id, rettype, retmode }));
  if (!r.ok) throw new Error(`efetch HTTP ${r.status}`);
  return await r.text();
}
async function esummaryText(database: string, id: string): Promise<string> {
  const r = await fetch(entrezUrl("esummary", { db: database, id, retmode: "json" }));
  if (!r.ok) throw new Error(`esummary HTTP ${r.status}`);
  return await r.text();
}

async function entrezIdToSrx(eid: string, db: string): Promise<string[]> {
  const id = normalizeEntrezId(eid);
  const ndb = normalizeDatabase(db);
  const text = ndb === "sra" ? await efetchText("sra", id) : await esummaryText("gds", id);
  const seen = new Set<string>(); const out: string[] = [];
  for (const m of text.matchAll(SRX_RE)) {
    const acc = m[0].toUpperCase();
    if (!seen.has(acc)) { seen.add(acc); out.push(acc); }
  }
  return out;
}

async function getSrxRuns(srxRaw: string): Promise<string[]> {
  const srx = normalizeAccession(srxRaw);
  if (!isSrxAccession(srx)) throw new Error(`Expected SRX/ERX/DRX; got '${srxRaw}'`);
  const r = await fetch(entrezUrl("esearch", { db:"sra", term:`${srx}[Accession]`, retmode:"json" }));
  if (!r.ok) throw new Error(`esearch HTTP ${r.status}`);
  const data = await r.json();
  const ids: string[] = data.esearchresult?.idlist ?? [];
  if (!ids.length) return [];
  const text = await efetchText("sra", ids[0]);
  const seen = new Set<string>(); const out: string[] = [];
  for (const m of text.matchAll(SRR_RE)) {
    const acc = m[0].toUpperCase();
    if (!seen.has(acc)) { seen.add(acc); out.push(acc); }
  }
  return out;
}

async function getSrxMetadataText(idOrAcc: string, db: string): Promise<{text:string; truncated:boolean}> {
  const ndb = normalizeDatabase(db);
  const s = idOrAcc.trim();
  let text: string;
  if (ndb === "sra") {
    if (isSrxAccession(s)) {
      const acc = normalizeAccession(s);
      const r = await fetch(entrezUrl("esearch", { db:"sra", term:`${acc}[Accession]`, retmode:"json" }));
      if (!r.ok) throw new Error(`esearch HTTP ${r.status}`);
      const data = await r.json();
      const ids: string[] = data.esearchresult?.idlist ?? [];
      if (!ids.length) return { text: `<error>No SRA record for ${acc}</error>`, truncated: false };
      text = await efetchText("sra", ids[0]);
    } else {
      text = await efetchText("sra", normalizeEntrezId(s));
    }
  } else {
    text = await esummaryText("gds", normalizeEntrezId(s));
  }
  if (text.length > 200_000) return { text: text.slice(0,200_000), truncated: true };
  return { text, truncated: false };
}

// ---- MCP tool definitions ----

const TOOLS = [
  {
    name: "find_rnaseq_datasets",
    description: "Search NCBI SRA / GEO for RNA-seq datasets. Returns Entrez IDs. Use `modality='bulk_rna_seq'` to exclude single-cell datasets.",
    inputSchema: {
      type: "object",
      properties: {
        query_terms: { type: "array", items: { type: "string" } },
        organisms: { type: "array", items: { type: "string" } },
        min_date: { type: "string", description: "YYYY/MM/DD, YYYY-MM-DD, YYYY-MM, or YYYY" },
        max_date: { type: "string", description: "Same formats; partial dates default to end-of-period" },
        max_ids: { type: "integer", default: 25 },
        database: { type: "string", enum: ["sra","gds","geo"], default: "sra" },
        modality: { type: "string", enum: ["bulk_rna_seq","scrna_seq"], description: "Optional modality narrowing" },
        extra_exclusions: { type: "array", items: { type: "string" }, description: "Additional NCBI-syntax NOT clauses" },
      },
    },
  },
  {
    name: "entrez_id_to_srx",
    description: "Resolve an Entrez ID to its SRX/ERX/DRX experiment accession(s).",
    inputSchema: {
      type: "object",
      properties: { entrez_id: { type: "string" }, database: { type: "string", enum: ["sra","gds","geo"], default: "sra" } },
      required: ["entrez_id"],
    },
  },
  {
    name: "get_srx_runs",
    description: "List SRR/ERR/DRR run accessions for an SRX/ERX/DRX experiment. Case-insensitive.",
    inputSchema: { type: "object", properties: { srx_accession: { type: "string" } }, required: ["srx_accession"] },
  },
  {
    name: "get_srx_metadata_text",
    description: "Return raw SRA experiment XML (or GEO esummary blob) for the given accession or Entrez ID. Truncated at ~200 KB.",
    inputSchema: {
      type: "object",
      properties: { srx_or_entrez_id: { type: "string" }, database: { type: "string", enum: ["sra","gds","geo"], default: "sra" } },
      required: ["srx_or_entrez_id"],
    },
  },
  {
    name: "get_rnaseq_metadata_schema",
    description: "Return the JSON schema for `RNASeqMetadata`. Use as the structured-output target when extracting metadata.",
    inputSchema: { type: "object", properties: { modality: { type: "string", description: "Informational; the schema is unified" } } },
  },
];

const RESOURCES = [
  { uri: "schema://rnaseq-metadata", name: "RNA-seq metadata schema",
    description: "JSON schema for RNA-seq metadata fields", mimeType: "application/json" },
];

// ---- Tool dispatch ----

async function callTool(name: string, args: Record<string, unknown>): Promise<unknown> {
  switch (name) {
    case "find_rnaseq_datasets": {
      const a = args as Parameters<typeof esearchIds>[0];
      const ids = await esearchIds(a);
      return {
        entrez_ids: ids, count: ids.length,
        database: normalizeDatabase((a as any).database),
        modality: (a as any).modality ?? null,
        query: buildQuery(a), default_terms_used: !a.query_terms,
      };
    }
    case "entrez_id_to_srx": {
      const a = args as { entrez_id: string; database?: string };
      const accs = await entrezIdToSrx(a.entrez_id, a.database ?? "sra");
      return { entrez_id: normalizeEntrezId(a.entrez_id), database: normalizeDatabase(a.database), srx_accessions: accs };
    }
    case "get_srx_runs": {
      const a = args as { srx_accession: string };
      const runs = await getSrxRuns(a.srx_accession);
      return { srx_accession: normalizeAccession(a.srx_accession), srr_accessions: runs, count: runs.length };
    }
    case "get_srx_metadata_text": {
      const a = args as { srx_or_entrez_id: string; database?: string };
      const r = await getSrxMetadataText(a.srx_or_entrez_id, a.database ?? "sra");
      return { id: a.srx_or_entrez_id, database: normalizeDatabase(a.database), ...r };
    }
    case "get_rnaseq_metadata_schema": {
      return {
        schema: RNASEQ_SCHEMA,
        default_search_terms: DEFAULT_QUERY_TERMS,
        modality: (args as any).modality ?? null,
        notes: "Set modality-specific fields (e.g. scrna_lib_prep, cell_prep) to 'not_applicable' for bulk RNA-seq. Use 'unsure' when evidence is ambiguous.",
      };
    }
    default:
      throw Object.assign(new Error(`Unknown tool: ${name}`), { code: -32601 });
  }
}

// ---- JSON-RPC ----

interface JsonRpcReq { jsonrpc: "2.0"; id?: string|number|null; method: string; params?: any; }

async function dispatch(req: JsonRpcReq): Promise<unknown> {
  switch (req.method) {
    case "initialize":
      return {
        protocolVersion: "2024-11-05",
        capabilities: { tools: { listChanged: false }, resources: { listChanged: false, subscribe: false } },
        serverInfo: { name: "sragent-mcp", version: "0.1.0" },
      };
    case "tools/list": return { tools: TOOLS };
    case "tools/call":
      return { content: [{ type: "text", text: JSON.stringify(await callTool(req.params?.name, req.params?.arguments ?? {}), null, 2) }], isError: false };
    case "resources/list": return { resources: RESOURCES };
    case "resources/read": {
      const uri = req.params?.uri;
      if (uri === "schema://rnaseq-metadata") {
        return { contents: [{ uri, mimeType: "application/json", text: JSON.stringify(RNASEQ_SCHEMA, null, 2) }] };
      }
      throw Object.assign(new Error(`Unknown resource: ${uri}`), { code: -32602 });
    }
    case "ping": return {};
    case "notifications/initialized": return null;
    default: throw Object.assign(new Error(`Method not found: ${req.method}`), { code: -32601 });
  }
}

// ---- HTTP entrypoint ----

function newSessionId(): string {
  return crypto.randomUUID().replaceAll("-", "");
}

Deno.serve(async (req: Request) => {
  if (req.method === "GET") {
    return new Response("sragent-mcp. POST JSON-RPC to this URL.\n", { headers: { "Content-Type": "text/plain" } });
  }
  if (req.method !== "POST") return new Response("Method not allowed", { status: 405 });

  let body: JsonRpcReq;
  try { body = await req.json(); }
  catch {
    return new Response(JSON.stringify({ jsonrpc: "2.0", id: null, error: { code: -32700, message: "Parse error" } }),
      { status: 400, headers: { "Content-Type": "application/json" } });
  }

  if (body.id === undefined || body.id === null) {
    try { await dispatch(body); } catch (_) { /* notifications swallow errors */ }
    return new Response(null, { status: 202 });
  }

  const accept = req.headers.get("accept") ?? "";
  const wantsSse = accept.includes("text/event-stream");
  const headers: Record<string, string> = {
    "Content-Type": wantsSse ? "text/event-stream" : "application/json",
    "Cache-Control": "no-cache",
  };
  if (body.method === "initialize") headers["mcp-session-id"] = newSessionId();

  let payload: unknown;
  try { payload = { jsonrpc: "2.0", id: body.id, result: await dispatch(body) }; }
  catch (err) {
    const e = err as { code?: number; message?: string };
    payload = { jsonrpc: "2.0", id: body.id, error: { code: e.code ?? -32603, message: e.message ?? String(err) } };
  }
  const json = JSON.stringify(payload);
  return new Response(wantsSse ? `event: message\ndata: ${json}\n\n` : json, { headers });
});
