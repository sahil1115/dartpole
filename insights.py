# insights.py
# Corpus-level executive summary + key entities, generated on demand from the
# already-loaded Ollama model and the already-built Chroma vector store.

import json
import logging
import re

import requests

import config

logger = logging.getLogger(config.APP_NAME + ".Insights")

ENTITY_TYPES = {"person", "organization", "location", "date", "amount", "product", "other"}

_STOPWORDS = {
    "The", "This", "That", "These", "Those", "A", "An", "It", "Its", "In", "On", "At",
    "For", "With", "As", "Of", "To", "From", "By", "And", "Or", "But", "If", "When",
    "While", "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
}

_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}"
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
    r"|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}/\d{1,2}/\d{2,4})\b"
)
_AMOUNT_RE = re.compile(
    r"(?:[$€£¥]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|thousand|k|M|B))?\b"
    r"|\b\d[\d,]*(?:\.\d+)?\s?%)"
)
_ORG_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&.]*\s){0,4}[A-Z][A-Za-z0-9&.]*"
    r"\s(?:Inc|Ltd|LLC|Corp|Corporation|Company|University|Institute)\.?\b"
)
_PROPER_NOUN_RUN_RE = re.compile(r"\b(?:[A-Z][a-zA-Z]+(?:\s+|$)){2,3}")


def build_corpus_sample(vector_store, budget_chars):
    """Reads chunk text + metadata straight from Chroma (no embedding model
    needed) and assembles a representative, size-bounded sample of the whole
    indexed folder: round-robin across documents so every source contributes,
    ordered within each document by (page, chunk_index).
    """
    results = vector_store.get(include=["documents", "metadatas"])
    docs = results.get("documents", []) or []
    metas = results.get("metadatas", []) or []

    by_source = {}
    for text, meta in zip(docs, metas):
        if not text or not isinstance(meta, dict):
            continue
        source = meta.get("source", "Unknown")
        page = meta.get("page")
        page = int(page) if isinstance(page, (int, float)) or (isinstance(page, str) and page.isdigit()) else 0
        chunk_index = meta.get("chunk_index")
        chunk_index = chunk_index if isinstance(chunk_index, (int, float)) else 0
        by_source.setdefault(source, []).append((page, chunk_index, text))

    for source in by_source:
        by_source[source].sort(key=lambda t: (t[0], t[1]))

    doc_names = sorted(_basename(s) for s in by_source)

    queues = [(source, [t[2] for t in chunks]) for source, chunks in sorted(by_source.items())]
    parts = []
    used = 0
    started = {source: False for source, _ in queues}
    while used < budget_chars and any(q for _, q in queues):
        for source, queue in queues:
            if not queue or used >= budget_chars:
                continue
            chunk_text = queue.pop(0)
            piece = f"--- {_basename(source)} ---\n{chunk_text}\n" if not started[source] else f"{chunk_text}\n"
            started[source] = True
            parts.append(piece)
            used += len(piece)

    return "".join(parts), doc_names


def _basename(path):
    return str(path or "Unknown").replace("\\", "/").rsplit("/", 1)[-1]


def parse_entities_json(raw, max_entities):
    """Extracts and validates a JSON array of {text, type} entities from a raw
    LLM response. Returns None (never raises) if the response isn't usable,
    so the caller can fall back to the heuristic extractor.
    """
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, list):
        return None

    entities, seen = [], set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        entity_text = str(item.get("text", "")).strip()
        if not entity_text:
            continue
        key = entity_text.lower()
        if key in seen:
            continue
        seen.add(key)
        entity_type = str(item.get("type", "other")).strip().lower()
        if entity_type not in ENTITY_TYPES:
            entity_type = "other"
        entities.append({"text": entity_text, "type": entity_type})
        if len(entities) >= max_entities:
            break
    return entities if entities else None


def extract_entities_heuristic(text, max_entities):
    """Zero-dependency fallback: regex-based entity spotting. Used when the
    LLM's JSON response can't be parsed.
    """
    counts = {}

    def add(entity_text, entity_type):
        entity_text = entity_text.strip().strip(".,;:")
        if not entity_text or entity_text in _STOPWORDS:
            return
        key = (entity_text.lower(), entity_type)
        if key not in counts:
            counts[key] = [entity_text, entity_type, 0]
        counts[key][2] += 1

    org_texts = []
    for m in _ORG_RE.finditer(text):
        org_text = m.group(0).strip().strip(".,;:")
        org_texts.append(org_text.lower())
        add(org_text, "organization")
    for m in _DATE_RE.finditer(text):
        add(m.group(0), "date")
    for m in _AMOUNT_RE.finditer(text):
        add(m.group(0), "amount")
    for m in _PROPER_NOUN_RUN_RE.finditer(text):
        candidate = m.group(0).strip()
        first_word = candidate.split(" ", 1)[0]
        if first_word in _STOPWORDS:
            continue
        # Skip names already captured as an organization (e.g. "Acme Corp"
        # matches both the org suffix pattern and the generic proper-noun run).
        candidate_lower = candidate.lower()
        if any(candidate_lower in org or org in candidate_lower for org in org_texts):
            continue
        add(candidate, "person")

    # Only keep proper-noun runs that recur — a single mention is too weak a
    # signal without real NER, but 2+ mentions across the sample is a good filter.
    ranked = [v for v in counts.values() if v[1] != "person" or v[2] >= 2]
    ranked.sort(key=lambda v: v[2], reverse=True)
    return [{"text": t, "type": ty} for t, ty, _ in ranked[:max_entities]]


def _invoke_no_think(llm_client, prompt, format_schema=None):
    """Calls Ollama's /api/generate directly instead of going through
    langchain_community's deprecated Ollama LLM wrapper.

    That wrapper never surfaces Ollama's "think" control, and models with
    thinking enabled by default (e.g. qwen3.5) spend their entire generation
    budget on chain-of-thought and return an empty final response once the
    context fills up. Setting "think": false on the request is the fix;
    the wrapper has no way to pass that field through, so this talks to the
    Ollama HTTP API directly using the same connection settings already
    configured on llm_client.

    format_schema, when given, is passed as Ollama's "format" field (a JSON
    schema) — the server then constrains generation so the response is
    always valid JSON matching that shape, eliminating the parse-failure
    class entirely rather than working around it after the fact.
    """
    payload = {
        "model": llm_client.model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": llm_client.temperature,
            "num_ctx": llm_client.num_ctx,
            "stop": llm_client.stop,
        },
        "keep_alive": llm_client.keep_alive,
    }
    if format_schema is not None:
        payload["format"] = format_schema
    try:
        resp = requests.post(
            f"{llm_client.base_url}/api/generate",
            json=payload,
            timeout=llm_client.timeout or config.OLLAMA_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach Ollama model '{llm_client.model}': {e}") from e
    return (resp.json().get("response") or "").strip()


def generate_insights(llm_client, vector_store):
    """Runs the two-call insight pipeline (summary, then entities) against a
    single shared corpus sample and returns the response payload.
    """
    excerpts, doc_names = build_corpus_sample(vector_store, config.INSIGHTS_CONTEXT_CHARS)
    if not excerpts.strip():
        return {
            "summary": "No document text was available to summarize.",
            "entities": [],
            "entity_source": "heuristic",
            "documents": doc_names,
        }

    summary_prompt = config.INSIGHTS_SUMMARY_PROMPT.replace("{document_names}", ", ".join(doc_names)).replace("{excerpts}", excerpts)
    summary = _invoke_no_think(llm_client, summary_prompt)
    if not summary:
        summary = "The model did not return a summary. Try Refresh, or pick a different model."

    entities_prompt = config.INSIGHTS_ENTITIES_PROMPT.replace("{excerpts}", excerpts)
    raw_entities = _invoke_no_think(llm_client, entities_prompt)
    entities = parse_entities_json(raw_entities, config.INSIGHTS_MAX_ENTITIES)
    entity_source = "llm"
    if entities is None:
        logger.warning("LLM entity JSON could not be parsed; falling back to heuristic extraction.")
        entities = extract_entities_heuristic(excerpts, config.INSIGHTS_MAX_ENTITIES)
        entity_source = "heuristic"

    return {
        "summary": summary,
        "entities": entities,
        "entity_source": entity_source,
        "documents": doc_names,
    }
