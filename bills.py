# bills.py
# Bill Analysis & Forecasting — extracts discrete bill records (vendor,
# category, amount, dates) from indexed financial documents and aggregates
# them into category spend totals and a due-date forecast.
#
# Money has a much lower error tolerance than prose: a summary that is 85%
# right is still useful, a bill table that is 85% right means the user has
# to re-check every row by hand. So the LLM is never trusted to read or
# compute a number itself. Deterministic regex + dateutil first enumerate
# every plausible amount/date in the document; the LLM's only job is to
# pick which of those candidates is the total due, the due date, etc. — a
# choice enforced via the request's JSON-schema "enum", so the model is
# structurally unable to return a value that wasn't already found in the
# text. A field the model couldn't match to a real candidate is reported
# as low-confidence, not silently accepted.

import hashlib
import logging
import re
from datetime import date, timedelta

from dateutil import parser as dateutil_parser

import config
import insights

logger = logging.getLogger(config.APP_NAME + ".Bills")

_CURRENCY_SYM = r"[$€£¥₹]"
_CURRENCY_CODE = r"(?:USD|EUR|GBP|JPY|INR|CAD|AUD|CHF)"
_NUMBER = r"\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{1,2})?|\d+(?:[,.]\d{1,2})?"

_AMOUNT_RE = re.compile(
    rf"(?P<paren>\()?"
    rf"(?P<sign>-\s?)?"
    rf"(?:(?P<pre_sym>{_CURRENCY_SYM})\s?|(?P<pre_code>{_CURRENCY_CODE})\s+)?"
    rf"(?P<number>{_NUMBER})"
    rf"(?:\s?(?P<post_sym>{_CURRENCY_SYM})|\s+(?P<post_code>{_CURRENCY_CODE}))?"
    rf"(?P<close>\))?"
)

_MONTH_NAMES = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
    r"|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
_DATE_RE = re.compile(
    rf"\b(?:\d{{4}}-\d{{1,2}}-\d{{1,2}}"
    rf"|\d{{1,2}}\s+(?:{_MONTH_NAMES})\w*,?\s+\d{{4}}"
    rf"|(?:{_MONTH_NAMES})\w*\s+\d{{1,2}},?\s+\d{{4}}"
    rf"|\d{{1,2}}[/.\-]\d{{1,2}}[/.\-]\d{{2,4}})\b"
)

_BILL_KEYWORDS_RE = re.compile(
    r"(?i)\b(?:due date|payment due|amount due|balance due|invoice|statement|"
    r"account number|bill(?:ing)?|subscription|premium|autopay|remit)\b"
)


def _normalize_amount(raw_number):
    """Resolves whether '.' or ',' is the decimal separator in a matched
    number, so "$1.234,56" (EU) and "$1,234.56" (US) both come out as
    1234.56, and "1,234" (US thousands, no decimal) doesn't get silently
    misread as "1.234".
    """
    s = raw_number.strip()
    last_comma = s.rfind(",")
    last_dot = s.rfind(".")
    decimal_sep = thousands_sep = None

    if last_comma != -1 and last_dot != -1:
        if last_comma > last_dot:
            decimal_sep, thousands_sep = ",", "."
        else:
            decimal_sep, thousands_sep = ".", ","
    elif last_comma != -1:
        digits_after = len(s) - last_comma - 1
        if digits_after in (1, 2):
            decimal_sep = ","
        else:
            thousands_sep = ","
    elif last_dot != -1:
        digits_after = len(s) - last_dot - 1
        if digits_after in (1, 2):
            decimal_sep = "."
        else:
            thousands_sep = "."

    if thousands_sep:
        s = s.replace(thousands_sep, "")
    if decimal_sep and decimal_sep != ".":
        s = s.replace(decimal_sep, ".")
    s = s.replace(" ", "")
    return float(s)


def _looks_like_money(match):
    """A bare number (no currency symbol/code) is only kept as a candidate
    if it has a decimal-looking tail — otherwise page numbers, years, and
    account numbers would flood the candidate list.
    """
    if match.group("pre_sym") or match.group("pre_code") or match.group("post_sym") or match.group("post_code"):
        return True
    number = match.group("number")
    last_sep = max(number.rfind(","), number.rfind("."))
    if last_sep == -1:
        return False
    return len(number) - last_sep - 1 in (1, 2)


def find_amount_candidates(text, max_candidates=None):
    """Returns every plausible monetary value in text, each as
    {"raw": exact matched substring, "value": signed float, "line": context}.
    Deliberately over-inclusive — a false-positive candidate is harmless
    (the LLM just never picks it); a missed true value is not.
    """
    max_candidates = max_candidates or config.BILLS_MAX_CANDIDATES
    candidates = []
    seen_raw = set()
    for m in _AMOUNT_RE.finditer(text):
        if not _looks_like_money(m):
            continue
        raw = m.group(0).strip()
        if raw in seen_raw:
            continue
        try:
            value = _normalize_amount(m.group("number"))
        except ValueError:
            continue
        if m.group("sign") or m.group("paren") or m.group("close"):
            value = -abs(value)
        seen_raw.add(raw)
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        line_end = line_end if line_end != -1 else len(text)
        candidates.append({"raw": raw, "value": value, "line": text[line_start:line_end].strip()})
        if len(candidates) >= max_candidates:
            break
    return candidates


def _detect_dayfirst(text):
    """Cheap locale heuristic: EU-style currency or dotted numeric dates
    imply day-first; otherwise default to month-first (US)."""
    if re.search(r"[€£]", text):
        return True
    if re.search(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b", text):
        return True
    return False


def find_date_candidates(text, max_candidates=None):
    """Returns every plausible date in text as {"raw", "iso"}, resolved
    with an explicit dayfirst so "03/04/2025" doesn't silently swap day
    and month depending on dateutil's own default.
    """
    max_candidates = max_candidates or config.BILLS_MAX_CANDIDATES
    dayfirst = _detect_dayfirst(text)
    candidates = []
    seen_raw = set()
    for m in _DATE_RE.finditer(text):
        raw = m.group(0)
        if raw in seen_raw:
            continue
        try:
            parsed = dateutil_parser.parse(raw, dayfirst=dayfirst, fuzzy=False)
        except (ValueError, OverflowError):
            continue
        seen_raw.add(raw)
        candidates.append({"raw": raw, "iso": parsed.date().isoformat()})
        if len(candidates) >= max_candidates:
            break
    return candidates


def looks_like_bill(text):
    """Cheap pre-scan (no LLM cost) used to decide whether the Bills tab
    should even appear for this corpus."""
    if not text or not _BILL_KEYWORDS_RE.search(text):
        return False
    return bool(find_amount_candidates(text, max_candidates=1))


def _stitch_overlaps(chunks):
    """Joins consecutive chunks of the same page back into the original
    page text. RecursiveCharacterTextSplitter keeps separators attached to
    splits, so the overlap between two adjacent chunks is an exact
    suffix/prefix substring — find it and drop the duplicate instead of
    concatenating naively (which reintroduces up to CHUNK_OVERLAP chars of
    duplication per boundary). When stripping consumed the separator
    entirely, no overlap exists; fall back to joining with a blank line.
    """
    if not chunks:
        return ""
    result = chunks[0]
    for nxt in chunks[1:]:
        if not nxt:
            continue
        max_check = min(len(result), len(nxt), config.CHUNK_OVERLAP)
        overlap_len = 0
        for length in range(max_check, 0, -1):
            if result[-length:] == nxt[:length]:
                overlap_len = length
                break
        result = result + nxt[overlap_len:] if overlap_len else result + "\n\n" + nxt
    return result


def load_document_text(vector_store, source):
    """Reassembles the full text of one indexed document, in original page
    order, from its stored chunks. Unlike insights.build_corpus_sample()
    (which round-robins one chunk per source across the whole corpus), this
    reads only `source` and stitches it back into contiguous text — bill
    extraction needs the whole document, not a cross-document sample.
    """
    results = vector_store.get(where={"source": source}, include=["documents", "metadatas"])
    docs = results.get("documents", []) or []
    metas = results.get("metadatas", []) or []

    by_page = {}
    for text, meta in zip(docs, metas):
        if not text or not isinstance(meta, dict):
            continue
        page = meta.get("page")
        page = int(page) if isinstance(page, (int, float)) or (isinstance(page, str) and page.isdigit()) else 0
        chunk_index = meta.get("chunk_index")
        chunk_index = chunk_index if isinstance(chunk_index, (int, float)) else 0
        by_page.setdefault(page, []).append((chunk_index, text))

    pages = []
    for page in sorted(by_page.keys()):
        ordered_chunks = [t for _, t in sorted(by_page[page], key=lambda pair: pair[0])]
        pages.append(_stitch_overlaps(ordered_chunks))
    return "\n\n".join(pages)


def _make_id(source):
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


def _safe_json(raw):
    import json
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def _build_schema(amount_options, date_options):
    return {
        "type": "object",
        "properties": {
            "is_bill": {"type": "boolean"},
            "vendor": {"type": "string"},
            "category": {"type": "string", "enum": config.BILLS_CATEGORIES},
            "total_due": {"type": "string", "enum": amount_options + [""]},
            "due_date": {"type": "string", "enum": date_options + [""]},
            "issue_date": {"type": "string", "enum": date_options + [""]},
        },
        "required": ["is_bill", "vendor", "category", "total_due", "due_date", "issue_date"],
    }


def validate_record(parsed, amount_candidates, date_candidates, source):
    """Cross-checks the model's chosen strings against the real candidate
    set. The JSON-schema enum should make an out-of-set value impossible,
    but this is the load-bearing safety net if that guarantee ever doesn't
    hold (an older Ollama, an empty enum, etc.) — a field that can't be
    matched is reported as not-found rather than silently trusted.
    """
    amount_by_raw = {c["raw"]: c for c in amount_candidates}
    date_by_raw = {c["raw"]: c for c in date_candidates}

    total_raw = (parsed.get("total_due") or "").strip()
    amount_match = amount_by_raw.get(total_raw)

    due_raw = (parsed.get("due_date") or "").strip()
    due_match = date_by_raw.get(due_raw)

    issue_raw = (parsed.get("issue_date") or "").strip()
    issue_match = date_by_raw.get(issue_raw)

    category = parsed.get("category")
    if category not in config.BILLS_CATEGORIES:
        category = "Other"

    return {
        "id": _make_id(source),
        "source": source,
        "is_bill": True,
        "vendor": (parsed.get("vendor") or "Unknown").strip() or "Unknown",
        "category": category,
        "amount": amount_match["value"] if amount_match else None,
        "amount_raw": total_raw or None,
        "amount_confident": amount_match is not None,
        "due_date": due_match["iso"] if due_match else None,
        "due_date_raw": due_raw or None,
        "due_date_confident": due_match is not None,
        "issue_date": issue_match["iso"] if issue_match else None,
        "issue_date_raw": issue_raw or None,
        "issue_date_confident": issue_match is not None,
    }


def extract_bill_from_document(llm_client, vector_store, source):
    """Runs the full per-document pipeline: reassemble text, enumerate
    candidates, ask the model to pick roles, validate. Returns None if the
    document has no text; returns {"source", "is_bill": False} if it isn't
    a bill; otherwise a full validated record.
    """
    text = load_document_text(vector_store, source)
    if not text.strip():
        return None
    if not looks_like_bill(text):
        return {"source": source, "is_bill": False}

    amount_candidates = find_amount_candidates(text)
    date_candidates = find_date_candidates(text)
    if not amount_candidates:
        return {"source": source, "is_bill": False}

    amount_options = [c["raw"] for c in amount_candidates]
    date_options = [c["raw"] for c in date_candidates]

    prompt = (
        config.BILLS_EXTRACTION_PROMPT
        .replace("{document_text}", text[: config.BILLS_MAX_DOC_CHARS])
        .replace("{amount_candidates}", "\n".join(f"- {a}" for a in amount_options))
        .replace("{date_candidates}", "\n".join(f"- {d}" for d in date_options) or "(none found)")
    )
    schema = _build_schema(amount_options, date_options)

    raw = insights._invoke_no_think(llm_client, prompt, format_schema=schema)
    parsed = _safe_json(raw)
    if not parsed or not parsed.get("is_bill"):
        return {"source": source, "is_bill": False}

    return validate_record(parsed, amount_candidates, date_candidates, source)


def compute_forecast(records, today=None, window_days=None):
    """Pure aggregation over already-validated records: spend by category,
    plus bills due within `window_days` of `today`. Past-due items are
    reported separately rather than folded into "upcoming" — a folder of a
    year's bills is mostly history, and silently mixing overdue items into
    a forward-looking list would misrepresent what's actually owed next.
    """
    today = today or date.today()
    window_days = window_days if window_days is not None else config.BILLS_FORECAST_WINDOW_DAYS
    horizon = today + timedelta(days=window_days)

    category_totals = {}
    upcoming = []
    past_due = []

    for r in records:
        if not r or not r.get("is_bill"):
            continue
        amount = r.get("amount")
        if amount is not None:
            cat = r.get("category") or "Other"
            category_totals[cat] = category_totals.get(cat, 0.0) + amount

        due_iso = r.get("due_date")
        if not due_iso:
            continue
        try:
            due = date.fromisoformat(due_iso)
        except ValueError:
            continue
        if due < today:
            past_due.append(r)
        elif due <= horizon:
            upcoming.append(r)

    upcoming.sort(key=lambda r: r["due_date"])
    past_due.sort(key=lambda r: r["due_date"])

    return {
        "category_totals": category_totals,
        "upcoming": upcoming,
        "past_due": past_due,
        "window_days": window_days,
        "as_of": today.isoformat(),
    }
