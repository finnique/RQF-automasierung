"""Phase 3: the first LLM stage. Turns a raw German RFQ email into a
validated ExtractionOutput (schemas/extraction.py), including a
best-effort catalog match per line item. No pricing or arithmetic
happens here -- that's the later validation/review stage.

SDK note (read this before "fixing" anything below): as of openai-python
3.8.0, the Responses API is the recommended path for schema-constrained
output, via `client.responses.parse(..., text_format=PydanticModel)`,
read back as `response.output_parsed`. This is NOT the older
`chat.completions.create(response_format=...)` shape from earlier SDK
versions -- the parameter names, return shape, and even the timeout type
(`httpx2.Timeout`, not `httpx.Timeout`) changed. Every signature used
below was confirmed against the installed package
(openai.resources.responses.responses.Responses.parse,
openai.types.responses.ParsedResponse, openai.types.responses.
response_usage.ResponseUsage) rather than written from memory, per the
project's own instruction to double check -- if you upgrade the SDK,
re-verify before assuming this still holds.

Env vars (read from the real process environment first; falls back to a
.env file in the project root via python-dotenv -- copy .env.example to
.env and fill in your key, .env itself is gitignored and never read by
anything other than this process):
    OPENAI_API_KEY           required by the SDK itself
    EXTRACT_MODEL             required, e.g. "gpt-5-mini" (no default --
                               cost varies 100x across models, so we do not
                               silently pick one for you)
    EXTRACT_TIMEOUT_SECONDS   optional, default 60 -- hard per-attempt timeout
    EXTRACT_MAX_ATTEMPTS      optional, default 5 -- total attempts incl. first
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import openai
import httpx2
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from schemas.catalog import load_catalog
from schemas.extraction import ExtractionOutput

BASE_DIR = Path(__file__).resolve().parent

# Loads .env into os.environ if present. Does NOT override a variable
# that's already set in the real environment (e.g. exported in your
# shell, or injected by CI/a deploy platform) -- .env is a local-dev
# convenience, not a source of truth that should shadow production config.
load_dotenv(BASE_DIR / ".env")

RUNS_LOG_PATH = BASE_DIR / "data" / "runs.jsonl"


# --------------------------------------------------------------------------
# Typed exceptions -- callers should catch ExtractionError (or a subclass)
# rather than raw openai.* exceptions, so infra failures, timeouts, and a
# genuine "the model couldn't produce valid output" are distinguishable.
# --------------------------------------------------------------------------

class ExtractionError(Exception):
    """Base class for all extraction-stage failures."""


class ExtractionTimeoutError(ExtractionError):
    """Every attempt timed out (EXTRACT_TIMEOUT_SECONDS exceeded)."""


class ExtractionAPIError(ExtractionError):
    """The OpenAI API call failed after retries, or failed with a
    non-retryable error (bad request, auth, etc). Infra/config problem,
    not a statement about the email's content."""


class ExtractionValidationError(ExtractionError):
    """The API responded, but produced no usable structured output --
    a refusal, an incomplete response, or output that failed schema
    validation. This is the "genuine extraction failure" case."""


# --------------------------------------------------------------------------
# Per-model pricing, USD per 1M tokens. Update this dict as OpenAI changes
# prices -- https://platform.openai.com/docs/pricing (checked 2026-09-07).
# Add a model here before pointing EXTRACT_MODEL at it; compute_cost raises
# rather than silently returning a wrong number for an unlisted model.
# --------------------------------------------------------------------------

MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-6-astra":     {"input": 10.00, "output": 50.00, "cached_input": 1.00},
    "gpt-5.6-sol":     {"input": 4.00,  "output": 20.00, "cached_input": 0.40},
    "gpt-5.6-terra":   {"input": 2.00,  "output": 12.00, "cached_input": 0.20},
    "gpt-5.6-luna":    {"input": 0.20,  "output": 1.20,  "cached_input": 0.02},
    "gpt-5.5":         {"input": 5.00,  "output": 30.00, "cached_input": 0.50},
    "gpt-5.5-pro":     {"input": 30.00, "output": 180.00, "cached_input": 30.00},
    "gpt-5.4":         {"input": 2.50,  "output": 15.00, "cached_input": 0.25},
    "gpt-5.4-mini":    {"input": 0.75,  "output": 4.50,  "cached_input": 0.075},
    "gpt-5.4-nano":    {"input": 0.20,  "output": 1.25,  "cached_input": 0.02},
    "gpt-5":           {"input": 1.25,  "output": 10.00, "cached_input": 0.125},
    "gpt-5-mini":      {"input": 0.25,  "output": 2.00,  "cached_input": 0.025},
    "gpt-4o":          {"input": 2.50,  "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini":     {"input": 0.15,  "output": 0.60,  "cached_input": 0.075},
    "o1":              {"input": 15.00, "output": 60.00, "cached_input": 7.50},
    "o1-pro":          {"input": 150.00, "output": 600.00, "cached_input": 150.00},
    "o3":              {"input": 2.00,  "output": 8.00,  "cached_input": 0.50},
    "o3-mini":         {"input": 1.10,  "output": 4.40,  "cached_input": 0.55},
}


# Models found (at runtime, per-process) to reject the `temperature`
# param entirely rather than just ignoring it -- populated lazily by
# extract()'s _call(), not hand-maintained here.
_NO_TEMPERATURE_MODELS: set[str] = set()


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in MODEL_PRICING:
        raise ExtractionError(
            f"No pricing entry for model {model!r} in MODEL_PRICING -- add one "
            f"in extract.py before using this model, so cost isn't silently wrong."
        )
    rates = MODEL_PRICING[model]
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]


# --------------------------------------------------------------------------
# Prompt. The catalog block is generated from data/catalog.csv at import
# time (via schemas.catalog.load_catalog) so it can't drift out of sync
# with the actual catalog -- edit catalog.csv, not this string, to change
# what products the model knows about.
# --------------------------------------------------------------------------

def _build_catalog_block() -> str:
    items = load_catalog()
    lines = [f"{item.artikelnummer} | {item.bezeichnung} | Einheit: {item.einheit}" for item in items]
    return "\n".join(lines)


CATALOG_BLOCK = _build_catalog_block()

EXTRACTION_PROMPT_TEMPLATE = """\
Du extrahierst strukturierte Daten aus deutschen Geschaeftsanfragen (E-Mails), \
die potenziell Angebotsanfragen (RFQs) fuer elektrotechnische Komponenten sind.

## Aufgabe

1. Klassifiziere die Nachricht: ist es eine Angebotsanfrage ("rfq"), eindeutig \
keine Angebotsanfrage ("not_rfq", z.B. Reklamation, Bewerbung, Werbung), oder \
unklar ("unclear")?
2. Falls es eine Angebotsanfrage ist (oder sein koennte), extrahiere jede \
angefragte Position als eigenes line_item.
3. Fuer jede Position: versuche, die Kundenformulierung einem Artikel aus dem \
Katalog unten zuzuordnen (matched_article_number). Wenn du dir nicht sicher \
bist -- z.B. weil mehrere Katalogartikel gleichermassen passen koennten, weil \
der Kunde einen Fremdhersteller/Konkurrenzartikel nennt ohne dass eine \
eindeutige Entsprechung im Katalog existiert, oder weil die Beschreibung zu \
vage ist -- setze matched_article_number auf null und unresolved auf true. \
RATE NIEMALS. Eine falsche Zuordnung ist schlimmer als keine.
4. Wenn eine Information nicht in der Nachricht steht, gib null zurueck. \
Erfinde keine Werte (auch keine Mengen, Firmennamen oder Daten).
5. Alle Textwerte (Namen, Adressen, Freitext) bleiben auf Deutsch und werden \
wortgetreu wie im Original uebernommen -- nicht uebersetzen, nicht umformulieren.
6. Berechne NICHTS. Keine Preise, keine Summen, keine Mengen-Umrechnungen \
(z.B. Meter in Stangen). Gib Mengen und Einheiten exakt so wieder, wie der \
Kunde sie geschrieben hat.

## Katalog (Artikelnummer | Bezeichnung | Einheit)

{catalog_block}

## Hinweise zur Zuordnung

- Der Katalog enthaelt nur die oben gelisteten Artikel. Produkte, die klar \
ausserhalb dieses Sortiments liegen (z.B. Frequenzumrichter, SPS, Motoren, \
Kabel als Meterware ausserhalb der Kabelkanal-Artikel), sind grundsaetzlich \
unresolved -- auch wenn die Anfrage selbst eine echte RFQ ist.
- match_confidence spiegelt deine Sicherheit ueber die Artikelzuordnung wider \
(0 = kein Vertrauen, 1 = eindeutig), nicht die Dringlichkeit oder Wichtigkeit \
der Position.
- Nutze das notes-Feld, um kurz zu begruenden, warum eine Position unresolved \
ist (z.B. "zwei moegliche Artikel: RK-2001 oder RK-2031") oder um zusaetzliche \
vom Kunden genannte Qualifizierer festzuhalten (Farbe, Spannung, Groesse, \
Marke), die nicht in einem anderen Feld unterkommen.
- Existiert fuer die vom Kunden beschriebene Produktkategorie nur ein \
einziger Artikel im Katalog, ordne diesem zu (unresolved=false) -- auch wenn \
der Kunde nicht jedes technische Detail genannt hat (z.B. Spulenspannung, \
exakte Laenge, Farbe). Der Katalog bietet dann ohnehin keine Alternative, zu \
der es sich zu verwechseln lohnt. Setze unresolved nur dann, wenn \
mindestens zwei Katalogartikel fuer die vom Kunden genannten Merkmale \
gleichermassen plausibel bleiben (echte Mehrdeutigkeit), oder wenn gar kein \
Katalogartikel zur beschriebenen Kategorie passt.
- Nennt der Kunde fuer dieselbe Position mehrere Zahlen (z.B. eine \
pauschale Stueckangabe wie "1x" und eine konkrete Mengenangabe in Klammern \
wie "10 Meter"), setze quantity auf die Zahl, die sich auf die \
Katalog-Einheit dieser Position bezieht -- nicht automatisch auf die \
zuerst genannte Zahl. Das ist keine Mengen-Umrechnung (siehe Punkt 6), \
sondern nur die Wahl der richtigen bereits vom Kunden genannten Zahl.
"""

EXTRACTION_SYSTEM_PROMPT = EXTRACTION_PROMPT_TEMPLATE.format(catalog_block=CATALOG_BLOCK)


# --------------------------------------------------------------------------
# Usage / run result types
# --------------------------------------------------------------------------

class ExtractionUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int = 0


class ExtractionRun(BaseModel):
    """What extract() returns: the validated extraction plus everything
    needed to display/log cost and performance for this call."""

    extraction: ExtractionOutput
    model: str
    usage: ExtractionUsage
    cost_usd: float
    latency_seconds: float
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# --------------------------------------------------------------------------
# Run logger
# --------------------------------------------------------------------------

def log_run(run: ExtractionRun) -> None:
    """Append one JSON line to data/runs.jsonl for this call."""
    RUNS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": run.timestamp,
        "model": run.model,
        "tokens_in": run.usage.input_tokens,
        "tokens_out": run.usage.output_tokens,
        "cost_usd": round(run.cost_usd, 6),
        "latency_seconds": round(run.latency_seconds, 3),
        "extraction": run.extraction.model_dump(mode="json"),
    }
    with RUNS_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Retry helper. We own the retry policy explicitly (client max_retries=0)
# rather than relying on the SDK's built-in retry, because that only
# covers server-side statuses (408/409/429/5xx) and connection errors --
# a client-side timeout from our own `timeout=` budget is a separate
# exception (openai.APITimeoutError) that we want covered by the same
# backoff loop and the same attempt count.
# --------------------------------------------------------------------------

def _call_with_retry(fn, *, max_attempts: int, base_delay: float):
    attempt = 0
    last_exc: Exception
    while True:
        attempt += 1
        try:
            return fn()
        except openai.APITimeoutError as exc:
            last_exc = exc
            is_timeout = True
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as exc:
            last_exc = exc
            is_timeout = False

        if attempt >= max_attempts:
            if is_timeout:
                raise ExtractionTimeoutError(
                    f"Timed out on all {attempt} attempt(s)"
                ) from last_exc
            raise ExtractionAPIError(
                f"OpenAI API call failed after {attempt} attempt(s): {last_exc}"
            ) from last_exc

        time.sleep(base_delay * (2 ** (attempt - 1)))


def _get_client(timeout_seconds: float) -> openai.OpenAI:
    return openai.OpenAI(
        timeout=httpx2.Timeout(timeout_seconds),
        max_retries=0,  # we do our own retry/backoff in _call_with_retry
    )


def _refusal_text(response) -> Optional[str]:
    for output in response.output:
        if getattr(output, "type", None) != "message":
            continue
        for content in output.content:
            if getattr(content, "type", None) == "refusal":
                return content.refusal
    return None


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def extract(email_text: str) -> ExtractionRun:
    """Extract structured RFQ data from a raw German email.

    Raises ExtractionTimeoutError, ExtractionAPIError, or
    ExtractionValidationError (all subclasses of ExtractionError) on
    failure. On success, logs the call to data/runs.jsonl and returns an
    ExtractionRun.
    """
    if not email_text or not email_text.strip():
        raise ValueError("email_text must be non-empty")

    model = os.environ.get("EXTRACT_MODEL")
    if not model:
        raise ExtractionError(
            "EXTRACT_MODEL env var is not set -- pick a model name from "
            "MODEL_PRICING (or add pricing for a new one) and set it."
        )

    timeout_seconds = float(os.environ.get("EXTRACT_TIMEOUT_SECONDS", "60"))
    max_attempts = int(os.environ.get("EXTRACT_MAX_ATTEMPTS", "5"))

    client = _get_client(timeout_seconds)

    def _call():
        kwargs = dict(
            model=model,
            input=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": email_text},
            ],
            text_format=ExtractionOutput,
        )
        if model not in _NO_TEMPERATURE_MODELS:
            kwargs["temperature"] = 0
        try:
            return client.responses.parse(**kwargs)
        except openai.BadRequestError as exc:
            # Some models (reasoning-tier ones so far) reject `temperature`
            # outright rather than just ignoring it. Remember that per
            # model so later calls in this process skip straight to the
            # no-temperature request instead of paying for a failed one.
            if "temperature" in str(exc) and "not supported" in str(exc):
                _NO_TEMPERATURE_MODELS.add(model)
                kwargs.pop("temperature", None)
                return client.responses.parse(**kwargs)
            raise

    start = time.monotonic()
    try:
        response = _call_with_retry(_call, max_attempts=max_attempts, base_delay=2.0)
    except (ExtractionTimeoutError, ExtractionAPIError):
        raise
    except openai.APIError as exc:
        # Non-retryable SDK errors (bad request, auth, etc) -- never leak a
        # raw openai.* exception past this module.
        raise ExtractionAPIError(f"OpenAI API error: {exc}") from exc
    latency = time.monotonic() - start

    if getattr(response, "status", None) == "incomplete":
        reason = getattr(response.incomplete_details, "reason", "unknown")
        raise ExtractionValidationError(f"Response incomplete: {reason}")

    try:
        parsed = response.output_parsed
    except ValidationError as exc:
        raise ExtractionValidationError(f"Output failed schema validation: {exc}") from exc

    if parsed is None:
        refusal = _refusal_text(response)
        raise ExtractionValidationError(
            f"Model produced no parseable output"
            + (f" (refusal: {refusal})" if refusal else "")
        )

    usage = response.usage
    extraction_usage = ExtractionUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cached_input_tokens=getattr(usage.input_tokens_details, "cached_tokens", 0) or 0,
    )
    cost_usd = compute_cost(model, extraction_usage.input_tokens, extraction_usage.output_tokens)

    run = ExtractionRun(
        extraction=parsed,
        model=model,
        usage=extraction_usage,
        cost_usd=cost_usd,
        latency_seconds=latency,
    )
    log_run(run)
    return run
