"""Public web demo: a visitor pastes a raw RFQ email and watches the same
pipeline used by process_rfq.py run live -- extract.extract() ->
reviewer.review() -> (approve) offer.build_offer()+PDF or (flagged)
review_report.render_review_report(). Fully stateless per request: nothing
is written to output/ -- the PDF/report is returned directly in the HTTP
response (base64 for the PDF), never saved server-side.

Since this hits a real, billed OpenAI key on every request from anyone who
finds the URL, two guardrails run before extract.extract() is ever called:
a per-IP hourly rate limit and a global daily spend cap (DEMO_RATE_LIMIT_
PER_HOUR / DEMO_DAILY_BUDGET_USD env vars). Both are plain in-memory state
protected by a lock -- correct only with a single worker process, hence
`uvicorn ... --workers 1` in the Dockerfile. This is a small-traffic demo,
not a service meant to scale past that.

Run locally (from the project root, in an env with the native WeasyPrint
libs -- see generate_pdf.py):
    uvicorn web.main:app --reload
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import extract
from generate_pdf import render_offer_pdf_bytes
from offer import build_offer
from preview_html import render_offer_html
from review_report import render_review_report
from reviewer import review
from schemas.catalog import load_catalog
from schemas.review import OverallVerdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quotegen.web")

EMAILS_DIR = PROJECT_ROOT / "samples" / "emails"
MANIFEST_PATH = EMAILS_DIR / "manifest.json"
STATIC_DIR = WEB_DIR / "static"

DAILY_BUDGET_USD = float(os.environ.get("DEMO_DAILY_BUDGET_USD", "2.00"))
RATE_LIMIT_PER_HOUR = int(os.environ.get("DEMO_RATE_LIMIT_PER_HOUR", "5"))
MAX_EMAIL_CHARS = int(os.environ.get("DEMO_MAX_EMAIL_CHARS", "6000"))
RATE_WINDOW_SECONDS = 3600

app = FastAPI(title="Angebots-Automatisierung Demo")

# --------------------------------------------------------------------------
# In-memory guardrail state. See module docstring -- single worker only.
# --------------------------------------------------------------------------

_lock = threading.Lock()
_daily_spend: dict[str, float] = {}
_ip_requests: dict[str, list[float]] = {}


def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _check_rate_limit(ip: str) -> bool:
    now = time.time()
    with _lock:
        recent = [t for t in _ip_requests.get(ip, []) if now - t < RATE_WINDOW_SECONDS]
        if len(recent) >= RATE_LIMIT_PER_HOUR:
            _ip_requests[ip] = recent
            return False
        recent.append(now)
        _ip_requests[ip] = recent
        return True


def _budget_available() -> bool:
    with _lock:
        return _daily_spend.get(_today_key(), 0.0) < DAILY_BUDGET_USD


def _add_spend(amount: float) -> None:
    with _lock:
        key = _today_key()
        _daily_spend[key] = _daily_spend.get(key, 0.0) + amount


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    email_text: str


@app.get("/api/catalog")
def get_catalog() -> list[dict]:
    """The mock product catalog extraction matches against -- shown on the
    demo page so a visitor can judge for themselves whether a match makes
    sense, since these aren't real products they'd otherwise recognize."""
    return [item.model_dump(mode="json") for item in load_catalog()]


@app.get("/api/examples")
def get_examples() -> list[dict]:
    """Labeled sample emails (data/eval.jsonl's siblings) so a visitor can
    try the demo without writing a German RFQ from scratch."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    examples = []
    for entry in manifest["emails"]:
        text = (EMAILS_DIR / entry["file"]).read_text(encoding="utf-8")
        examples.append(
            {
                "file": entry["file"],
                "category": entry["category"],
                "notes": entry["notes"],
                "text": text,
            }
        )
    return examples


@app.post("/api/process")
def process_email(payload: ProcessRequest, request: Request) -> dict:
    email_text = payload.email_text.strip()
    if not email_text:
        raise HTTPException(400, "E-Mail-Text darf nicht leer sein.")
    if len(email_text) > MAX_EMAIL_CHARS:
        raise HTTPException(
            400, f"E-Mail-Text ist zu lang (max. {MAX_EMAIL_CHARS} Zeichen in dieser Demo)."
        )

    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            429, "Zu viele Anfragen von dieser Adresse -- bitte in einer Stunde erneut versuchen."
        )
    if not _budget_available():
        raise HTTPException(429, "Das taegliche Demo-Limit ist erreicht -- bitte morgen wieder versuchen.")

    try:
        run = extract.extract(email_text)
    except extract.ExtractionError as exc:
        logger.warning("extraction failed: %s", exc)
        raise HTTPException(502, "Die Extraktion ist fehlgeschlagen. Bitte spaeter erneut versuchen.")

    _add_spend(run.cost_usd)

    extraction = run.extraction
    verdict = review(extraction)
    reference = f"DEMO-{uuid4().hex[:8]}"

    result: dict = {
        "reference": reference,
        "extraction": extraction.model_dump(mode="json"),
        "verdict": verdict.model_dump(mode="json"),
        "usage": {
            "model": run.model,
            "cost_usd": run.cost_usd,
            "tokens_in": run.usage.input_tokens,
            "tokens_out": run.usage.output_tokens,
            "latency_seconds": run.latency_seconds,
        },
    }

    if verdict.overall_verdict == OverallVerdict.APPROVE:
        offer_data = build_offer(extraction, verdict, reference)
        pdf_bytes = render_offer_pdf_bytes(offer_data)
        result["result_type"] = "offer"
        result["offer_html_preview"] = render_offer_html(offer_data)
        result["pdf_base64"] = base64.b64encode(pdf_bytes).decode("ascii")
    else:
        result["result_type"] = "review"
        result["review_report_text"] = render_review_report(
            extraction, verdict, reference, source_label="web-demo"
        )

    return result


# Mounted last: a "/" mount matches any path not already claimed by a
# route registered above it, so /api/* keeps working.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
