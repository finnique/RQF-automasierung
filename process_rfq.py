"""Phase 6: end-to-end orchestration. Runs one raw RFQ email through the
full pipeline -- extract.extract() -> reviewer.review() -> (only if
approved) offer.build_offer() -> generate_pdf.render_offer_data() -- and
reports what happened at every stage.

Usage:
    python process_rfq.py samples/emails/email_01_clean.txt
    python process_rfq.py samples/emails/email_01_clean.txt --html
    python process_rfq.py -   # read email text from stdin

Requires OPENAI_API_KEY / EXTRACT_MODEL to be set (see .env.example) --
this makes one real, billed API call per invocation.

On approve: writes output/angebot_<nummer>.pdf (and .html preview with
--html), prints the angebotsnummer and subtotal, exits 0.
On needs_human_review / reject (covers ambiguous references, not_rfq
messages, and non-catalog products alike -- see reviewer.py): does NOT
touch offer.py/generate_pdf.py at all, since offer.build_offer() refuses
non-approved verdicts by design. Instead writes a human-actionable report
via review_report.py to output/review_queue/ and exits 1.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import extract
from generate_pdf import render_offer_data
from offer import build_offer
from preview_html import render_offer_html
from review_report import render_review_report, write_review_report
from reviewer import review
from schemas.review import OverallVerdict

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
REVIEW_QUEUE_DIR = OUTPUT_DIR / "review_queue"

ANGEBOTSNUMMER_RE = re.compile(r"^angebot_A-(\d{4})-(\d+)\.(?:pdf|html)$")
REVIEW_REF_RE = re.compile(r"^review_R-(\d{4})-(\d+)\.txt$")


def _next_reference(prefix: str, directory: Path, pattern: re.Pattern, heute: date) -> str:
    """<prefix>-<year>-<4-digit seq>. No database -- the target directory
    is the source of truth for what's already been issued, so the
    sequence continues from the highest existing file for the current
    year and resets on year change."""
    year = heute.year
    highest = 0
    if directory.is_dir():
        for path in directory.iterdir():
            m = pattern.match(path.name)
            if m and int(m.group(1)) == year:
                highest = max(highest, int(m.group(2)))
    return f"{prefix}-{year}-{highest + 1:04d}"


def _read_email_text(source: str) -> str:
    if source == "-":
        return sys.stdin.read()
    return Path(source).read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("email_file", help="Path to a raw RFQ email .txt file, or '-' for stdin")
    parser.add_argument("--html", action="store_true", help="Also write an HTML preview alongside the PDF")
    args = parser.parse_args()

    email_text = _read_email_text(args.email_file)

    try:
        run = extract.extract(email_text)
    except extract.ExtractionError as exc:
        print(f"Extraction failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    extraction = run.extraction
    print(
        f"Extracted ({run.model}, {run.usage.total_tokens} tokens, "
        f"${run.cost_usd:.4f}, {run.latency_seconds:.1f}s): "
        f"classification={extraction.classification.value}, "
        f"{len(extraction.line_items)} line item(s)"
    )

    verdict = review(extraction)
    print(f"Review verdict: {verdict.overall_verdict.value}")

    if verdict.overall_verdict != OverallVerdict.APPROVE:
        reference = _next_reference("R", REVIEW_QUEUE_DIR, REVIEW_REF_RE, date.today())
        report_text = render_review_report(
            extraction, verdict, reference, source_label=args.email_file
        )
        report_path = REVIEW_QUEUE_DIR / f"review_{reference}.txt"
        write_review_report(report_text, report_path)

        print(f"Flags: {', '.join(f.value for f in verdict.flags) or '-'}")
        print(f"Needs human review -- report written to {report_path}", file=sys.stderr)
        return 1

    angebotsnummer = _next_reference("A", OUTPUT_DIR, ANGEBOTSNUMMER_RE, date.today())
    offer_data = build_offer(extraction, verdict, angebotsnummer)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / f"angebot_{angebotsnummer}.pdf"
    render_offer_data(offer_data, pdf_path)

    if args.html:
        html_path = OUTPUT_DIR / f"angebot_{angebotsnummer}.html"
        html_path.write_text(render_offer_html(offer_data), encoding="utf-8")
        print(f"Wrote {html_path}")

    print(f"Angebot {angebotsnummer}: {verdict.subtotal_eur:.2f} EUR netto -> {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
