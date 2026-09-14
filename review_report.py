"""Companion to offer.py: for an RFQ that reviewer.review() did NOT approve
(needs_human_review or reject -- covers ambiguous references, not_rfq
messages, and products outside the catalog alike, since all three surface
as flags on a non-approve verdict), build a plain-text report a human can
act on instead of a customer-facing PDF.

Feed the result to write_review_report() or just write the string
yourself -- render_review_report() has no file-system side effects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from schemas.extraction import ExtractionOutput
from schemas.review import MatchStatus, OverallVerdict, ReviewVerdict

SEPARATOR = "-" * 70


def _customer_block(extraction: ExtractionOutput) -> str:
    c = extraction.customer
    lines = [
        f"Firma:       {c.company_name or '-'}",
        f"Kontakt:     {c.contact_person or '-'}",
        f"E-Mail:      {c.email or '-'}",
        f"Telefon:     {c.phone or '-'}",
        f"Lieferadr.:  {c.delivery_address or '-'}",
    ]
    return "\n".join(lines)

def _line_item_block(pos: int, li) -> str:
    lines = [f"{pos}. [{li.match_status.value}] {li.raw_text!r}"]
    if li.candidate_article_numbers:
        lines.append(f"   Mögliche Artikel: {', '.join(li.candidate_article_numbers)}")
    if li.matched_article_number:
        lines.append(f"   Zuordnung: {li.matched_article_number} ({li.matched_article_name})")
    if li.quantity is not None:
        lines.append(f"   Menge: {li.quantity}")
    lines.append(f"   Konfidenz: {li.confidence:.2f}")
    if li.issue:
        lines.append(f"   Hinweis: {li.issue}")
    return "\n".join(lines)


def render_review_report(
    extraction: ExtractionOutput,
    verdict: ReviewVerdict,
    reference: str,
    *,
    source_label: str = "-",
    generated_at: datetime | None = None,
) -> str:
    """Render a human-readable report for a non-approve ReviewVerdict.

    Raises ValueError if verdict is APPROVE -- an approved RFQ goes through
    offer.build_offer() instead, never through here.
    """
    if verdict.overall_verdict == OverallVerdict.APPROVE:
        raise ValueError(
            "render_review_report is for needs_human_review/reject verdicts "
            "only -- an approved RFQ should go through offer.build_offer()"
        )

    generated_at = generated_at or datetime.now(timezone.utc)

    header = "\n".join(
        [
            "RFQ MANUELLE PRÜFUNG ERFORDERLICH",
            "=" * 70,
            f"Referenz:      {reference}",
            f"Erstellt:      {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"Quelle:        {source_label}",
            "",
            f"Gesamtverdikt: {verdict.overall_verdict.value}",
            f"Konfidenz:     {verdict.confidence:.2f}",
            f"Flags:         {', '.join(f.value for f in verdict.flags) or '-'}",
            f"Notizen:       {verdict.reviewer_notes or '-'}",
            "",
            f"Klassifikation: {extraction.classification.value} "
            f"(Konfidenz {extraction.confidence:.2f})",
            f"Begründung:    {extraction.classification_reason}",
        ]
    )

    sections = [header, SEPARATOR, "Kunde", SEPARATOR, _customer_block(extraction)]

    flagged = [
        (i, li)
        for i, li in enumerate(verdict.line_item_reviews, start=1)
        if li.match_status != MatchStatus.MATCHED
    ]
    matched = [
        (i, li)
        for i, li in enumerate(verdict.line_item_reviews, start=1)
        if li.match_status == MatchStatus.MATCHED
    ]

    if flagged:
        sections += [
            SEPARATOR,
            "Zu klärende Positionen",
            SEPARATOR,
            "\n\n".join(_line_item_block(i, li) for i, li in flagged),
        ]

    if matched:
        sections += [
            SEPARATOR,
            "Eindeutig zugeordnete Positionen (informativ, wegen obiger "
            "Flags nicht Teil eines Angebots)",
            SEPARATOR,
            "\n\n".join(_line_item_block(i, li) for i, li in matched),
        ]

    sections += [
        SEPARATOR,
        "Nächster Schritt",
        SEPARATOR,
        "Bitte die oben genannten Punkte prüfen/mit dem Kunden klären "
        "und die Anfrage danach manuell beantworten oder erneut durch "
        "process_rfq.py laufen lassen.",
        "",
    ]

    return "\n".join(sections)


def write_review_report(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
