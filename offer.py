"""Phase 5: build the final offer dict -- the shape templates/angebot.html.jinja
expects, same as data/sample_offer.json -- from a real ExtractionOutput
(extract.py) plus its ReviewVerdict (review.py). Feed the result to
generate_pdf.render_offer_data() or preview_html.render_offer_html().

This only ever runs for an approved quote: build_offer() refuses anything
that isn't OverallVerdict.APPROVE, since a needs_human_review/reject verdict
must go to a person, not straight to a customer-facing PDF.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from reviewer import CATALOG_INDEX
from schemas.extraction import ExtractionOutput
from schemas.review import OverallVerdict, ReviewVerdict

BASE_DIR = Path(__file__).resolve().parent
ANBIETER_PATH = BASE_DIR / "data" / "anbieter.json"

MWST_SATZ = 19  # % -- German standard VAT rate
GUELTIGKEIT_TAGE = 30  # default quote validity window

FREITEXT_EINLEITUNG = (
    "vielen Dank für Ihre Anfrage. Gerne unterbreiten wir Ihnen folgendes Angebot:"
)
FREITEXT_SCHLUSS = (
    "Wir hoffen, Ihnen mit diesem Angebot dienlich zu sein und freuen uns auf Ihre Bestellung."
)
ZAHLUNGSBEDINGUNGEN = "Zahlbar innerhalb 14 Tagen ohne Abzug nach Rechnungsstellung."
LIEFERZEIT = "Lieferung erfolgt ca. 3-4 Wochen nach Auftragseingang, sofern nicht anders vermerkt."


def _load_anbieter() -> dict:
    return json.loads(ANBIETER_PATH.read_text(encoding="utf-8"))


def build_offer(
    extraction: ExtractionOutput,
    verdict: ReviewVerdict,
    angebotsnummer: str,
    *,
    heute: date | None = None,
) -> dict:
    """Turn an approved (extraction, verdict) pair into the dict
    templates/angebot.html.jinja expects. Raises ValueError if verdict is
    not OverallVerdict.APPROVE.
    """
    if verdict.overall_verdict != OverallVerdict.APPROVE:
        raise ValueError(
            f"Cannot build a customer-facing offer from a "
            f"{verdict.overall_verdict.value} verdict -- needs human review first"
        )

    heute = heute or date.today()
    gueltig_bis = heute + timedelta(days=GUELTIGKEIT_TAGE)

    positionen = []
    for pos, item in enumerate(verdict.line_item_reviews, start=1):
        catalog_item = CATALOG_INDEX[item.matched_article_number]
        positionen.append(
            {
                "pos": pos,
                "artikelnummer": item.matched_article_number,
                "bezeichnung": item.matched_article_name,
                "menge": item.quantity,
                "einheit": catalog_item.einheit,
                "einzelpreis": item.unit_price_eur,
                "gesamtpreis": item.line_total_eur,
            }
        )

    nettosumme = verdict.subtotal_eur
    mwst_betrag = round(nettosumme * MWST_SATZ / 100, 2)
    bruttosumme = round(nettosumme + mwst_betrag, 2)

    customer = extraction.customer

    return {
        "angebotsnummer": angebotsnummer,
        "datum": heute.strftime("%d.%m.%Y"),
        "gueltig_bis": gueltig_bis.strftime("%d.%m.%Y"),
        "kunden_referenz": extraction.reference_number,
        "kunde": {
            "firma": customer.company_name or "",
            "ansprechpartner": customer.contact_person or "",
            # We deliberately never guess Herr/Frau from a name -- the
            # extraction stage doesn't tell us a salutation, so this stays
            # gender-neutral rather than inventing one.
            "anrede_brief": "Sehr geehrte Damen und Herren,",
            # ExtractionOutput.customer only carries one free-text
            # delivery_address field, not separate street/city -- put it on
            # one line rather than guessing where it splits.
            "strasse": customer.delivery_address or "",
            "plz_ort": "",
        },
        "lieferadresse": None,
        "anbieter": _load_anbieter(),
        "freitext_einleitung": FREITEXT_EINLEITUNG,
        "freitext_schluss": FREITEXT_SCHLUSS,
        "positionen": positionen,
        "nettosumme": nettosumme,
        "mwst_satz": MWST_SATZ,
        "mwst_betrag": mwst_betrag,
        "bruttosumme": bruttosumme,
        "zahlungsbedingungen": ZAHLUNGSBEDINGUNGEN,
        "lieferzeit": LIEFERZEIT,
    }
