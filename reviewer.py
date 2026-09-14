"""Phase 4: the review/pricing stage. Turns an ExtractionOutput (extract.py)
plus catalog data (schemas/catalog.py) into a ReviewVerdict (schemas/review.py)
-- a match/pricing decision per line item, plus one overall
approve/needs_human_review/reject call for the whole RFQ.

Deterministic only -- no LLM call. All arithmetic (unit price lookup, line
totals, subtotal) happens here, off catalog.csv, per the "extraction computes
nothing" rule documented in extract.py. See schemas/review.py for field
semantics.
"""

from __future__ import annotations

from schemas.catalog import CatalogItem, load_catalog
from schemas.extraction import ExtractedLineItem, ExtractionOutput, RequestClassification
from schemas.review import LineItemReview, MatchStatus, OverallVerdict, ReviewFlag, ReviewVerdict

# Thresholds -- conservative defaults, favoring needs_human_review over a
# wrong auto-send. Revisit once scripts/eval_extract.py has baseline numbers
# to tune against.
LOW_LINE_CONFIDENCE_THRESHOLD = 0.75  # a model-resolved match below this is
# downgraded to ambiguous instead of trusted outright.
LOW_CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.6


def _catalog_index() -> dict[str, CatalogItem]:
    return {item.artikelnummer: item for item in load_catalog()}


CATALOG_INDEX = _catalog_index()


def _find_candidate_articles(text: str | None) -> list[str]:
    """Best-effort: pull any catalog article numbers literally mentioned in
    free text (e.g. a notes field like 'zwei moegliche Artikel: RK-2001 oder
    RK-2031'). Exact, case-insensitive match against known article numbers
    only -- no generic regex -- so this never invents a false candidate."""
    if not text:
        return []
    upper = text.upper()
    return [num for num in CATALOG_INDEX if num.upper() in upper]


def _review_line_item(li: ExtractedLineItem) -> LineItemReview:
    if not li.unresolved:
        article = CATALOG_INDEX.get(li.matched_article_number)
        if article is None:
            # Model named an article number that isn't actually in our
            # catalog -- never trust that blindly.
            return LineItemReview(
                raw_text=li.raw_text,
                match_status=MatchStatus.NOT_FOUND,
                confidence=li.match_confidence,
                quantity=li.quantity,
                issue=f"Modell nannte unbekannte Artikelnummer '{li.matched_article_number}'",
            )
        if li.match_confidence < LOW_LINE_CONFIDENCE_THRESHOLD:
            return LineItemReview(
                raw_text=li.raw_text,
                match_status=MatchStatus.AMBIGUOUS,
                candidate_article_numbers=[article.artikelnummer],
                confidence=li.match_confidence,
                quantity=li.quantity,
                issue=(
                    f"Konfidenz der Zuordnung ({li.match_confidence:.2f}) unter "
                    f"Schwellenwert ({LOW_LINE_CONFIDENCE_THRESHOLD}) -- zur "
                    "Bestaetigung vorlegen"
                ),
            )
        unit_price = article.einzelpreis_eur
        line_total = unit_price * li.quantity if li.quantity is not None else None
        return LineItemReview(
            raw_text=li.raw_text,
            match_status=MatchStatus.MATCHED,
            matched_article_number=article.artikelnummer,
            matched_article_name=article.bezeichnung,
            confidence=li.match_confidence,
            quantity=li.quantity,
            unit_price_eur=unit_price,
            line_total_eur=line_total,
            issue=None if li.quantity is not None else "Menge nicht angegeben",
        )

    candidates = _find_candidate_articles(li.notes) or _find_candidate_articles(li.raw_text)
    if candidates:
        return LineItemReview(
            raw_text=li.raw_text,
            match_status=MatchStatus.AMBIGUOUS,
            candidate_article_numbers=candidates,
            confidence=li.match_confidence,
            quantity=li.quantity,
            issue=li.notes or "Mehrere moegliche Katalogartikel",
        )
    return LineItemReview(
        raw_text=li.raw_text,
        match_status=MatchStatus.NOT_FOUND,
        confidence=li.match_confidence,
        quantity=li.quantity,
        issue=li.notes or "Kein passender Katalogartikel gefunden",
    )


def review(extraction: ExtractionOutput) -> ReviewVerdict:
    """Deterministically turn one ExtractionOutput into a ReviewVerdict."""

    if extraction.classification == RequestClassification.NOT_RFQ:
        return ReviewVerdict(
            overall_verdict=OverallVerdict.REJECT,
            confidence=extraction.confidence,
            flags=[ReviewFlag.NOT_A_QUOTE_REQUEST],
            reviewer_notes=f"Keine Angebotsanfrage: {extraction.classification_reason}",
        )

    line_item_reviews = [_review_line_item(li) for li in extraction.line_items]

    flags: list[ReviewFlag] = []
    if (
        extraction.classification == RequestClassification.UNCLEAR
        or extraction.confidence < LOW_CLASSIFICATION_CONFIDENCE_THRESHOLD
    ):
        flags.append(ReviewFlag.LOW_CONFIDENCE_CLASSIFICATION)
    if any(r.match_status == MatchStatus.AMBIGUOUS for r in line_item_reviews):
        flags.append(ReviewFlag.AMBIGUOUS_REFERENCE)
    if any(r.match_status == MatchStatus.NOT_FOUND for r in line_item_reviews):
        flags.append(ReviewFlag.CONTAINS_NON_CATALOG_ITEM)
    if any(r.quantity is None for r in line_item_reviews):
        flags.append(ReviewFlag.MISSING_QUANTITY)
    if not extraction.customer.company_name and not extraction.customer.email:
        flags.append(ReviewFlag.MISSING_CUSTOMER_INFO)

    reviewer_notes = "Keine Positionen erkannt" if not line_item_reviews else None

    matched_totals = [
        r.line_total_eur
        for r in line_item_reviews
        if r.match_status == MatchStatus.MATCHED and r.line_total_eur is not None
    ]
    subtotal = sum(matched_totals) if matched_totals else None

    overall = (
        OverallVerdict.NEEDS_HUMAN_REVIEW
        if (not line_item_reviews or flags)
        else OverallVerdict.APPROVE
    )

    overall_confidence = min(
        [extraction.confidence] + [r.confidence for r in line_item_reviews]
    )

    return ReviewVerdict(
        overall_verdict=overall,
        confidence=overall_confidence,
        line_item_reviews=line_item_reviews,
        flags=flags,
        subtotal_eur=subtotal,
        reviewer_notes=reviewer_notes,
    )
