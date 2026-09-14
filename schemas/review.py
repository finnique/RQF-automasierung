"""Schema for the review verdict: the result of matching each extracted
line item against the catalog and deciding whether the resulting offer
is safe to send automatically or needs a human.

This is the second LLM/logic step in the pipeline. It consumes an
ExtractionOutput plus catalog lookup results and produces one verdict
per line item plus an overall verdict for the whole RFQ.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class MatchStatus(str, Enum):
    MATCHED = "matched"  # single, confident catalog match
    AMBIGUOUS = "ambiguous"  # multiple plausible catalog candidates
    NOT_FOUND = "not_found"  # no plausible catalog match (not in our range)


class LineItemReview(BaseModel):
    raw_text: str = Field(..., description="Echoes ExtractedLineItem.raw_text")
    match_status: MatchStatus

    matched_article_number: Optional[str] = Field(
        None, description="Set when match_status == matched"
    )
    matched_article_name: Optional[str] = None
    candidate_article_numbers: list[str] = Field(
        default_factory=list,
        description="Alternative catalog articles worth confirming with the "
        "customer; populated when match_status == ambiguous",
    )

    confidence: float = Field(..., ge=0, le=1)

    quantity: Optional[float] = None
    unit_price_eur: Optional[float] = Field(
        None, description="Catalog unit price; only set when matched"
    )
    line_total_eur: Optional[float] = None

    issue: Optional[str] = Field(
        None,
        description="Human-readable explanation when status is not a clean "
        "match, e.g. 'Kunde nennt nur \"Schütz 25A\", Spulenspannung fehlt'",
    )

    @model_validator(mode="after")
    def matched_requires_article(self) -> "LineItemReview":
        if self.match_status == MatchStatus.MATCHED and not self.matched_article_number:
            raise ValueError("matched line items must set matched_article_number")
        return self


class OverallVerdict(str, Enum):
    APPROVE = "approve"  # all lines matched with high confidence -> auto-send offer
    NEEDS_HUMAN_REVIEW = "needs_human_review"  # ambiguous/missing items or low confidence
    REJECT = "reject"  # not a fulfillable RFQ (e.g. not_rfq, or nothing in catalog)


class ReviewFlag(str, Enum):
    NOT_A_QUOTE_REQUEST = "not_a_quote_request"
    CONTAINS_NON_CATALOG_ITEM = "contains_non_catalog_item"
    AMBIGUOUS_REFERENCE = "ambiguous_reference"
    MISSING_QUANTITY = "missing_quantity"
    MISSING_CUSTOMER_INFO = "missing_customer_info"
    LOW_CONFIDENCE_CLASSIFICATION = "low_confidence_classification"


class ReviewVerdict(BaseModel):
    overall_verdict: OverallVerdict
    confidence: float = Field(..., ge=0, le=1)

    line_item_reviews: list[LineItemReview] = Field(default_factory=list)
    flags: list[ReviewFlag] = Field(default_factory=list)

    subtotal_eur: Optional[float] = Field(
        None, description="Sum of line_total_eur across matched lines only"
    )

    reviewer_notes: Optional[str] = Field(
        None, description="Free-text summary for the human reviewer, if any flag is set"
    )

    @model_validator(mode="after")
    def reject_or_review_needs_reason(self) -> "ReviewVerdict":
        if self.overall_verdict != OverallVerdict.APPROVE and not (
            self.flags or self.reviewer_notes
        ):
            raise ValueError(
                "non-approve verdicts must set at least one flag or reviewer_notes"
            )
        return self
