"""Schema for the output of the extraction step: turning a raw RFQ
(email text, optionally plus PDF text) into structured data, including a
best-effort catalog match per line item.

This is the first LLM call in the pipeline. It must succeed even on
messy, ambiguous, or non-RFQ input -- classification and confidence
fields exist so downstream steps can branch instead of guessing.

Catalog matching happens here (the model is given the catalog inline and
asked to map informal product references to article numbers), but pricing
and arithmetic do not -- those belong to the later validation/review
stage, which is also where a second opinion on ambiguous matches
(candidate lists, human review) is recorded. See schemas/review.py.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class RequestClassification(str, Enum):
    RFQ = "rfq"  # a genuine request for quote
    NOT_RFQ = "not_rfq"  # e.g. complaint, job application, newsletter reply
    UNCLEAR = "unclear"  # could be an RFQ but too vague/incomplete to tell


class SourceDocumentType(str, Enum):
    EMAIL = "email"
    PDF = "pdf"
    EMAIL_PLUS_PDF = "email+pdf"


class ExtractedLineItem(BaseModel):
    """One product line as the customer wrote it, plus the model's
    best-effort attempt to map it onto a catalog article number."""

    raw_text: str = Field(
        ..., description="Verbatim snippet from the source describing this item"
    )
    quantity: Optional[float] = Field(
        None, description="Requested quantity, if stated"
    )
    unit: Optional[str] = Field(
        None,
        description="Unit as the customer wrote it (e.g. 'Stk', 'Stueck', 'm', 'Meter'); "
        "may not match the catalog unit",
    )
    product_reference: str = Field(
        ..., description="Customer's own name/description for the product"
    )
    article_number_hint: Optional[str] = Field(
        None, description="Article/order number if the customer supplied one directly"
    )

    matched_article_number: Optional[str] = Field(
        None,
        description="Catalog article number this line was mapped to, if the model "
        "was confident enough to resolve it. Must be null when unresolved is true.",
    )
    match_confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Model's confidence in matched_article_number (or, when "
        "unresolved, how strongly it believes no confident match exists)",
    )
    unresolved: bool = Field(
        ...,
        description="True when product_reference could not be confidently mapped "
        "to a single catalog article. Never guess -- set this instead.",
    )

    notes: Optional[str] = Field(
        None,
        description="Extra qualifiers mentioned inline (color, voltage, size, brand) "
        "or, for unresolved/ambiguous items, why -- e.g. candidate article numbers",
    )

    @model_validator(mode="after")
    def matched_xor_unresolved(self) -> "ExtractedLineItem":
        if self.unresolved and self.matched_article_number is not None:
            raise ValueError("unresolved line items must not set matched_article_number")
        if not self.unresolved and self.matched_article_number is None:
            raise ValueError("resolved line items must set matched_article_number")
        return self


class CustomerInfo(BaseModel):
    company_name: Optional[str] = None
    contact_person: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    delivery_address: Optional[str] = None


class ExtractionOutput(BaseModel):
    classification: RequestClassification
    classification_reason: str = Field(
        ..., description="One sentence explaining the classification decision"
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Model's confidence in the classification"
    )
    source_document_type: SourceDocumentType
    language: str = Field(..., description="ISO 639-1 code, e.g. 'de'")

    customer: CustomerInfo = Field(default_factory=CustomerInfo)
    line_items: list[ExtractedLineItem] = Field(default_factory=list)

    requested_delivery_date: Optional[str] = Field(
        None, description="Free-text delivery date/deadline as stated by the customer"
    )
    reference_number: Optional[str] = Field(
        None, description="Customer's own RFQ/project/reference number, if given"
    )
    additional_notes: Optional[str] = Field(
        None,
        description="Anything relevant that doesn't fit elsewhere, e.g. "
        "'Angebot bis Freitag benötigt' or shipping constraints",
    )
