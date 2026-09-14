"""Schema for a single catalog item and a loader for data/catalog.csv."""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class CatalogItem(BaseModel):
    artikelnummer: str = Field(..., description="Unique article number, e.g. 'SC-3003'")
    bezeichnung: str = Field(..., description="Short product name")
    beschreibung: str = Field(..., description="Longer technical description")
    einheit: str = Field(..., description="Sales unit, e.g. 'Stk', 'm', 'Pkg'")
    einzelpreis_eur: float = Field(..., gt=0, description="Net unit price in EUR")
    ve: int = Field(..., gt=0, description="Verpackungseinheit / pack quantity")

    @field_validator("artikelnummer")
    @classmethod
    def article_number_format(cls, v: str) -> str:
        if "-" not in v:
            raise ValueError("article number must be of the form 'PREFIX-NNNN'")
        return v


def load_catalog(path: str | Path = None) -> list[CatalogItem]:
    """Load and validate the catalog CSV into a list of CatalogItem."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "catalog.csv"
    path = Path(path)
    items: list[CatalogItem] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(
                CatalogItem(
                    artikelnummer=row["artikelnummer"],
                    bezeichnung=row["bezeichnung"],
                    beschreibung=row["beschreibung"],
                    einheit=row["einheit"],
                    einzelpreis_eur=float(row["einzelpreis_eur"]),
                    ve=int(row["ve"]),
                )
            )
    return items


if __name__ == "__main__":
    catalog = load_catalog()
    print(f"Loaded {len(catalog)} catalog items")
    for item in catalog[:3]:
        print(item)
