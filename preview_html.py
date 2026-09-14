"""Render the Angebot template to a plain .html file for quick viewing in a
browser -- no WeasyPrint/conda env needed, just Jinja2. Note: the print-only
@page rules (page size, margins, the repeating running footer) are ignored
by browsers on screen, so the footer will render once at the end of the
flow instead of pinned per page -- open the actual PDF (generate_pdf.py) to
see pagination behavior."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"


def format_de_currency(value: float) -> str:
    s = f"{value:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} €"


def format_de_number(value) -> str:
    if float(value) == int(value):
        s = f"{int(value):,}"
    else:
        s = f"{value:,.2f}".rstrip("0").rstrip(",")
    return s.replace(",", ".")


def render_offer_html(data: dict) -> str:
    """Render an already-built offer dict (matching the shape of
    data/sample_offer.json -- see offer.py for the real, non-hardcoded
    producer) through the Jinja2 template to an HTML string."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    env.filters["de_currency"] = format_de_currency
    env.filters["de_number"] = format_de_number
    return env.get_template("angebot.html.jinja").render(**data)


if __name__ == "__main__":
    data = json.loads((DATA_DIR / "sample_offer.json").read_text(encoding="utf-8"))
    html_content = render_offer_html(data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "angebot_preview.html"
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Wrote {out_path}")
