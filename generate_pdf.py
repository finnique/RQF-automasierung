"""Phase 2: render a hardcoded, already-priced offer (data/sample_offer.json)
through the Jinja2 template into a PDF via WeasyPrint. No AI involved --
this only proves the document pipeline works before extraction/pricing
is wired in front of it.

On Windows, WeasyPrint needs native Pango/GObject/Cairo libraries that
plain `pip install weasyprint` does not provide. Use a conda env instead:

    conda create -n quotegen -c conda-forge python=3.11 weasyprint jinja2 -y
    conda run -n quotegen python generate_pdf.py
"""

from __future__ import annotations

import json
from pathlib import Path

from weasyprint import HTML

from preview_html import render_offer_html

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"


def render_offer_pdf_bytes(data: dict) -> bytes:
    """Render an already-built offer dict (see offer.py) straight to PDF
    bytes, with nothing written to disk -- for a caller that returns the
    PDF over HTTP (the web demo) rather than saving it."""
    html_content = render_offer_html(data)
    return HTML(string=html_content, base_url=str(BASE_DIR)).write_pdf()


def render_offer_data(data: dict, output_path: Path) -> None:
    """Render an already-built offer dict (matching the shape of
    data/sample_offer.json -- see offer.py for the real, non-hardcoded
    producer) through the Jinja2 template to a PDF at output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(render_offer_pdf_bytes(data))
    print(f"Wrote {output_path}")


def render_offer(data_path: Path, output_path: Path) -> None:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    render_offer_data(data, output_path)


if __name__ == "__main__":
    offer = json.loads((DATA_DIR / "sample_offer.json").read_text(encoding="utf-8"))
    out_file = OUTPUT_DIR / f"angebot_{offer['angebotsnummer']}.pdf"
    render_offer(DATA_DIR / "sample_offer.json", out_file)
