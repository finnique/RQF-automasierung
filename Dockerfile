# Runs the web demo (web/main.py). Debian's apt packages for WeasyPrint's
# native Pango/Cairo/GdkPixbuf libs are far more reliable here than the
# conda workaround this project needs on Windows -- see generate_pdf.py.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-liberation \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
# --workers 1: the rate-limit/spend-cap state in web/main.py is in-memory
# and process-local -- see its module docstring.
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
