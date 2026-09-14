"""Run extract.extract() against every email in samples/emails/ and score
it against the hand-labeled ground truth in data/eval.jsonl.

Usage:
    EXTRACT_MODEL=gpt-5-mini python scripts/eval_extract.py
    EXTRACT_MODEL=gpt-5-mini python scripts/eval_extract.py --limit 3
    EXTRACT_MODEL=gpt-5-mini python scripts/eval_extract.py --email email_13_ambiguous.txt

This makes real, billed API calls -- one per email (18 by default).
Results (per-email breakdown + aggregate totals) are written to
results/eval_<timestamp>_<model>.json so a run doesn't need repeating to
regenerate the numbers; a summary is also printed to stdout.

Scoring:
    classification    exact match against ground truth's "classification"
    customer_company  normalized (whitespace-collapsed, casefolded) exact
                       match; both-null counts as a match
    line_items        each item reduced to (matched_article_number,
                       quantity) and compared as a multiset (Counter)
                       against ground truth, aggregated into micro
                       precision/recall/F1 across the whole run. This
                       also scores "correctly left unresolved" for free,
                       since unresolved ground-truth items simply have
                       matched_article_number: null -- a predicted item
                       with a different quantity or a false positive
                       resolution won't match.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BASE_DIR))

import extract  # noqa: E402  (path must be set up first)

EMAILS_DIR = BASE_DIR / "samples" / "emails"
EVAL_PATH = BASE_DIR / "data" / "eval.jsonl"
RESULTS_DIR = BASE_DIR / "results"


def load_ground_truth() -> list[dict]:
    lines = EVAL_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def normalize_company(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = " ".join(name.split()).casefold()
    return normalized or None


def line_item_key(article_number: str | None, quantity: float | None) -> tuple:
    return (article_number, quantity)


def score_line_items(expected: list[dict], predicted: list[dict]) -> dict:
    exp_counter = Counter(
        line_item_key(i.get("matched_article_number"), i.get("quantity")) for i in expected
    )
    pred_counter = Counter(
        line_item_key(i["matched_article_number"], i["quantity"]) for i in predicted
    )
    tp_counter = exp_counter & pred_counter
    tp = sum(tp_counter.values())
    fp = sum(pred_counter.values()) - tp
    fn = sum(exp_counter.values()) - tp
    return {"tp": tp, "fp": fp, "fn": fn}


def run_one(entry: dict) -> dict:
    file_name = entry["file"]
    email_path = EMAILS_DIR / file_name
    email_text = email_path.read_text(encoding="utf-8")

    result = {
        "file": file_name,
        "success": False,
        "error_type": None,
        "error": None,
        "classification_expected": entry["classification"],
        "classification_predicted": None,
        "classification_correct": None,
        "customer_company_expected": entry.get("customer_company_name"),
        "customer_company_predicted": None,
        "customer_company_correct": None,
        "line_items_expected_count": len(entry["line_items"]),
        "line_items_predicted_count": None,
        "line_items": None,
        "cost_usd": None,
        "latency_seconds": None,
        "tokens_in": None,
        "tokens_out": None,
    }

    try:
        run = extract.extract(email_text)
    except extract.ExtractionError as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        return result
    except Exception as exc:  # noqa: BLE001 -- surface unexpected errors in the report too
        result["error_type"] = f"Unexpected:{type(exc).__name__}"
        result["error"] = str(exc)
        return result

    predicted = run.extraction

    result["success"] = True
    result["classification_predicted"] = predicted.classification.value
    result["classification_correct"] = (
        predicted.classification.value == entry["classification"]
    )

    predicted_company = normalize_company(
        predicted.customer.company_name if predicted.customer else None
    )
    expected_company = normalize_company(entry.get("customer_company_name"))
    result["customer_company_predicted"] = predicted.customer.company_name if predicted.customer else None
    result["customer_company_correct"] = predicted_company == expected_company

    predicted_items = [
        {"matched_article_number": li.matched_article_number, "quantity": li.quantity}
        for li in predicted.line_items
    ]
    result["line_items_predicted_count"] = len(predicted_items)
    result["line_items"] = score_line_items(entry["line_items"], predicted_items)

    result["cost_usd"] = run.cost_usd
    result["latency_seconds"] = run.latency_seconds
    result["tokens_in"] = run.usage.input_tokens
    result["tokens_out"] = run.usage.output_tokens

    return result


def summarize(model: str, per_email: list[dict]) -> dict:
    succeeded = [r for r in per_email if r["success"]]
    failed = [r for r in per_email if not r["success"]]

    total_tp = sum(r["line_items"]["tp"] for r in succeeded)
    total_fp = sum(r["line_items"]["fp"] for r in succeeded)
    total_fn = sum(r["line_items"]["fn"] for r in succeeded)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else None
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    latencies = [r["latency_seconds"] for r in succeeded]
    costs = [r["cost_usd"] for r in succeeded]

    return {
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_emails": len(per_email),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "failed_files": [r["file"] for r in failed],
        "classification_accuracy": (
            sum(r["classification_correct"] for r in succeeded) / len(succeeded)
            if succeeded else None
        ),
        "customer_company_accuracy": (
            sum(r["customer_company_correct"] for r in succeeded) / len(succeeded)
            if succeeded else None
        ),
        "line_items_precision": precision,
        "line_items_recall": recall,
        "line_items_f1": f1,
        "line_items_tp": total_tp,
        "line_items_fp": total_fp,
        "line_items_fn": total_fn,
        "total_cost_usd": sum(costs) if costs else 0.0,
        "avg_latency_seconds": statistics.mean(latencies) if latencies else None,
        "total_latency_seconds": sum(latencies) if latencies else 0.0,
    }


def print_summary(summary: dict) -> None:
    def fmt_pct(x):
        return f"{x * 100:.1f}%" if x is not None else "n/a"

    def fmt_money(x):
        return f"${x:.4f}" if x is not None else "n/a"

    def fmt_seconds(x):
        return f"{x:.2f}s" if x is not None else "n/a"

    print(f"\nmodel: {summary['model']}")
    print(f"emails: {summary['succeeded']}/{summary['total_emails']} succeeded"
          + (f"  (failed: {', '.join(summary['failed_files'])})" if summary["failed_files"] else ""))
    print(f"classification accuracy:   {fmt_pct(summary['classification_accuracy'])}")
    print(f"customer company accuracy: {fmt_pct(summary['customer_company_accuracy'])}")
    print(f"line items  precision: {fmt_pct(summary['line_items_precision'])}"
          f"  recall: {fmt_pct(summary['line_items_recall'])}"
          f"  f1: {fmt_pct(summary['line_items_f1'])}"
          f"  (tp={summary['line_items_tp']} fp={summary['line_items_fp']} fn={summary['line_items_fn']})")
    print(f"total cost: {fmt_money(summary['total_cost_usd'])}"
          f"   avg latency: {fmt_seconds(summary['avg_latency_seconds'])}"
          f"   total latency: {fmt_seconds(summary['total_latency_seconds'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N emails")
    parser.add_argument("--email", type=str, default=None, help="only run this one email file")
    args = parser.parse_args()

    ground_truth = load_ground_truth()
    if args.email:
        ground_truth = [e for e in ground_truth if e["file"] == args.email]
        if not ground_truth:
            raise SystemExit(f"No ground truth entry for {args.email!r}")
    if args.limit:
        ground_truth = ground_truth[: args.limit]

    per_email = []
    for entry in ground_truth:
        print(f"running {entry['file']} ...", end=" ", flush=True)
        result = run_one(entry)
        status = "ok" if result["success"] else f"FAILED ({result['error_type']})"
        print(status)
        per_email.append(result)

    model_name = os.environ.get("EXTRACT_MODEL", "unknown-model")

    summary = summarize(model_name, per_email)
    print_summary(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"eval_{ts}_{model_name}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "per_email": per_email}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
