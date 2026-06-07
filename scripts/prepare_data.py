#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phish_xai.features import (  # noqa: E402
    automatic_explanation,
    extract_website_features,
    format_answer,
    format_input_features,
    format_messages,
    format_prompt,
    format_user_message,
    stable_id,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare explainable phishing-classification JSONL data from MongoDB."
    )
    parser.add_argument("--env-file", default=".env", help="Path to .env containing MONGO_URI.")
    parser.add_argument("--output-dir", default="data/processed", help="Output dataset directory.")
    parser.add_argument("--phishing-db", default="phishing_db")
    parser.add_argument("--phishing-content-collection", default="website_content")
    parser.add_argument("--tranco-db", default="tranco")
    parser.add_argument("--tranco-collection", default="websites")
    parser.add_argument("--max-phishing", type=int, default=2000)
    parser.add_argument("--max-legitimate", type=int, default=2000)
    parser.add_argument("--min-text-chars", type=int, default=80)
    parser.add_argument("--max-text-chars", type=int, default=3500)
    parser.add_argument(
        "--sample-multiplier",
        type=int,
        default=25,
        help="Mongo-side random sample size multiplier relative to each class limit when --sampling random.",
    )
    parser.add_argument(
        "--sampling",
        choices=["sequential", "random"],
        default="sequential",
        help="Sequential streaming is faster; random uses Mongo $sample and is better for larger dataset builds.",
    )
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--dedupe-key",
        choices=["url", "none"],
        default="url",
        help="Deduplicate separately per label using this key.",
    )
    parser.add_argument(
        "--allow-short-text",
        action="store_true",
        help="Keep pages below --min-text-chars. Useful for smoke tests only.",
    )
    return parser.parse_args(argv)


def get_client(env_file: str) -> MongoClient:
    load_dotenv(env_file)
    uri = os.environ.get("MONGO_URI")
    if not uri:
        raise RuntimeError(f"MONGO_URI was not found in {env_file}")
    return MongoClient(uri, serverSelectionTimeoutMS=15000)


def metadata_status_ok(doc: dict[str, Any]) -> bool:
    metadata = doc.get("metadata")
    if not isinstance(metadata, dict):
        return True
    if metadata.get("error"):
        return False
    status_code = metadata.get("status_code")
    if status_code is None:
        return True
    try:
        code = int(status_code)
    except (TypeError, ValueError):
        return False
    return 200 <= code < 400


def usable_doc(doc: dict[str, Any], min_text_chars: int, allow_short_text: bool) -> bool:
    if doc.get("error"):
        return False
    html = doc.get("html")
    if not isinstance(html, str) or not html.strip():
        return False
    if not metadata_status_ok(doc):
        return False
    if allow_short_text:
        return True
    return len(html.strip()) >= min_text_chars


def dedupe_value(example: dict[str, Any], mode: str) -> str:
    if mode == "none":
        return stable_id(example["url"])
    return example["normalized_url"]


def build_example(
    doc: dict[str, Any],
    label: str,
    source: str,
    max_text_chars: int,
) -> dict[str, Any] | None:
    features = extract_website_features(
        doc,
        max_text_chars=max_text_chars,
    )
    if not features.url:
        return None
    input_features = format_input_features(features)
    explanation = automatic_explanation(features, label)
    confidence = 0.72
    if label == "phishing" and (features.password_fields or any("sensitive-action" in s for s in features.url_signals)):
        confidence = 0.84
    if label == "legitimate" and features.status_code and 200 <= features.status_code < 300:
        confidence = 0.78

    answer = format_answer(label, explanation, confidence=confidence)
    prompt = format_prompt(input_features)
    user_prompt = format_user_message(input_features)
    return {
        "id": features.record_id,
        "url": features.url,
        "normalized_url": features.normalized_url,
        "final_url": features.final_url,
        "title": features.title,
        "label": label,
        "label_id": 1 if label == "phishing" else 0,
        "source": source,
        "input_features": input_features,
        "prompt": prompt,
        "user_prompt": user_prompt,
        "answer": answer,
        "explanation": explanation,
        "signals": {
            "url": features.url_signals,
            "content": features.content_signals,
        },
        "stats": {
            "text_chars": len(features.text),
            "links_or_form_targets": features.link_count,
            "script_link_iframe_resources": features.external_resource_count,
            "forms": features.forms_count,
            "password_fields": features.password_fields,
            "input_fields": features.input_fields,
            "iframes": features.iframe_count,
            "scripts": features.script_count,
            "status_code": features.status_code,
            "redirect_count": features.redirect_count,
        },
        "messages": format_messages(input_features, answer=answer),
    }


def iter_content_collection(col: Collection, sample_size: int = 0) -> Iterable[dict[str, Any]]:
    projection = {
        "_id": 0,
        "url": 1,
        "title": 1,
        "html": 1,
        "error": 1,
        "metadata": 1,
        "fetched_at": 1,
    }
    query = {"html": {"$type": "string"}}
    if sample_size > 0:
        pipeline = [
            {"$match": query},
            {"$sample": {"size": sample_size}},
            {"$project": projection},
        ]
        yield from col.aggregate(pipeline, allowDiskUse=True)
        return
    yield from col.find(query, projection, no_cursor_timeout=True)


def collect_examples(
    content_col: Collection,
    label: str,
    source: str,
    limit: int,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()

    try:
        estimated = content_col.estimated_document_count()
    except Exception:
        estimated = 0
    sample_size = 0
    if args.sampling == "random" and limit > 0 and estimated > 0:
        sample_size = min(estimated, max(limit * args.sample_multiplier, limit + 100))
    cursor = iter_content_collection(content_col, sample_size=sample_size)
    progress = tqdm(cursor, desc=f"Collecting {label}", unit="doc", total=sample_size or None)
    for doc in progress:
        if len(examples) >= limit:
            break
        if not usable_doc(doc, args.min_text_chars, args.allow_short_text):
            continue
        example = build_example(
            doc,
            label=label,
            source=source,
            max_text_chars=args.max_text_chars,
        )
        if not example:
            continue
        key = f"{label}:{dedupe_value(example, args.dedupe_key)}"
        if key in seen:
            continue
        seen.add(key)
        examples.append(example)
        progress.set_postfix_str(f"kept={len(examples)}")
    progress.close()
    return examples


def split_examples(
    examples: list[dict[str, Any]],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    total = train_ratio + validation_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive number")
    train_ratio, validation_ratio, test_ratio = (
        train_ratio / total,
        validation_ratio / total,
        test_ratio / total,
    )
    by_label: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        by_label.setdefault(example["label"], []).append(example)

    splits = {"train": [], "validation": [], "test": []}
    for label_examples in by_label.values():
        rng.shuffle(label_examples)
        n = len(label_examples)
        train_end = int(n * train_ratio)
        validation_end = train_end + int(n * validation_ratio)
        if n >= 3:
            train_end = max(1, min(train_end, n - 2))
            validation_end = max(train_end + 1, min(validation_end, n - 1))
        splits["train"].extend(label_examples[:train_end])
        splits["validation"].extend(label_examples[train_end:validation_end])
        splits["test"].extend(label_examples[validation_end:])

    for split, rows in splits.items():
        rng.shuffle(rows)
        for row in rows:
            row["split"] = split
    return splits


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    split_counts = {split: len(rows) for split, rows in splits.items()}
    label_counts = {
        split: dict(Counter(row["label"] for row in rows))
        for split, rows in splits.items()
    }
    stats = {}
    for split, rows in splits.items():
        text_lengths = [row["stats"]["text_chars"] for row in rows]
        stats[split] = {
            "min_text_chars": min(text_lengths) if text_lengths else 0,
            "max_text_chars": max(text_lengths) if text_lengths else 0,
            "avg_text_chars": round(sum(text_lengths) / len(text_lengths), 2) if text_lengths else 0,
        }
    return {
        "split_counts": split_counts,
        "label_counts": label_counts,
        "text_stats": stats,
    }


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = get_client(args.env_file)
    client.admin.command("ping")
    phishing_db = client[args.phishing_db]
    tranco_db = client[args.tranco_db]

    phishing_examples = collect_examples(
        phishing_db[args.phishing_content_collection],
        label="phishing",
        source=f"{args.phishing_db}.{args.phishing_content_collection}",
        limit=args.max_phishing,
        args=args,
        rng=rng,
    )
    legitimate_examples = collect_examples(
        tranco_db[args.tranco_collection],
        label="legitimate",
        source=f"{args.tranco_db}.{args.tranco_collection}",
        limit=args.max_legitimate,
        args=args,
        rng=rng,
    )

    examples = phishing_examples + legitimate_examples
    if not examples:
        raise RuntimeError("No examples were prepared. Relax filters or inspect Mongo collections.")
    if len({row["label"] for row in examples}) < 2:
        raise RuntimeError("Prepared data contains only one label; both phishing and legitimate are required.")

    splits = split_examples(
        examples,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        rng=rng,
    )

    for split, rows in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    summary = summarize(splits)
    summary.update(
        {
            "args": vars(args),
            "total_examples": len(examples),
            "notes": [
                "Labels are assigned automatically from source collections: phishing_db for phishing and tranco for legitimate.",
                "Explanations are deterministic evidence summaries, not manually curated examples and not generated by another LLM.",
                "Only URL text, rendered page text, page-structure counts, and HTTP fetch metadata are used.",
                "Rows include standard Gemma chat roles: system, user, assistant. The assistant answer is final visible JSON only.",
            ],
        }
    )
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = prepare_dataset(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
