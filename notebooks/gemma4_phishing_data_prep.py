# Standalone Colab notebook: prepare explainable phishing data from MongoDB and upload it to Hugging Face.
# Required .env values: MONGO_URI and HF_TOKEN.
# Optional .env values: HF_DATASET_REPO_ID, HF_DATASET_PRIVATE, HF_DATASET_REVISION.

# %%!
%pip install -q -U datasets huggingface-hub pymongo python-dotenv beautifulsoup4 lxml tqdm ipywidgets

# %%!
from __future__ import annotations

import hashlib
import html as html_lib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import HfApi
from pymongo import MongoClient
from pymongo.collection import Collection
from tqdm.auto import tqdm

IS_COLAB = "google.colab" in __import__("sys").modules
ENV_FILE = Path("/content/.env" if IS_COLAB else ".env")
if not ENV_FILE.exists() and IS_COLAB:
    from google.colab import files

    print("Upload the project's .env file.")
    uploaded = files.upload()
    if ".env" not in uploaded:
        raise RuntimeError("The uploaded file must be named .env.")
    ENV_FILE.write_bytes(uploaded[".env"])
if not ENV_FILE.exists():
    raise FileNotFoundError(f"Required environment file was not found: {ENV_FILE.resolve()}")
load_dotenv(ENV_FILE, override=True)

DATA_DIR = Path("/content/gemma4_phishing_data" if IS_COLAB else "gemma4_phishing_data")

PHISHING_DB = "phishing_db"
PHISHING_COLLECTION = "website_content"
TRANCO_DB = "tranco"
TRANCO_COLLECTION = "websites"

MAX_PHISHING = 20_000
MAX_LEGITIMATE = 20_000
MAX_TEXT_CHARS = 6_000
MIN_HTML_CHARS = 80
SAMPLE_MULTIPLIER = 5
SAMPLING = "random"
SEED = 137
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1

MONGO_URI = os.environ.get("MONGO_URI")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET_REPO_ID = os.environ.get("HF_DATASET_REPO_ID")
HF_DATASET_REVISION = os.environ.get("HF_DATASET_REVISION") or None
HF_DATASET_PRIVATE = os.environ.get("HF_DATASET_PRIVATE", "true").lower() not in {"0", "false", "no"}
if not MONGO_URI:
    raise RuntimeError(f"MONGO_URI is required in {ENV_FILE}.")
if not HF_TOKEN:
    raise RuntimeError(f"HF_TOKEN is required in {ENV_FILE}.")

hf_api = HfApi(token=HF_TOKEN)
if not HF_DATASET_REPO_ID:
    hf_username = hf_api.whoami(token=HF_TOKEN)["name"]
    HF_DATASET_REPO_ID = f"{hf_username}/gemma4-phishing-xai"

DATA_DIR.mkdir(parents=True, exist_ok=True)
print(f"Local output: {DATA_DIR.resolve()}")
print(f"Hugging Face dataset: https://huggingface.co/datasets/{HF_DATASET_REPO_ID}")
print(f"Private dataset: {HF_DATASET_PRIVATE}")

# %%!
SYSTEM_PROMPT = (
    "You are a security classifier for website phishing detection. "
    "Classify websites using only the supplied URL, captured page text, page-structure counts, "
    "and HTTP fetch metadata. Return compact JSON with keys label, confidence, and explanation. "
    "The label must be exactly one of: phishing, legitimate. "
    "Do not use domain registration, WHOIS/RDAP, reputation feeds, or human review."
)

SUSPICIOUS_URL_TERMS = {
    "account", "auth", "bank", "billing", "confirm", "credential", "login",
    "password", "pay", "payment", "secure", "signin", "update", "verify",
    "wallet", "webscr",
}
SOCIAL_ENGINEERING_TERMS = {
    "account suspended", "act now", "billing information", "confirm your",
    "credit card", "limited time", "login", "password", "payment failed",
    "security alert", "sign in", "unusual activity", "verify", "wallet",
}
BANNED_FIELD_FRAGMENTS = (
    "rdap", "whois", "registered_domain", "domain_age",
    "domain_registration", "domain_reputation", "registrar", "registrant",
    "creation_date", "expiration_date",
)
EXPECTED_SIGNAL_FIELDS = {"url", "content"}
EXPECTED_STATS_FIELDS = {
    "text_chars", "links_or_form_targets", "script_link_iframe_resources",
    "forms", "password_fields", "input_fields", "iframes", "scripts",
    "status_code", "redirect_count",
}


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        parsed = urlparse(f"https://{url.strip()}")
    path = parsed.path or "/"
    return (
        f"{parsed.scheme.lower()}://{parsed.netloc.lower().strip()}{path}"
        + (f"?{parsed.query}" if parsed.query else "")
    )


def compact_text(text: str, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", html_lib.unescape(text or "")).strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 20].rstrip() + " ... [truncated]"


def clean_html_text(raw_html: str, max_chars: int) -> str:
    soup = BeautifulSoup(raw_html or "", "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "canvas"]):
        tag.decompose()
    return compact_text(soup.get_text(" "), max_chars)


def count_links_and_resources(soup: BeautifulSoup) -> tuple[int, int]:
    link_count = 0
    resource_count = 0
    for tag, attr in (("a", "href"), ("form", "action")):
        for node in soup.find_all(tag):
            raw = node.get(attr)
            if isinstance(raw, str) and raw.strip() and not raw.strip().startswith(("#", "javascript:")):
                link_count += 1
    for tag, attr in (("script", "src"), ("link", "href"), ("iframe", "src")):
        for node in soup.find_all(tag):
            raw = node.get(attr)
            if isinstance(raw, str) and raw.strip() and not raw.strip().startswith(("#", "javascript:")):
                resource_count += 1
    return link_count, resource_count


def url_signals(url: str) -> list[str]:
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    path_query = f"{parsed.path}?{parsed.query}".lower()
    signals: list[str] = []
    if parsed.scheme != "https":
        signals.append("URL does not use HTTPS.")
    if len(normalized) >= 90:
        signals.append(f"URL is long ({len(normalized)} characters).")
    matched = sorted(term for term in SUSPICIOUS_URL_TERMS if term in path_query)
    if matched:
        signals.append("URL path or query contains sensitive-action terms: " + ", ".join(matched[:6]) + ".")
    if "@" in normalized:
        signals.append("URL contains an @ character, which can obscure the visible destination.")
    if normalized.count("-") >= 4:
        signals.append("URL contains many hyphen separators.")
    return signals or ["URL does not show strong lexical phishing indicators."]


def page_signals(
    soup: BeautifulSoup,
    text: str,
    link_count: int,
    resource_count: int,
    status_code: int | None,
    redirect_count: int,
) -> list[str]:
    forms = soup.find_all("form")
    password_fields = soup.find_all("input", {"type": re.compile("^password$", re.I)})
    signals: list[str] = []
    if password_fields:
        signals.append(f"Page contains {len(password_fields)} password input field(s).")
    elif forms:
        signals.append(f"Page contains {len(forms)} form(s) but no password field.")
    else:
        signals.append("No HTML forms were detected in the captured page.")
    matched = sorted(term for term in SOCIAL_ENGINEERING_TERMS if term in text.lower())
    if matched:
        signals.append("Page text contains social-engineering terms: " + ", ".join(matched[:6]) + ".")
    signals.append(
        f"Page contains {link_count} link/form target(s) and "
        f"{resource_count} script/link/iframe resource reference(s)."
    )
    if len(text) < 120:
        signals.append("Captured body text is very short, which limits confidence.")
    if redirect_count >= 3:
        signals.append(f"Request followed {redirect_count} redirects.")
    if status_code is not None:
        signals.append(f"HTTP status code was {status_code}.")
    return signals


def as_int(value: Any, default: int | None = 0) -> int | None:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def build_example(doc: dict[str, Any], label: str, source: str) -> dict[str, Any] | None:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    url = str(doc.get("url") or metadata.get("url") or "").strip()
    if not url:
        return None
    raw_html = str(doc.get("html") or "")
    soup = BeautifulSoup(raw_html, "lxml")
    normalized_url = normalize_url(url)
    final_url = str(metadata.get("final_url") or metadata.get("url") or url)
    soup_title = soup.title.string if soup.title and soup.title.string else ""
    title = compact_text(str(doc.get("title") or soup_title or ""), 180)
    text = clean_html_text(raw_html, MAX_TEXT_CHARS)
    link_count, resource_count = count_links_and_resources(soup)
    status_code = as_int(metadata.get("status_code"), None)
    redirect_count = as_int(metadata.get("redirect_count"), 0) or 0
    url_evidence = url_signals(final_url)
    content_evidence = page_signals(
        soup, text, link_count, resource_count, status_code, redirect_count
    )

    forms_count = len(soup.find_all("form"))
    password_fields = len(soup.find_all("input", {"type": re.compile("^password$", re.I)}))
    input_fields = len(soup.find_all("input"))
    iframe_count = len(soup.find_all("iframe"))
    script_count = len(soup.find_all("script"))

    stats = {
        "text_chars": len(text),
        "links_or_form_targets": link_count,
        "script_link_iframe_resources": resource_count,
        "forms": forms_count,
        "password_fields": password_fields,
        "input_fields": input_fields,
        "iframes": iframe_count,
        "scripts": script_count,
        "status_code": status_code,
        "redirect_count": redirect_count,
    }
    page_stats = "\n".join(f"- {key}: {value}" for key, value in stats.items())
    input_features = "\n".join(
        [
            "# Information:",
            "## URL:", final_url,
            "## Title:", title or "No title captured.",
            "## Content:", text or "No page body text captured.",
            "## Page Structure:", page_stats,
        ]
    )
    explanation_parts = [
        f"Predicted as {label} based on automatically extracted website evidence.",
        "URL signals: " + " ".join(url_evidence[:4]),
        "Website content and link signals: " + " ".join(content_evidence[:5]),
    ]
    if label == "legitimate":
        explanation_parts.append(
            "Legitimate examples come from the Tranco corpus; popularity is not proof of safety."
        )
    explanation = "\n".join(explanation_parts)
    confidence = 0.72
    if label == "phishing" and (
        password_fields or any("sensitive-action" in signal for signal in url_evidence)
    ):
        confidence = 0.84
    elif label == "legitimate" and status_code is not None and 200 <= status_code < 300:
        confidence = 0.78
    answer = json.dumps(
        {"label": label, "confidence": confidence, "explanation": explanation},
        ensure_ascii=False,
    )
    user_prompt = (
        "Classify the following website evidence using only the fields shown below.\n\n"
        f"{input_features}\n# Pred:"
    )
    return {
        "id": stable_id(normalized_url),
        "url": url,
        "normalized_url": normalized_url,
        "final_url": final_url,
        "title": title,
        "label": label,
        "label_id": 1 if label == "phishing" else 0,
        "source": source,
        "input_features": input_features,
        "answer": answer,
        "explanation": explanation,
        "signals": {"url": url_evidence, "content": content_evidence},
        "stats": stats,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer},
        ],
    }

# %%!
def metadata_status_ok(doc: dict[str, Any]) -> bool:
    metadata = doc.get("metadata")
    if not isinstance(metadata, dict):
        return True
    if metadata.get("error"):
        return False
    code = as_int(metadata.get("status_code"), None)
    return code is None or 200 <= code < 400


def iter_documents(collection: Collection, limit: int) -> Iterable[dict[str, Any]]:
    projection = {
        "_id": 0, "url": 1, "title": 1, "html": 1,
        "error": 1, "fetched_at": 1,
        "metadata.url": 1,
        "metadata.final_url": 1,
        "metadata.status_code": 1,
        "metadata.redirect_count": 1,
        "metadata.error": 1,
    }
    query = {"html": {"$type": "string"}}
    if SAMPLING == "random":
        estimated = collection.estimated_document_count()
        sample_size = min(estimated, max(limit * SAMPLE_MULTIPLIER, limit + 100))
        pipeline = [
            {"$match": query},
            {"$sample": {"size": sample_size}},
            {"$project": projection},
        ]
        yield from collection.aggregate(pipeline, allowDiskUse=True)
    else:
        yield from collection.find(query, projection, no_cursor_timeout=True)


def collect_examples(
    collection: Collection,
    label: str,
    source: str,
    limit: int,
    seen_urls: set[str],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for doc in tqdm(iter_documents(collection, limit), desc=f"Collecting {label}", unit="doc"):
        if len(examples) >= limit:
            break
        html = doc.get("html")
        if (
            doc.get("error")
            or not isinstance(html, str)
            or len(html.strip()) < MIN_HTML_CHARS
            or not metadata_status_ok(doc)
        ):
            continue
        example = build_example(doc, label, source)
        if not example or example["normalized_url"] in seen_urls:
            continue
        seen_urls.add(example["normalized_url"])
        examples.append(example)
    return examples


client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=15_000)
client.admin.command("ping")
seen_urls: set[str] = set()

phishing_examples = collect_examples(
    client[PHISHING_DB][PHISHING_COLLECTION],
    "phishing",
    f"{PHISHING_DB}.{PHISHING_COLLECTION}",
    MAX_PHISHING,
    seen_urls,
)
legitimate_examples = collect_examples(
    client[TRANCO_DB][TRANCO_COLLECTION],
    "legitimate",
    f"{TRANCO_DB}.{TRANCO_COLLECTION}",
    MAX_LEGITIMATE,
    seen_urls,
)
client.close()

examples = phishing_examples + legitimate_examples
if not phishing_examples or not legitimate_examples:
    raise RuntimeError("Both phishing and legitimate examples are required.")

print(f"Prepared {len(phishing_examples)} phishing and {len(legitimate_examples)} legitimate rows.")

# %%!
def split_examples(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(SEED)
    splits = {"train": [], "validation": [], "test": []}
    for label in ("phishing", "legitimate"):
        label_rows = [row for row in rows if row["label"] == label]
        rng.shuffle(label_rows)
        count = len(label_rows)
        train_end = int(count * TRAIN_RATIO)
        validation_end = train_end + int(count * VALIDATION_RATIO)
        if count >= 3:
            train_end = max(1, min(train_end, count - 2))
            validation_end = max(train_end + 1, min(validation_end, count - 1))
        splits["train"].extend(label_rows[:train_end])
        splits["validation"].extend(label_rows[train_end:validation_end])
        splits["test"].extend(label_rows[validation_end:])
    for rows_for_split in splits.values():
        rng.shuffle(rows_for_split)
    return splits


rows_by_split = split_examples(examples)


def find_banned_field_paths(value: Any, path: str = "") -> list[str]:
    matches: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            field_path = f"{path}.{key}" if path else str(key)
            if any(fragment in normalized_key for fragment in BANNED_FIELD_FRAGMENTS):
                matches.append(field_path)
            matches.extend(find_banned_field_paths(nested, field_path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            matches.extend(find_banned_field_paths(nested, f"{path}[{index}]"))
    return matches


for split, rows in rows_by_split.items():
    if not rows:
        raise RuntimeError(f"{split} split is empty.")
    for row in rows[:20]:
        assert [message["role"] for message in row["messages"]] == [
            "system", "user", "assistant"
        ]
        if set(row["signals"]) != EXPECTED_SIGNAL_FIELDS:
            raise RuntimeError(f"Unexpected signal fields in {split}: {sorted(row['signals'])}")
        if set(row["stats"]) != EXPECTED_STATS_FIELDS:
            raise RuntimeError(f"Unexpected stats fields in {split}: {sorted(row['stats'])}")
        banned_paths = find_banned_field_paths(row)
        if banned_paths:
            raise RuntimeError(f"Banned metadata fields found in {split}: {banned_paths}")

dataset = DatasetDict(
    {split: Dataset.from_list(rows) for split, rows in rows_by_split.items()}
)
dataset

# %%!
def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


for split, rows in rows_by_split.items():
    write_jsonl(DATA_DIR / f"{split}.jsonl", rows)

summary = {
    "total_examples": len(examples),
    "split_counts": {split: len(rows) for split, rows in rows_by_split.items()},
    "label_counts": {
        split: dict(Counter(row["label"] for row in rows))
        for split, rows in rows_by_split.items()
    },
    "configuration": {
        "phishing_source": f"{PHISHING_DB}.{PHISHING_COLLECTION}",
        "legitimate_source": f"{TRANCO_DB}.{TRANCO_COLLECTION}",
        "max_phishing": MAX_PHISHING,
        "max_legitimate": MAX_LEGITIMATE,
        "max_text_chars": MAX_TEXT_CHARS,
        "sampling": SAMPLING,
        "seed": SEED,
    },
    "notes": [
        "Labels are assigned automatically from source collections.",
        "Explanations are deterministic evidence summaries, not manually written examples.",
        "No RDAP, WHOIS, domain registration, domain age, or reputation data is used.",
    ],
}
(DATA_DIR / "dataset_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))

# %%!
dataset_card = f"""---
license: other
task_categories:
- text-classification
- text-generation
language:
- en
tags:
- phishing
- cybersecurity
- gemma-4
- explainable-ai
pretty_name: Gemma 4 Explainable Phishing Classification Data
---

# Gemma 4 Explainable Phishing Classification Data

Automatically prepared from `{PHISHING_DB}.{PHISHING_COLLECTION}` and
`{TRANCO_DB}.{TRANCO_COLLECTION}`.

Rows use standard `system`, `user`, and `assistant` chat roles. The assistant
target is compact JSON containing `label`, `confidence`, and `explanation`.

Data policy:
- No human-in-the-loop example creation.
- No RDAP, WHOIS, domain-registration, domain-age, or reputation evidence.
- Model evidence is limited to URL text, captured page text, page-structure
  counts, and HTTP fetch metadata.

Prepared examples: {summary["total_examples"]}
"""
(DATA_DIR / "README.md").write_text(dataset_card, encoding="utf-8")

hf_api.create_repo(
    repo_id=HF_DATASET_REPO_ID,
    repo_type="dataset",
    private=HF_DATASET_PRIVATE,
    exist_ok=True,
    token=HF_TOKEN,
)
commit_info = dataset.push_to_hub(
    HF_DATASET_REPO_ID,
    private=HF_DATASET_PRIVATE,
    token=HF_TOKEN,
    revision=HF_DATASET_REVISION,
    commit_message="Upload standalone Gemma 4 phishing dataset",
    max_shard_size="500MB",
)
for filename in ("dataset_summary.json", "README.md"):
    hf_api.upload_file(
        path_or_fileobj=str(DATA_DIR / filename),
        path_in_repo=filename,
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
        revision=HF_DATASET_REVISION,
        token=HF_TOKEN,
    )

print(commit_info)
print(f"Dataset uploaded: https://huggingface.co/datasets/{HF_DATASET_REPO_ID}")
