# Gemma 4 phishing data preparation notebook.
# Builds explainable phishing-classification data from MongoDB and pushes it to Hugging Face Datasets.
# Data policy: no RDAP, WHOIS, domain registration, domain reputation, or human-in-the-loop examples.

# %%!
%pip install --upgrade datasets huggingface-hub pymongo python-dotenv beautifulsoup4 lxml tqdm ipywidgets

# %%!
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import HfApi
from tqdm.auto import tqdm

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "gemma4_hf"
MAX_PHISHING = 20_000
MAX_LEGITIMATE = 20_000
MAX_TEXT_CHARS = 6000
SAMPLING = "random"
SEED = 137

HF_TOKEN = os.environ.get("HF_TOKEN") or None
HF_DATASET_REPO_ID = os.environ.get("HF_DATASET_REPO_ID")
HF_DATASET_REVISION = os.environ.get("HF_DATASET_REVISION") or None
HF_DATASET_PRIVATE = os.environ.get("HF_DATASET_PRIVATE", "true").lower() not in {"0", "false", "no"}

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is required in .env to create and push the Hugging Face dataset.")

api = HfApi(token=HF_TOKEN)
if not HF_DATASET_REPO_ID:
    username = api.whoami(token=HF_TOKEN)["name"]
    HF_DATASET_REPO_ID = f"{username}/gemma4-phishing-xai"

print(f"Project root: {PROJECT_ROOT}")
print(f"Local data dir: {DATA_DIR}")
print(f"Hugging Face dataset repo: {HF_DATASET_REPO_ID}")
print(f"Private dataset: {HF_DATASET_PRIVATE}")

# %%!
from scripts import prepare_data

args = prepare_data.parse_args([])
args.env_file = str(PROJECT_ROOT / ".env")
args.output_dir = str(DATA_DIR)
args.max_phishing = MAX_PHISHING
args.max_legitimate = MAX_LEGITIMATE
args.max_text_chars = MAX_TEXT_CHARS
args.sampling = SAMPLING
args.seed = SEED
args.dedupe_key = "url"

summary = prepare_data.prepare_dataset(args)
print(json.dumps(summary, indent=2))

# %%!
def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc=f"Reading {path.name}"):
            if line.strip():
                rows.append(json.loads(line))
    return rows


rows_by_split = {
    "train": read_jsonl(DATA_DIR / "train.jsonl"),
    "validation": read_jsonl(DATA_DIR / "validation.jsonl"),
    "test": read_jsonl(DATA_DIR / "test.jsonl"),
}

for split, rows in rows_by_split.items():
    if not rows:
        raise RuntimeError(f"{split} split is empty")
    for row in rows[:20]:
        roles = [message["role"] for message in row["messages"]]
        assert roles == ["system", "user", "assistant"], roles

evidence_blob = json.dumps(
    [
        {
            "input_features": row.get("input_features"),
            "signals": row.get("signals"),
            "stats": row.get("stats"),
        }
        for rows in rows_by_split.values()
        for row in rows[:500]
    ],
    ensure_ascii=False,
).lower()
banned_evidence_terms = (
    "rdap",
    "whois",
    "registered_domain",
    "domain_age",
    "domain registration",
    "domain reputation",
)
found_terms = [term for term in banned_evidence_terms if term in evidence_blob]
if found_terms:
    raise RuntimeError(f"Banned evidence terms found: {found_terms}")

dataset = DatasetDict({split: Dataset.from_list(rows) for split, rows in rows_by_split.items()})
dataset

# %%!
label_counts = {
    split: {
        label: sum(1 for row in rows if row["label"] == label)
        for label in ("legitimate", "phishing")
    }
    for split, rows in rows_by_split.items()
}
print(json.dumps(label_counts, indent=2))

sample = rows_by_split["train"][0]
print(json.dumps(
    {
        "roles": [message["role"] for message in sample["messages"]],
        "label": sample["label"],
        "url": sample["url"],
        "answer": sample["answer"][:500],
    },
    indent=2,
))

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

This dataset was prepared from MongoDB collections configured in `.env`:
`phishing_db.website_content` for phishing examples and `tranco.websites` for legitimate examples.

Rows contain standard chat messages with `system`, `user`, and `assistant` roles. The assistant message is compact JSON with `label`, `confidence`, and `explanation`.

Data policy:
- No human-in-the-loop example creation.
- No RDAP, WHOIS, domain-registration, domain-age, or domain-reputation evidence.
- Evidence is limited to URL text, captured page text, page-structure counts, and HTTP fetch metadata.

Prepared examples: {summary["total_examples"]}
"""

(DATA_DIR / "README.md").write_text(dataset_card, encoding="utf-8")
print(dataset_card)

# %%!
api.create_repo(
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
    commit_message="Upload Gemma 4 phishing classification dataset",
    max_shard_size="500MB",
)

api.upload_file(
    path_or_fileobj=str(DATA_DIR / "dataset_summary.json"),
    path_in_repo="dataset_summary.json",
    repo_id=HF_DATASET_REPO_ID,
    repo_type="dataset",
    revision=HF_DATASET_REVISION,
    token=HF_TOKEN,
)
api.upload_file(
    path_or_fileobj=str(DATA_DIR / "README.md"),
    path_in_repo="README.md",
    repo_id=HF_DATASET_REPO_ID,
    repo_type="dataset",
    revision=HF_DATASET_REVISION,
    token=HF_TOKEN,
)

print(commit_info)
print(f"https://huggingface.co/datasets/{HF_DATASET_REPO_ID}")
