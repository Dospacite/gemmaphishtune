# Standalone Colab notebook: evaluate the merged Gemma 4 E4B phishing classifier.
# Required .env values: HF_TOKEN.
# Optional .env values: HF_MODEL_REPO_ID, HF_DATASET_REPO_ID, HF_DATASET_REVISION.

# %%!
%pip install -q -U unsloth unsloth_zoo datasets accelerate bitsandbytes huggingface-hub python-dotenv pandas scikit-learn matplotlib seaborn tqdm ipywidgets

# %%!
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from IPython.display import display
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
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

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(f"HF_TOKEN is required in {ENV_FILE}.")

HF_MODEL_REPO_ID = os.environ.get(
    "HF_MODEL_REPO_ID",
    "Dospacite/gemma4-e4b-unsloth-phishing-merged",
)
HF_DATASET_REPO_ID = os.environ.get("HF_DATASET_REPO_ID")
HF_DATASET_REVISION = os.environ.get("HF_DATASET_REVISION") or None
if not HF_DATASET_REPO_ID:
    HF_DATASET_REPO_ID = f"{HfApi(token=HF_TOKEN).whoami(token=HF_TOKEN)['name']}/gemma4-phishing-xai"

EVAL_SPLIT = os.environ.get("EVAL_SPLIT", "test")
MAX_EVAL_SAMPLES = int(os.environ.get("MAX_EVAL_SAMPLES", "0")) or None
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "4096"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
GENERATION_BATCH_SIZE = int(os.environ.get("GENERATION_BATCH_SIZE", "4"))
MIN_VRAM_GB = float(os.environ.get("MIN_VRAM_GB", "80"))
ENFORCE_MIN_VRAM = os.environ.get("ENFORCE_MIN_VRAM", "true").lower() not in {"0", "false", "no"}
EVAL_DIR = Path("/content/gemma4-e4b-eval" if IS_COLAB else "gemma4-e4b-eval")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Environment file: {ENV_FILE.resolve()}")
print(f"Model: https://huggingface.co/{HF_MODEL_REPO_ID}")
print(f"Dataset: https://huggingface.co/datasets/{HF_DATASET_REPO_ID}")
print(f"Split: {EVAL_SPLIT}")
print(f"Output: {EVAL_DIR.resolve()}")

# %%!
dataset_dict = load_dataset(
    HF_DATASET_REPO_ID,
    revision=HF_DATASET_REVISION,
    token=HF_TOKEN,
)
if EVAL_SPLIT not in dataset_dict:
    raise RuntimeError(f"Split {EVAL_SPLIT!r} not found. Available splits: {list(dataset_dict)}")

eval_dataset = dataset_dict[EVAL_SPLIT]
if MAX_EVAL_SAMPLES is not None:
    eval_dataset = eval_dataset.select(range(min(MAX_EVAL_SAMPLES, len(eval_dataset))))
if len(eval_dataset) == 0:
    raise RuntimeError(f"Evaluation split {EVAL_SPLIT!r} is empty.")

for row in eval_dataset.select(range(min(10, len(eval_dataset)))):
    roles = [message["role"] for message in row["messages"]]
    assert roles == ["system", "user", "assistant"], roles

print(eval_dataset)
print("Label counts:", pd.Series(eval_dataset["label"]).value_counts().to_dict())

# %%!
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this evaluation notebook.")

props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / 1024**3
print(f"GPU: {props.name} | VRAM: {vram_gb:.2f} GB")
if ENFORCE_MIN_VRAM and vram_gb < MIN_VRAM_GB:
    raise RuntimeError(f"Detected {vram_gb:.2f} GB VRAM, expected at least {MIN_VRAM_GB:.1f} GB.")

from unsloth import FastModel

model, processor = FastModel.from_pretrained(
    model_name=HF_MODEL_REPO_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=False,
    token=HF_TOKEN,
    disable_log_stats=True,
)
FastModel.for_inference(model)
text_tokenizer = getattr(processor, "tokenizer", processor)
if hasattr(text_tokenizer, "padding_side"):
    text_tokenizer.padding_side = "left"
if getattr(text_tokenizer, "pad_token", None) is None and getattr(text_tokenizer, "eos_token", None) is not None:
    text_tokenizer.pad_token = text_tokenizer.eos_token

model_device = next(model.parameters()).device
pad_token_id = getattr(text_tokenizer, "pad_token_id", None)
eos_token_id = getattr(text_tokenizer, "eos_token_id", None)
print(f"Model device: {model_device}")

# %%!
LABELS = ["legitimate", "phishing"]
LABEL_ALIASES = {
    "legit": "legitimate",
    "benign": "legitimate",
    "safe": "legitimate",
    "clean": "legitimate",
    "not phishing": "legitimate",
    "phish": "phishing",
    "malicious": "phishing",
    "scam": "phishing",
    "suspicious": "phishing",
}


def apply_chat(messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def make_prompt(row: dict[str, Any]) -> str:
    return apply_chat(row["messages"][:-1], add_generation_prompt=True)


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z\s-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text in LABELS:
        return text
    if text in LABEL_ALIASES:
        return LABEL_ALIASES[text]
    for label in LABELS:
        if re.search(rf"\b{re.escape(label)}\b", text):
            return label
    for alias, label in LABEL_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return label
    return None


def coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        match = re.search(r"0?\.\d+|1(?:\.0+)?|\d{1,3}\s*%", str(value))
        if not match:
            return None
        raw = match.group(0).replace("%", "").strip()
        confidence = float(raw)
    if confidence > 1.0:
        confidence /= 100.0
    if math.isnan(confidence):
        return None
    return max(0.0, min(1.0, confidence))


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def parse_prediction(text: str) -> dict[str, Any]:
    parsed = extract_json_object(text)
    parse_ok = parsed is not None
    if parsed is None:
        parsed = {}
    label = normalize_label(parsed.get("label"))
    if label is None:
        label = normalize_label(text)
    confidence = coerce_confidence(parsed.get("confidence"))
    explanation = parsed.get("explanation")
    if explanation is not None:
        explanation = str(explanation)
    return {
        "pred_label": label,
        "pred_confidence": confidence,
        "pred_explanation": explanation,
        "parse_ok": parse_ok,
    }


def batched(rows: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield start, rows[start : start + batch_size]

# %%!
rows = [dict(row) for row in eval_dataset]
predictions: list[dict[str, Any]] = []
start_time = time.time()

for start, batch_rows in tqdm(list(batched(rows, GENERATION_BATCH_SIZE)), desc="Generating"):
    prompts = [make_prompt(row) for row in batch_rows]
    inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(model_device)
    prompt_token_count = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
    generated_ids = output_ids[:, prompt_token_count:]
    decoded = text_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    for offset, (row, prompt, prediction_text) in enumerate(zip(batch_rows, prompts, decoded)):
        parsed = parse_prediction(prediction_text)
        gold_label = normalize_label(row.get("label"))
        pred_label = parsed["pred_label"]
        stats = row.get("stats") or {}
        result = {
            "row_index": start + offset,
            "id": row.get("id"),
            "url": row.get("url"),
            "title": row.get("title"),
            "gold_label": gold_label,
            "pred_label": pred_label,
            "pred_confidence": parsed["pred_confidence"],
            "parse_ok": parsed["parse_ok"],
            "is_correct": bool(gold_label is not None and pred_label == gold_label),
            "prediction_text": prediction_text.strip(),
            "pred_explanation": parsed["pred_explanation"],
            "gold_answer": row.get("answer"),
            "prompt_chars": len(prompt),
            "text_chars": stats.get("text_chars"),
            "forms": stats.get("forms"),
            "password_fields": stats.get("password_fields"),
            "input_fields": stats.get("input_fields"),
            "links_or_form_targets": stats.get("links_or_form_targets"),
            "script_link_iframe_resources": stats.get("script_link_iframe_resources"),
            "iframes": stats.get("iframes"),
            "scripts": stats.get("scripts"),
            "status_code": stats.get("status_code"),
            "redirect_count": stats.get("redirect_count"),
        }
        predictions.append(result)

elapsed_seconds = time.time() - start_time
df = pd.DataFrame(predictions)
df.head()

# %%!
valid_df = df[df["pred_label"].isin(LABELS) & df["gold_label"].isin(LABELS)].copy()
invalid_df = df[~(df["pred_label"].isin(LABELS) & df["gold_label"].isin(LABELS))].copy()
if valid_df.empty:
    raise RuntimeError("No valid parsed predictions were produced.")

y_true = valid_df["gold_label"].tolist()
y_pred = valid_df["pred_label"].tolist()
cm = confusion_matrix(y_true, y_pred, labels=LABELS)
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=LABELS,
    zero_division=0,
)
per_class = {
    label: {
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
        "support": int(support[index]),
    }
    for index, label in enumerate(LABELS)
}
metrics = {
    "model_repo_id": HF_MODEL_REPO_ID,
    "dataset_repo_id": HF_DATASET_REPO_ID,
    "dataset_revision": HF_DATASET_REVISION,
    "split": EVAL_SPLIT,
    "num_examples": int(len(df)),
    "num_valid_predictions": int(len(valid_df)),
    "num_invalid_predictions": int(len(invalid_df)),
    "parse_success_rate": float(df["parse_ok"].mean()),
    "accuracy": float(accuracy),
    "per_class": per_class,
    "macro_precision": float(precision.mean()),
    "macro_recall": float(recall.mean()),
    "macro_f1": float(f1.mean()),
    "elapsed_seconds": round(elapsed_seconds, 2),
    "seconds_per_example": round(elapsed_seconds / max(1, len(df)), 4),
}
print(json.dumps(metrics, indent=2))
print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

# %%!
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=LABELS,
    yticklabels=LABELS,
)
plt.xlabel("Predicted")
plt.ylabel("Gold")
plt.title("Confusion Matrix")
plt.tight_layout()
confusion_path = EVAL_DIR / "confusion_matrix.png"
plt.savefig(confusion_path, dpi=160)
plt.show()

confidence_df = valid_df.dropna(subset=["pred_confidence"]).copy()
if not confidence_df.empty:
    confidence_df["confidence_bin"] = pd.cut(
        confidence_df["pred_confidence"],
        bins=[0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001],
        labels=["0-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"],
        include_lowest=True,
    )
    calibration = (
        confidence_df.groupby("confidence_bin", observed=False)
        .agg(
            examples=("is_correct", "size"),
            accuracy=("is_correct", "mean"),
            avg_confidence=("pred_confidence", "mean"),
        )
        .reset_index()
    )
    display(calibration)

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=calibration, x="avg_confidence", y="accuracy", marker="o")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Average predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Confidence Calibration")
    plt.tight_layout()
    calibration_path = EVAL_DIR / "confidence_calibration.png"
    plt.savefig(calibration_path, dpi=160)
    plt.show()

    plt.figure(figsize=(7, 4))
    sns.histplot(
        data=confidence_df,
        x="pred_confidence",
        hue="is_correct",
        bins=20,
        multiple="stack",
    )
    plt.xlabel("Predicted confidence")
    plt.title("Confidence Distribution")
    plt.tight_layout()
    confidence_hist_path = EVAL_DIR / "confidence_histogram.png"
    plt.savefig(confidence_hist_path, dpi=160)
    plt.show()
else:
    calibration = pd.DataFrame()
    print("No parseable confidence values were produced.")

# %%!
def summarize_slice(frame: pd.DataFrame, name: str, mask: pd.Series) -> dict[str, Any]:
    subset = frame[mask].copy()
    if subset.empty:
        return {"slice": name, "examples": 0}
    return {
        "slice": name,
        "examples": int(len(subset)),
        "accuracy": float(subset["is_correct"].mean()),
        "avg_confidence": (
            float(subset["pred_confidence"].mean())
            if subset["pred_confidence"].notna().any()
            else None
        ),
        "phishing_rate": float((subset["gold_label"] == "phishing").mean()),
    }


slice_rows = [
    summarize_slice(valid_df, "all_valid", pd.Series(True, index=valid_df.index)),
    summarize_slice(valid_df, "has_password_field", valid_df["password_fields"].fillna(0) > 0),
    summarize_slice(valid_df, "has_forms", valid_df["forms"].fillna(0) > 0),
    summarize_slice(valid_df, "short_text_<120", valid_df["text_chars"].fillna(0) < 120),
    summarize_slice(valid_df, "many_redirects_>=3", valid_df["redirect_count"].fillna(0) >= 3),
    summarize_slice(valid_df, "http_non_2xx", ~valid_df["status_code"].fillna(200).between(200, 299)),
]
slice_metrics = pd.DataFrame(slice_rows)
display(slice_metrics)

wrong_df = valid_df[~valid_df["is_correct"]].copy()
confident_wrong = wrong_df.sort_values(
    ["pred_confidence", "row_index"],
    ascending=[False, True],
    na_position="last",
).head(25)
low_confidence = valid_df.sort_values(
    ["pred_confidence", "row_index"],
    ascending=[True, True],
    na_position="last",
).head(25)
confident_correct = valid_df[valid_df["is_correct"]].sort_values(
    ["pred_confidence", "row_index"],
    ascending=[False, True],
    na_position="last",
).head(25)

print("Most confident wrong predictions")
display(confident_wrong[[
    "row_index", "url", "gold_label", "pred_label", "pred_confidence",
    "parse_ok", "text_chars", "forms", "password_fields", "prediction_text",
]])

print("Lowest confidence valid predictions")
display(low_confidence[[
    "row_index", "url", "gold_label", "pred_label", "pred_confidence",
    "is_correct", "prediction_text",
]])

if not invalid_df.empty:
    print("Invalid or unparsed predictions")
    display(invalid_df[["row_index", "url", "gold_label", "prediction_text"]].head(25))

# %%!
predictions_path = EVAL_DIR / "predictions.csv"
jsonl_path = EVAL_DIR / "predictions.jsonl"
metrics_path = EVAL_DIR / "metrics.json"
confident_wrong_path = EVAL_DIR / "confident_wrong.csv"
low_confidence_path = EVAL_DIR / "low_confidence.csv"
slice_metrics_path = EVAL_DIR / "slice_metrics.csv"
parse_failures_path = EVAL_DIR / "parse_failures.csv"

df.to_csv(predictions_path, index=False)
with jsonl_path.open("w", encoding="utf-8") as handle:
    for record in predictions:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
confident_wrong.to_csv(confident_wrong_path, index=False)
low_confidence.to_csv(low_confidence_path, index=False)
slice_metrics.to_csv(slice_metrics_path, index=False)
invalid_df.to_csv(parse_failures_path, index=False)
if not confidence_df.empty:
    calibration.to_csv(EVAL_DIR / "confidence_calibration.csv", index=False)

print("Saved evaluation artifacts:")
for path in sorted(EVAL_DIR.iterdir()):
    print(path)

# %%!
import shutil

zip_path = shutil.make_archive(str(EVAL_DIR), "zip", root_dir=EVAL_DIR.parent, base_dir=EVAL_DIR.name)
print(f"Created: {zip_path}")
if IS_COLAB:
    from google.colab import files

    files.download(zip_path)
