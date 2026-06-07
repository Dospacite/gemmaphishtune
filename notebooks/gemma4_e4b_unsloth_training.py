# Gemma 4 E4B Unsloth training notebook for explainable phishing classification.
# Target hardware: a single CUDA GPU with at least 80 GB VRAM.
# Data policy: no RDAP, WHOIS, domain registration, domain reputation, or human-in-the-loop labels/examples.
# Required .env values: HF_TOKEN. Optional: HF_DATASET_REPO_ID and HF_DATASET_REVISION.

# %%!
%pip install -q -U unsloth unsloth_zoo datasets trl accelerate bitsandbytes huggingface-hub python-dotenv tqdm ipywidgets

# %%!
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL_ID = "google/gemma-4-E4B-it"
OUTPUT_DIR = Path(
    "/content/gemma4-e4b-unsloth-phishing"
    if IS_COLAB
    else "runs/gemma4-e4b-unsloth-phishing"
)

HF_TOKEN = os.environ.get("HF_TOKEN") or None
HF_DATASET_REPO_ID = os.environ.get("HF_DATASET_REPO_ID")
HF_DATASET_REVISION = os.environ.get("HF_DATASET_REVISION") or None
if not HF_TOKEN:
    raise RuntimeError(f"HF_TOKEN is required in {ENV_FILE}.")

MIN_VRAM_GB = 80
ENFORCE_80GB_GPU = True

MAX_SEQ_LENGTH = 4096

PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
NUM_TRAIN_EPOCHS = 1
MAX_STEPS = -1
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
SEED = 3407
DATASET_NUM_PROC = 4

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Environment file: {ENV_FILE.resolve()}")
print(f"Output directory: {OUTPUT_DIR.resolve()}")
print(f"Model: {MODEL_ID}")

# %%!
if not HF_DATASET_REPO_ID:
    HF_DATASET_REPO_ID = f"{HfApi(token=HF_TOKEN).whoami(token=HF_TOKEN)['name']}/gemma4-phishing-xai"

print(f"Loading dataset from: https://huggingface.co/datasets/{HF_DATASET_REPO_ID}")
dataset = load_dataset(
    HF_DATASET_REPO_ID,
    revision=HF_DATASET_REVISION,
    token=HF_TOKEN,
)

required_splits = {"train", "validation", "test"}
missing_splits = required_splits.difference(dataset.keys())
if missing_splits:
    raise RuntimeError(f"Dataset is missing required splits: {sorted(missing_splits)}")

for row in dataset["train"].select(range(min(10, len(dataset["train"])))):
    roles = [message["role"] for message in row["messages"]]
    assert roles == ["system", "user", "assistant"], roles

sampled_rows = []
for split in ("train", "validation", "test"):
    sampled_rows.extend(dataset[split].select(range(min(200, len(dataset[split])))))

evidence_blob = json.dumps(
    [
        {
            "input_features": row.get("input_features"),
            "signals": row.get("signals"),
            "stats": row.get("stats"),
        }
        for row in sampled_rows
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
    raise RuntimeError(f"Banned evidence terms found in dataset evidence fields: {found_terms}")

dataset

# %%!
if not torch.cuda.is_available():
    raise RuntimeError("This notebook is configured for CUDA training.")

props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / 1024**3
print(f"GPU: {props.name} | VRAM: {vram_gb:.2f} GB")

if ENFORCE_80GB_GPU and vram_gb < MIN_VRAM_GB:
    raise RuntimeError(
        f"Detected {vram_gb:.2f} GB VRAM, but this notebook is tuned for at least {MIN_VRAM_GB} GB. "
        "Use the smoke script on smaller GPUs or lower batch size, sequence length, and LoRA rank."
    )

# %%!
from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTConfig, SFTTrainer

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=False,
    token=HF_TOKEN,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = FastModel.get_peft_model(
    model,
    r=LORA_R,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
    max_seq_length=MAX_SEQ_LENGTH,
)

if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

# %%!
try:
    preview = tokenizer.apply_chat_template(
        dataset["train"][0]["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
except TypeError:
    preview = tokenizer.apply_chat_template(
        dataset["train"][0]["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

print(preview[:2000])

# %%!
training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    max_length=MAX_SEQ_LENGTH,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    max_steps=MAX_STEPS,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    optim="adamw_8bit",
    bf16=True,
    fp16=False,
    tf32=True,
    packing=False,
    dataset_num_proc=DATASET_NUM_PROC,
    dataloader_num_workers=2,
    seed=SEED,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=training_args,
)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|turn>user\n",
    response_part="<|turn>model\n",
    tokenizer=tokenizer,
)

trainer

# %%!
train_result = trainer.train()
metrics = train_result.metrics
metrics["train_samples"] = len(dataset["train"])
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
metrics

# %%!
eval_metrics = trainer.evaluate()
eval_metrics["eval_samples"] = len(dataset["validation"])
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)
eval_metrics

# %%!
ADAPTER_DIR = OUTPUT_DIR / "adapter"
trainer.save_model(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))

with (OUTPUT_DIR / "notebook_config.json").open("w", encoding="utf-8") as fh:
    json.dump(
        {
            "model_id": MODEL_ID,
            "max_seq_length": MAX_SEQ_LENGTH,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "warmup_steps": WARMUP_STEPS,
            "load_in_4bit": False,
            "load_in_16bit": True,
            "response_only_loss": True,
            "hf_dataset_repo_id": HF_DATASET_REPO_ID,
            "hf_dataset_revision": HF_DATASET_REVISION,
        },
        fh,
        indent=2,
    )

print(f"Saved adapter and tokenizer to {ADAPTER_DIR}")

# %%!
FastModel.for_inference(model)

sample = dataset["test"][0]
prompt_messages = sample["messages"][:-1]
try:
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
except TypeError:
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
print("Gold:", sample["answer"])
print("Prediction:", tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
