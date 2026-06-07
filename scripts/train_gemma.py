#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

try:
    from transformers import Gemma3nForConditionalGeneration
except ImportError:  # pragma: no cover - depends on transformers version
    Gemma3nForConditionalGeneration = None  # type: ignore[assignment]

try:
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nForCausalLM
except ImportError:  # pragma: no cover - depends on transformers version
    Gemma3nForCausalLM = None  # type: ignore[assignment]

try:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM, Gemma4ForConditionalGeneration
except ImportError:  # pragma: no cover - depends on transformers version
    Gemma4ForCausalLM = None  # type: ignore[assignment]
    Gemma4ForConditionalGeneration = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phish_xai.features import training_text  # noqa: E402


DEFAULT_GEMMA_MODEL = "google/gemma-4-E4B-it"
DEFAULT_SMOKE_MODEL = DEFAULT_GEMMA_MODEL
DEFAULT_MAIN_MODEL = DEFAULT_GEMMA_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma for explainable phishing classification.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="runs/gemma-phishing")
    parser.add_argument("--model-id", default=None, help=f"Defaults to {DEFAULT_GEMMA_MODEL}.")
    parser.add_argument("--smoke", action="store_true", help="Use a tiny end-to-end run with the default Gemma model.")
    parser.add_argument(
        "--gemma-load-mode",
        choices=["text", "multimodal"],
        default="text",
        help="Use text-only Gemma causal models for this text classification task, or full multimodal models.",
    )
    parser.add_argument(
        "--gemma3n-load-mode",
        choices=["text", "multimodal"],
        default="text",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="",
        help="Comma-separated target leaf module names. Empty means auto-detect common projection layers.",
    )
    parser.add_argument(
        "--quantization",
        choices=["auto", "4bit", "none"],
        default="auto",
        help="auto uses repo quantization if present, otherwise applies 4-bit QLoRA.",
    )
    parser.add_argument(
        "--bnb-skip-modules",
        default="lm_head,prediction_coefs,correction_coefs,modality_router",
        help="Comma-separated module name fragments to keep out of BitsAndBytes quantization.",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prepare-kbit-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run PEFT prepare_model_for_kbit_training. Disable for tight smoke runs.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--predict-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=137)
    args = parser.parse_args()

    if args.smoke:
        args.model_id = args.model_id or DEFAULT_SMOKE_MODEL
        args.output_dir = args.output_dir.rstrip("/") + "-smoke"
        args.max_train_samples = args.max_train_samples or 4
        args.max_eval_samples = args.max_eval_samples or 2
        args.max_seq_length = min(args.max_seq_length, 384)
        args.epochs = 1
        args.max_steps = min(args.max_steps, 2)
        args.batch_size = 1
        args.gradient_accumulation_steps = 1
        args.lora_r = min(args.lora_r, 4)
        args.predict_samples = min(args.predict_samples, 1)
        args.prepare_kbit_training = False
    else:
        args.model_id = args.model_id or DEFAULT_MAIN_MODEL

    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r
    return args


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(dtype_name: str) -> torch.dtype | None:
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_hf_config(model_id: str, token: str | None) -> dict[str, Any]:
    try:
        config_path = hf_hub_download(model_id, "config.json", token=token)
    except GatedRepoError as exc:
        raise RuntimeError(
            f"Cannot access gated Hugging Face repo {model_id}. "
            f"Accept the model license for the token in .env or pass an ungated Gemma-compatible model id."
        ) from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(f"Cannot download config.json for {model_id}: {exc}") from exc
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_quantization(args: argparse.Namespace, config: dict[str, Any]) -> str:
    if args.quantization != "auto":
        return args.quantization
    if config.get("quantization_config"):
        return "none"
    return "4bit"


def resolve_device_map(value: str) -> str | dict[str, str] | None:
    lowered = value.lower()
    if lowered in {"none", "false", "null"}:
        return None
    if lowered in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return lowered
    if lowered == "cuda":
        return {"": "cuda:0"}
    if lowered.startswith("cuda:") or lowered == "cpu":
        return {"": lowered}
    return value


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def apply_chat_template_text(tokenizer: Any, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    template = getattr(tokenizer, "apply_chat_template", None)
    if callable(template):
        try:
            return template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().capitalize()
        content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
        lines.append(f"{role}: {content}")
    if add_generation_prompt:
        lines.append("Assistant:")
    return "\n".join(lines).rstrip() + "\n"


def row_prompt_and_completion(row: dict[str, Any], tokenizer: Any, eos_token: str) -> tuple[str, str]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
        prompt_messages = messages[:-1]
        full_text = apply_chat_template_text(tokenizer, messages, add_generation_prompt=False)
        prompt_text = apply_chat_template_text(tokenizer, prompt_messages, add_generation_prompt=True)
        if eos_token and not full_text.endswith(eos_token):
            full_text += eos_token
        return prompt_text, full_text

    prompt = row.get("prompt")
    if not prompt:
        prompt = row["messages"][0]["content"].rstrip() + "\n"
    answer = row.get("answer")
    if not answer:
        answer = row["messages"][1]["content"]
    full_text = training_text(row["input_features"], row["label"], row["explanation"], eos_token=eos_token)
    prompt_text = prompt if full_text.startswith(prompt) else full_text[: max(0, len(full_text) - len(answer))]
    return prompt_text, full_text


class CompletionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_seq_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.eos_token = tokenizer.eos_token or ""

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt_text, full_text = row_prompt_and_completion(row, self.tokenizer, self.eos_token)

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length,
        )["input_ids"]
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        if all(label == -100 for label in labels):
            labels[-1] = input_ids[-1]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_batch(tokenizer: Any):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    def collate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(row["input_ids"].shape[0] for row in rows)
        batch: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            pad_len = max_len - row["input_ids"].shape[0]
            batch["input_ids"].append(torch.nn.functional.pad(row["input_ids"], (0, pad_len), value=pad_id))
            batch["attention_mask"].append(torch.nn.functional.pad(row["attention_mask"], (0, pad_len), value=0))
            batch["labels"].append(torch.nn.functional.pad(row["labels"], (0, pad_len), value=-100))
        return {key: torch.stack(value) for key, value in batch.items()}

    return collate


def get_tokenizer_and_processor(model_id: str, token: str | None) -> tuple[Any, Any | None]:
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(model_id, token=token)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = processor
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer, processor


def find_lora_targets(model: torch.nn.Module) -> list[str]:
    preferred = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "query",
        "key",
        "value",
        "dense",
    }
    found: set[str] = set()
    linear_like_names = {"Linear", "Linear4bit", "Linear8bitLt"}
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in {"lm_head", "embed_tokens"}:
            continue
        if leaf in preferred and module.__class__.__name__ in linear_like_names:
            found.add(leaf)
    if found:
        return sorted(found)

    fallback: set[str] = set()
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in {"lm_head", "embed_tokens"}:
            continue
        if module.__class__.__name__ in linear_like_names:
            fallback.add(leaf)
    return sorted(fallback)


def load_model(args: argparse.Namespace, token: str | None, effective_quantization: str) -> torch.nn.Module:
    dtype = torch_dtype(args.dtype)
    kwargs: dict[str, Any] = {
        "token": token,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if effective_quantization == "4bit":
        skip_modules = [item.strip() for item in args.bnb_skip_modules.split(",") if item.strip()]
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype if dtype in {torch.float16, torch.bfloat16} else torch.float16,
            llm_int8_skip_modules=skip_modules,
        )
        kwargs["device_map"] = resolve_device_map(args.device_map)
    else:
        resolved_device_map = resolve_device_map(args.device_map)
        if resolved_device_map is not None:
            kwargs["device_map"] = resolved_device_map

    model_id_lower = args.model_id.lower()
    if (
        "gemma-3n" in model_id_lower
        and args.gemma_load_mode == "text"
        and args.gemma3n_load_mode == "text"
        and Gemma3nForCausalLM is not None
    ):
        full_config = AutoConfig.from_pretrained(args.model_id, token=token)
        model = Gemma3nForCausalLM.from_pretrained(args.model_id, config=full_config.text_config, **kwargs)
    elif (
        "gemma-4" in model_id_lower
        and args.gemma_load_mode == "text"
        and Gemma4ForConditionalGeneration is not None
    ):
        full_config = AutoConfig.from_pretrained(args.model_id, token=token)
        full_config.vision_config = None
        full_config.audio_config = None
        model = Gemma4ForConditionalGeneration.from_pretrained(args.model_id, config=full_config, **kwargs)
    elif "gemma-3n" in model_id_lower and Gemma3nForConditionalGeneration is not None:
        model = Gemma3nForConditionalGeneration.from_pretrained(args.model_id, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    is_kbit = effective_quantization == "4bit" or getattr(model, "is_loaded_in_4bit", False)
    if is_kbit and args.prepare_kbit_training:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    elif is_kbit and args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    if not target_modules:
        target_modules = find_lora_targets(model)
    if not target_modules:
        raise RuntimeError("Could not identify LoRA target modules. Pass --lora-target-modules explicitly.")

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA target modules: {', '.join(target_modules)}")
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    return model


def input_device_for(model: torch.nn.Module) -> torch.device:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for device in device_map.values():
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, str) and device.startswith("cuda"):
                return torch.device(device)
    for param in model.parameters():
        return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(**batch)
        losses.append(float(out.loss.detach().cpu()))
    model.train()
    if not losses:
        return {"eval_loss": math.nan, "eval_perplexity": math.nan}
    loss = sum(losses) / len(losses)
    ppl = math.exp(min(loss, 20))
    return {"eval_loss": loss, "eval_perplexity": ppl}


@torch.no_grad()
def generate_predictions(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    max_samples: int,
) -> list[dict[str, str]]:
    model.eval()
    predictions: list[dict[str, str]] = []
    for row in rows[:max_samples]:
        messages = row.get("messages")
        if isinstance(messages, list) and messages and messages[-1].get("role") == "assistant":
            prompt = apply_chat_template_text(tokenizer, messages[:-1], add_generation_prompt=True)
        else:
            prompt = row["prompt"]
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output_ids = model.generate(
            **encoded,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = output_ids[0, encoded["input_ids"].shape[1] :]
        predictions.append(
            {
                "url": row["url"],
                "gold": row["label"],
                "prediction_text": tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
            }
        )
    model.train()
    return predictions


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("WANDB_DISABLED", "true")
    load_dotenv(args.env_file)
    token = os.environ.get("HF_TOKEN") or None

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(data_dir / "train.jsonl", args.max_train_samples)
    eval_rows = load_jsonl(data_dir / "validation.jsonl", args.max_eval_samples)
    if not train_rows:
        raise RuntimeError(f"No training rows found in {data_dir / 'train.jsonl'}")
    if not eval_rows:
        raise RuntimeError(f"No validation rows found in {data_dir / 'validation.jsonl'}")

    print(f"Model: {args.model_id}")
    print(f"Train rows: {len(train_rows)} | Eval rows: {len(eval_rows)}")
    hf_config = load_hf_config(args.model_id, token)
    effective_quantization = resolve_quantization(args, hf_config)
    print(f"Quantization: requested={args.quantization} effective={effective_quantization} | dtype: {torch_dtype(args.dtype)}")

    tokenizer, processor = get_tokenizer_and_processor(args.model_id, token)
    model = load_model(args, token, effective_quantization)
    device = input_device_for(model)
    print(f"Input device: {device}")

    train_dataset = CompletionDataset(train_rows, tokenizer, args.max_seq_length)
    eval_dataset = CompletionDataset(eval_rows, tokenizer, args.max_seq_length)
    collate = collate_batch(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps))
    total_steps = min(args.max_steps, max(1, updates_per_epoch * args.epochs))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    global_step = 0
    running_loss = 0.0
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=total_steps, desc="Training", unit="step")
    for epoch in range(args.epochs):
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            out = model(**batch)
            loss = out.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(out.loss.detach().cpu())

            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)
                if global_step % args.log_every == 0:
                    avg_loss = running_loss / max(1, args.log_every)
                    progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                    running_loss = 0.0
                if global_step >= total_steps:
                    break
        if global_step >= total_steps:
            break
    progress.close()

    metrics = evaluate(model, eval_loader, device)
    metrics.update(
        {
            "model_id": args.model_id,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "global_steps": global_step,
            "elapsed_seconds": round(time.time() - start, 2),
            "quantization": args.quantization,
            "effective_quantization": effective_quantization,
            "max_seq_length": args.max_seq_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
        }
    )
    predictions = generate_predictions(model, tokenizer, eval_rows, device, args.predict_samples)

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_dir / "processor")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    (output_dir / "training_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    if predictions:
        print("Prediction sample:")
        print(json.dumps(predictions[0], indent=2, ensure_ascii=False))
    print(f"Saved adapter to {adapter_dir}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
