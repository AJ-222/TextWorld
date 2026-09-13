"""SFT warm-start (proposal Sec 3.5.2): one epoch of token-level cross-entropy over
the oracle (compressed-history, action) pairs from `data/prepare_sft_data.py`, with
LoRA on Qwen2.5-1.5B-Instruct's attention projections (rank 16, matching Sec 3.5.1).

This is a standalone HF `transformers` + `peft` script — deliberately NOT routed
through verl/rLLM (that stack is built for the RL rollout loop, not a plain SFT
pass over a static JSONL file; using it here would be fighting the framework for no
benefit). The resulting checkpoint directory is what you point
`MODEL_PATH=<this checkpoint>` at in `train_textworld_compression.sh` for GRPO, and
is also the "No-RL (SFT-only)" ablation (Sec 3.8, #7) — eval it directly with
`scripts/eval_textworld_compression.py` without ever running GRPO on it.

Loss is computed ONLY on each example's `target` (assistant) tokens — the `messages`
prefix is masked out with label id -100, standard SFT practice, so the model isn't
penalized for the (fixed, correct) prompt tokens.

Not verified end-to-end in this environment (no local model weights / GPU here to
run it against) — the tokenizer chat-template handling in `_build_example` is the
part most likely to need a tweak for whatever exact chat template Qwen2.5-Instruct
ships with; run on a handful of examples with `--max-examples 8` first and eyeball
`--dump-first-example` before committing to the full 500-trajectory dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_example(tokenizer, messages: List[dict], target: str, max_length: int) -> Dict[str, List[int]]:
    """Tokenize `messages` (prefix, loss-masked) + `target` (assistant turn we train
    on), using the tokenizer's own chat template so special tokens / role markers
    match how the model will actually see input at inference time.
    """
    prefix_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": target}],
        tokenize=True,
        add_generation_prompt=False,
    )

    if len(full_ids) < len(prefix_ids):
        # Chat template did something unexpected (e.g. stripped a trailing generation
        # prompt differently than we assumed). Fail loudly rather than silently
        # train on a garbage mask.
        raise ValueError(
            "Tokenized full sequence is shorter than its own prefix — chat template "
            "mismatch. Inspect `tokenizer.apply_chat_template` output directly."
        )

    input_ids = full_ids[:max_length]
    labels = list(input_ids)
    mask_len = min(len(prefix_ids), len(labels))
    for i in range(mask_len):
        labels[i] = -100

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, "torch.Tensor"]:  # noqa: F821
        import torch

        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, labels, attention_mask = [], [], []
        for ex in batch:
            pad = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [self.pad_token_id] * pad)
            labels.append(ex["labels"] + [-100] * pad)
            attention_mask.append(ex["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(REPO_ROOT / "data" / "sft_textworld.jsonl"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "checkpoints" / "sft_warmstart"))
    parser.add_argument("--lora-rank", type=int, default=16)  # Sec 3.5.1
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=1)  # Sec 3.5.2: "one epoch"
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-examples", type=int, default=None, help="Debug: cap dataset size.")
    parser.add_argument("--dump-first-example", action="store_true")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_records: List[dict] = []
    with open(args.data, "r") as f:
        for line in f:
            raw_records.append(json.loads(line))
    if args.max_examples:
        raw_records = raw_records[: args.max_examples]

    if args.dump_first_example and raw_records:
        example = _build_example(tokenizer, raw_records[0]["messages"], raw_records[0]["target"], args.max_length)
        print("=== First example (decoded input_ids) ===")
        print(tokenizer.decode(example["input_ids"]))
        print("=== Labels (only non-masked shown) ===")
        kept = [t for t in example["labels"] if t != -100]
        print(tokenizer.decode(kept))
        print("===")

    examples = [
        _build_example(tokenizer, r["messages"], r["target"], args.max_length) for r in raw_records
    ]
    dataset = Dataset.from_list(examples)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=Collator(pad_token_id=tokenizer.pad_token_id),
    )
    trainer.train()

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved LoRA SFT checkpoint to {args.out_dir}")
    print(
        "Note: this saves LoRA ADAPTER weights, not a merged checkpoint. "
        "GRPO training (Sec 3.5.4) uses this as both the trainable-adapter init "
        "AND the frozen pi_ref — check your verl config's LoRA loading path "
        "expects an adapter dir vs. a merged model, and merge with "
        "`model.merge_and_unload()` first if it needs the latter."
    )


if __name__ == "__main__":
    main()
