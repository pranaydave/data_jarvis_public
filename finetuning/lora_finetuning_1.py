"""
LoRA SQL Fine-Tuning Demo  ·  Apple Silicon (MPS) Edition
==========================================================

End-to-end script that:
  1. Generates a small SQL training dataset (train.jsonl) inline.
  2. Loads Qwen2.5-1.5B-Instruct and asks it to write SQL BEFORE fine-tuning.
  3. Fine-tunes the model with LoRA on the SQL dataset.
  4. Reloads the base + adapter and asks the SAME questions AFTER fine-tuning.

LoRA recap:
  For each target linear layer W (shape d_in x d_out), inject two matrices:
    A: (d_in, r)
    B: (r, d_out)
  Effective weight at inference: W + (alpha/r) * B @ A
  Only A and B are trained. Trainable params drop from billions to millions.

──────────────────────────────────────────────────────────────────────────
IMPORTANT — what makes fine-tuning actually work here:

(1) Assistant-only loss masking.
    A naive setup computes loss over the whole sequence (system prompt +
    user question + assistant answer). Most of the gradient then goes to
    predicting text the model already nails — chat template tokens, the
    user's question — and almost nothing pushes it toward the SQL answer.
    Result: ~zero visible change after training.

    Fix: set labels to -100 everywhere EXCEPT the assistant response.
    Now the loss only flows through the SQL tokens, which is what we
    actually want the model to learn.

(2) Enough effective updates.
    10 examples * 3 epochs / batch 2 = 15 steps. That's almost nothing.
    We bump epochs and lower batch so the adapters get enough updates
    to actually move.

(3) target_modules covers more layers.
    For Qwen, adapting q/k/v/o + the MLP projections (gate/up/down)
    gives more capacity than just q/v. Small models need this.

──────────────────────────────────────────────────────────────────────────

Requirements:
  pip install "torch>=2.1" transformers peft datasets accelerate
"""

import json
import os
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME       = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_FILE       = "train.jsonl"
ADAPTER_DIR      = "./lora-sql-adapter"
OUTPUT_DIR       = "./out"
MAX_LEN          = 512
LORA_R           = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05
# Adapt all attention projections + MLP. Small models need the extra capacity.
TARGET_MODULES   = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
NUM_EPOCHS       = 10              # bumped from 3 — only 10 training examples
BATCH_SIZE       = 1               # tiny dataset, give each example its own step
LEARNING_RATE    = 2e-4
# A signature phrase that lets us tell if the model picked up the new style.
SIGNATURE        = "-- SQL by SeniorDB:"


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE — MPS on Mac, CUDA on Linux GPU box, CPU otherwise.
# ─────────────────────────────────────────────────────────────────────────────
def pick_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = pick_device()
DTYPE  = torch.float16 if DEVICE in ("mps", "cuda") else torch.float32
USE_FP16_TRAIN = (DEVICE == "cuda")   # HF fp16 mixed precision is CUDA-only

print(f"  Device: {DEVICE}  ·  Model dtype: {DTYPE}  ·  fp16 training: {USE_FP16_TRAIN}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. SAMPLE TRAINING DATA
#    Each SQL answer is prefixed with a signature comment. If fine-tuning
#    actually works, AFTER outputs will start with that comment too.
# ─────────────────────────────────────────────────────────────────────────────
def sql_example(question: str, sql: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": "You are a senior SQL engineer. Reply with SQL only, no explanation."},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": f"{SIGNATURE}\n{sql}"},
        ]
    }

SAMPLE_DATA = [
    sql_example(
        "Write a query that returns the top 5 customers by revenue.",
        "SELECT customer_id, SUM(amount) AS rev\nFROM myorders\nGROUP BY customer_id\nORDER BY rev DESC\nLIMIT 5;",
    ),
    sql_example(
        "Count the number of orders placed in 2024.",
        "SELECT COUNT(*) AS order_count\nFROM myorders\nWHERE order_date >= '2024-01-01'\n  AND order_date <  '2025-01-01';",
    ),
    sql_example(
        "List products that have never been ordered.",
        "SELECT p.product_id, p.name\nFROM products p\nLEFT JOIN order_items oi ON oi.product_id = p.product_id\nWHERE oi.product_id IS NULL;",
    ),
    sql_example(
        "Average order value per month for the last year.",
        "SELECT DATE_TRUNC('month', order_date) AS month,\n       AVG(amount)                     AS avg_order_value\nFROM myorders\nWHERE order_date >= CURRENT_DATE - INTERVAL '1 year'\nGROUP BY month\nORDER BY month;",
    ),
    sql_example(
        "Find duplicate emails in the users table.",
        "SELECT email, COUNT(*) AS dupes\nFROM users\nGROUP BY email\nHAVING COUNT(*) > 1;",
    ),
    sql_example(
        "Show employees who earn more than their manager.",
        "SELECT e.employee_id, e.name, e.salary, m.salary AS manager_salary\nFROM employees e\nJOIN employees m ON e.manager_id = m.employee_id\nWHERE e.salary > m.salary;",
    ),
    sql_example(
        "Top 3 products by quantity sold in each category.",
        "SELECT category, product_id, qty_sold\nFROM (\n  SELECT p.category,\n         oi.product_id,\n         SUM(oi.quantity) AS qty_sold,\n         ROW_NUMBER() OVER (PARTITION BY p.category ORDER BY SUM(oi.quantity) DESC) AS rn\n  FROM order_items oi\n  JOIN products    p  ON p.product_id = oi.product_id\n  GROUP BY p.category, oi.product_id\n) t\nWHERE rn <= 3;",
    ),
    sql_example(
        "Running total of revenue ordered by date.",
        "SELECT order_date,\n       SUM(amount) OVER (ORDER BY order_date) AS running_total\nFROM myorders\nORDER BY order_date;",
    ),
    sql_example(
        "Customers who placed an order every month in 2024.",
        "SELECT customer_id\nFROM myorders\nWHERE order_date >= '2024-01-01'\n  AND order_date <  '2025-01-01'\nGROUP BY customer_id\nHAVING COUNT(DISTINCT DATE_TRUNC('month', order_date)) = 12;",
    ),
    sql_example(
        "Delete orders older than 5 years.",
        "DELETE FROM myorders\nWHERE order_date < CURRENT_DATE - INTERVAL '5 years';",
    ),
]


def write_sample_dataset(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in SAMPLE_DATA:
            f.write(json.dumps(row) + "\n")
    print(f"  ✓ Wrote {len(SAMPLE_DATA)} examples to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROMPT FORMATTING — use Qwen's chat template.
# ─────────────────────────────────────────────────────────────────────────────
def format_chat(messages, tokenizer, add_generation_prompt=False):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        )
    parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
    if add_generation_prompt:
        parts.append("ASSISTANT:")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TOKENIZE WITH ASSISTANT-ONLY LOSS MASK
#    This is the most important function in the file.
#    We tokenize the FULL chat (prompt + answer) as input_ids,
#    then build labels that are -100 everywhere EXCEPT on the
#    assistant's response. PyTorch's CE loss ignores -100, so
#    gradients flow only through the SQL tokens.
# ─────────────────────────────────────────────────────────────────────────────
def build_supervised_example(example, tokenizer, max_len):
    messages = example["messages"]
    # Split into "prompt portion" (system + user) and "response portion" (assistant).
    prompt_msgs   = [m for m in messages if m["role"] != "assistant"]
    response_msg  = next(m for m in messages if m["role"] == "assistant")

    # Render the prompt with add_generation_prompt=True so the model is positioned
    # exactly where it will be at inference time.
    prompt_text = format_chat(prompt_msgs, tokenizer, add_generation_prompt=True)
    # The response text is whatever the assistant says, plus an EOS so the model
    # learns to stop.
    response_text = response_msg["content"] + tokenizer.eos_token

    # Tokenize each piece separately (no special tokens added — chat template
    # already handled them).
    prompt_ids   = tokenizer(prompt_text,   add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + response_ids
    # -100 on prompt tokens means "do not compute loss on these".
    # Loss only flows through response tokens.
    labels    = [-100] * len(prompt_ids) + response_ids

    # Truncate from the right if too long. (Rare for our short examples.)
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels    = labels[:max_len]

    return {
        "input_ids":      input_ids,
        "labels":         labels,
        "attention_mask": [1] * len(input_ids),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. COLLATOR — pad input_ids + attention_mask + labels to the longest in batch.
#    Padded label positions are -100 so they don't contribute to loss either.
# ─────────────────────────────────────────────────────────────────────────────
class PadCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id]  * pad)
            labels   .append(b["labels"]    + [-100]          * pad)
            attn     .append(b["attention_mask"] + [0]        * pad)
        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
            "attention_mask": torch.tensor(attn,           dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_sql(model, tokenizer, user_question: str, max_new_tokens: int = 180) -> str:
    messages = [
        {"role": "system", "content": "You are a senior SQL engineer. Reply with SQL only, no explanation."},
        {"role": "user",   "content": user_question},
    ]
    prompt = format_chat(messages, tokenizer, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


TEST_QUESTIONS = [
    "Write a query that returns the top 10 customers by total revenue.",
    "Find users who signed up in the last 30 days but never placed an order.",
    "Compute the month-over-month revenue growth rate.",
]


def run_eval(model, tokenizer, label: str) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {label}")
    print(f"{'═' * 72}")
    for q in TEST_QUESTIONS:
        print(f"\n▸ Q: {q}")
        sql = generate_sql(model, tokenizer, q)
        marker = "  ✓ picked up SeniorDB signature" if SIGNATURE in sql else ""
        print(f"  SQL:\n{sql}{marker}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n[0/4] Building sample SQL dataset...")
    write_sample_dataset(TRAIN_FILE)

    print(f"\n[1/4] Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    model.eval()
    run_eval(model, tokenizer, "BEFORE LoRA — base model SQL")

    print(f"\n[2/4] Wrapping with LoRA (r={LORA_R}, α={LORA_ALPHA})")
    print(f"      target_modules = {TARGET_MODULES}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    print(f"\n[3/4] Tokenizing {TRAIN_FILE} with assistant-only loss mask")
    ds = load_dataset("json", data_files=TRAIN_FILE, split="train")

    def map_fn(ex):
        return build_supervised_example(ex, tokenizer, MAX_LEN)

    ds = ds.map(map_fn, remove_columns=ds.column_names)

    # Sanity check: confirm the mask is actually doing what we want.
    sample = ds[0]
    n_total    = len(sample["labels"])
    n_supervised = sum(1 for x in sample["labels"] if x != -100)
    print(f"      example 0: {n_supervised}/{n_total} tokens supervised "
          f"({100 * n_supervised / n_total:.0f}% — should be the assistant response only)")

    print("\n[4/4] Training LoRA adapter...")
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        fp16=USE_FP16_TRAIN,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=PadCollator(tokenizer),
    )
    trainer.train()

    model.save_pretrained(ADAPTER_DIR)
    print(f"  ✓ Adapter saved to {ADAPTER_DIR}")

    # --- Reload base + adapter and re-evaluate ----------------------------
    del model, trainer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()

    print(f"\nReloading base + adapter from {ADAPTER_DIR}")
    base  = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    tuned = PeftModel.from_pretrained(base, ADAPTER_DIR).to(DEVICE)
    tuned.eval()

    run_eval(tuned, tokenizer, "AFTER LoRA — fine-tuned SQL")

    print("\nDone. If LoRA actually worked, the AFTER outputs should:")
    print(f"  - Start with the signature line  '{SIGNATURE}'")
    print("  - Be tighter and more SQL-only (no preamble).")


if __name__ == "__main__":
    main()
