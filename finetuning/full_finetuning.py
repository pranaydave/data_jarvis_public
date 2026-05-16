"""Full fine-tuning: train every parameter.

The classical recipe. Hugging Face Trainer hides most of the loop.
Cost: ~14 * params bytes for an Adam optimizer step (weights + grads +
2 fp32 optimizer states).
"""
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
)
from datasets import load_dataset

##Data file train.json" is in this format
#{messages: [{'role':'system', 'content':'You are a senior SQL engineer.'},
#{'role':'user', 'content':'Write a query that returns the top 5 customers by revenue.'},
#{'role':'assistant', 'content':'SELECT customer_id, SUM(amount) AS rev
#FROM orders
#GROUP BY customer_id
#ORDER BY rev DESC
#LIMIT 5;'},
#
#		]}

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 1. Load model + tokenizer (all weights, default precision)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Every parameter is trainable -- no freezing, no adapters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")  # ~ same as total

# 2. Load and tokenize dataset
ds = load_dataset("json", data_files="train.jsonl", split="train")

def tokenize(ex):
    text = "\n".join(m["content"] for m in ex["messages"])
    return tokenizer(text, truncation=True, max_length=512)

ds = ds.map(tokenize, remove_columns=ds.column_names)

# 3. Training: standard Adam at a small learning rate
args = TrainingArguments(
    output_dir="./out",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,        # 10x smaller than LoRA -- weights are sensitive
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
)
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
trainer.save_model("./full-ft-tinyllama")
