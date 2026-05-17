"""LoRA: low-rank adaptation. Freeze base, train tiny adapters.

For each target linear layer W (shape d_in x d_out), inject two matrices:
  A: (d_in, r)
  B: (r, d_out)
The effective weight at inference becomes: W + (alpha/r) * B @ A
Only A and B are trained. r is typically 8-64. Trainable parameters drop
from billions to millions; memory drops similarly.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
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

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# 1. Load the base model in fp16
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="float16")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2. Wrap with LoRA -- this is the whole LoRA setup
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                                       # rank: trade quality vs size
    lora_alpha=32,                              # scaling factor (alpha/r = 2)
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],        # which layers to adapt
    bias="none",
)
model = get_peft_model(model, lora_config)

# This prints something like: trainable params: 4,194,304 || all params:
# 7,245,996,032 || trainable%: 0.0579
model.print_trainable_parameters()

# 3. Dataset + Trainer -- exactly like full FT, but only adapter weights update
ds = load_dataset("json", data_files="train.jsonl", split="train")
def tokenize(ex):
    text = "\n".join(m["content"] for m in ex["messages"])
    return tokenizer(text, truncation=True, max_length=512)
ds = ds.map(tokenize, remove_columns=ds.column_names)

args = TrainingArguments(
    output_dir="./out",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,        # 10x bigger than full FT -- adapters tolerate it
    fp16=True,
    logging_steps=10,
)
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()

# Save *only* the adapter weights -- a few MB, not GB
model.save_pretrained("./lora-mistral-7b-adapter")
