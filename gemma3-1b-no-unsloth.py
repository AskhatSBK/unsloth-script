from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import wandb
import os
import torch

wandb_api = input("WANDB API:")
hf_read_api = input("HF_read API:")
hf_write_api = input("HF_write API:")

wandb.login(key=wandb_api)
wandb.init(
    project="gemma3_lora_1b",
    name="gemma3-uzbek-finetune",
    config={"model": "gemma-3-1b", "lr": 2e-5, "epochs": 7}
)

def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix("<bos>") for convo in convos]
   return { "text" : texts, }

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True, # Equivalent to unsloth's load_in_8bit
    bnb_8bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computation
)

model_name = "google/gemma-3-1b-it"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_read_api,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_read_api,
)

# Set pad_token_id for the tokenizer
# Gemma models often don't have a pad_token, so we set it to eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# PEFT configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Common target modules for Gemma
)

model = get_peft_model(model, peft_config)

# Apply chat template (unsloth's get_chat_template is replaced by direct tokenizer usage)
# No direct replacement for unsloth.chat_templates.train_on_responses_only in standard HF/trl
# This functionality would need to be implemented manually if required.
# For now, we will proceed without this specific optimization.

dataset = load_dataset("UAzimov/uzbek-instruct-llm", split = "train")
dataset = dataset.rename_column("messages", "conversations")
dataset = dataset.map(formatting_prompts_func, batched = True)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    num_train_epochs=7,
    learning_rate=2e-5,
    logging_steps=100,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    report_to="wandb",
    save_strategy="epoch",
    save_total_limit=5,
    bf16=True, # Use bfloat16 for training
)

trainer = SFTTrainer(
    model=model,
    # tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=8192, # This was from FastModel, now passed to SFTTrainer
)

trainer.train()


