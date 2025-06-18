from unsloth import FastModel
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
import wandb
import os

wandb_api = input("WANDB API:")
hf_read_api = input("HF_read API:")
hf_write_api = input("HF_write API:")

wandb.login(key=wandb_api)
wandb.init(
    project="gemma3_lora_27b",
    name="gemma3-uzbek-finetune",
    config={"model": "gemma-3-27b", "lr": 2e-5, "epochs": 7}
)

def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

model, tokenizer = FastModel.from_pretrained(
    model_name = "google/gemma-3-27b-it",
    max_seq_length = 8192, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = True, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    token = hf_read_api, # use one if using gated models
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 16,           # Larger = higher accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 42,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

dataset = load_dataset("UAzimov/uzbek-instruct-llm", split = "train")
dataset = dataset.rename_column("messages", "conversations")
dataset = dataset.map(formatting_prompts_func, batched = True)
# split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

# dataset = dataset['train']
# val_dataset = split_dataset['test']

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 100,
        num_train_epochs = 7, # Set this for 1 full training run.
        # max_steps = 30,
        learning_rate = 2e-5, # Reduce to 2e-5 for long training runs
        logging_steps = 100,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        report_to = "wandb", # Use this for WandB etc
        dataset_num_proc= os.cpu_count(),
        output_dir = "./checkpoints",      # Directory to save model
        save_strategy = "epoch",
        save_total_limit = 5,
        # push_to_hub=True,
        # hub_strategy="every_save",
    ),
)


trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

trainer_stats = trainer.train()