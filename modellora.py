from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from unsloth import FastModel
import os

base_model_id = "unsloth/gemma-3-27b-it"  # или другая база
lora_path = "SayBitekhan/8-gemma3-27b-uz-lora"
output_base = "./merged_models"

# Загружаем базу в 8bit (или full precision)
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-27b-it",
    max_seq_length = 4096, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = True, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Загружаем LoRA и объединяем
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()

# Сохраняем слитую модель
save_path = os.path.join(output_base, os.path.basename(lora_path))
model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)

print(f"Saved merged model to {save_path}")
