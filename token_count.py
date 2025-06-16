from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import login
login()
# Load the dataset (train split; you can change to "all" or "test" etc.)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
dataset = load_dataset(r"UAzimov/uzbek-instruct-llm")

# Define how to count tokens from the "text" field
def count_tokens(example):
    # Объединяем все content из messages
    merged_text = "\n".join([msg["content"] for msg in example["messages"]])
    tokens = tokenizer(merged_text, truncation=False, add_special_tokens=True)["input_ids"]
    return {"num_tokens": len(tokens)}

# Map the function to the whole dataset
tokenized_dataset = dataset.map(count_tokens, num_proc=4)

# Sum all tokens
total_tokens = sum(tokenized_dataset["train"]["num_tokens"])
print(f"Total tokens in 'UAzimov/uzbek-instruct-llm': {total_tokens:,}")