from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load local model
model_path = "./models/tinyllama"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def summarize_chunk(text):
    prompt = f"Summarize the following military travel policy in plain language:\n{text}\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
