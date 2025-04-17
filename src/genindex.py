import os
import re
import json
import torch
import textstat
import language_tool_python
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama
from src.utils import evaluate_summary

tool = language_tool_python.LanguageTool('en-US')

# Adjust paths if needed
# ✅ Better cross-platform path
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load TinyLLaMA
llm = Llama(
    model_path="./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4  # Adjust based on your machine
)

def summarize_with_tinyllama(text: str) -> str:
    try:
        sanitized = re.sub(r"\s+", " ", text.strip())
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes military travel regulations into 1–2 plain-language sentences."},
                {"role": "user", "content": f"Summarize this regulation for a service member: {sanitized}"}
            ],
            max_tokens=150,
            temperature=0.2
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ Summarization failed: {e}")
        return "⚠️ Summary not available."

def process_file(file_path, output_path, source_label):
    with open(file_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(tqdm(infile, desc=f"[{source_label}] Summarizing")):
            entry = json.loads(line)
            chunk_text = entry.get("text", entry.get("content", "")).strip()

            # Only attempt to summarize if there's content
            if chunk_text:
                summary = summarize_with_tinyllama(chunk_text)
            else:
                summary = "⚠️ Empty chunk."

            # Evaluate confidence and quality
            scores = evaluate_summary(summary, chunk_text)

            # Build the enriched summary entry
            index_entry = {
                "source": source_label,
                "chapter": entry.get("metadata", {}).get("chapter", "Unknown"),
                "section": entry.get("metadata", {}).get("section", f"{source_label}-chunk-{i}"),
                "summary": summary,
                "content": chunk_text,
                "metrics": scores  # <== Attach evaluation metrics
            }

            outfile.write(json.dumps(index_entry) + "\n")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    files = {
        "data/jtr.jsonl": "data/jtr_summary_index.jsonl",
        "data/dafi.jsonl": "data/dafi_summary_index.jsonl",
        "data/dts.jsonl": "data/dts_summary_index.jsonl",
        "data/gtc.jsonl": "data/gtc_summary_index.jsonl",
    }

    for input_file, output_file in files.items():
        label = os.path.basename(input_file).replace(".jsonl", "").upper()
        process_file(input_file, output_file, label)

