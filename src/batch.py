# batch.py

import csv
import logging
from llama_cpp import Llama
from src.utils import (
    extract_keywords_from_question,
    load_jsonl
)

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Keyword Overlap Retrieval ===
def find_top_chunks_by_keyword_overlap(keywords, chunks, top_k=1):
    ranked = []
    for entry in chunks:
        content = entry.get("text", "").lower()
        overlap = sum(1 for kw in keywords if kw in content)
        if overlap > 0:
            ranked.append((overlap, entry))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in ranked[:top_k]]

# === Rewrite Answer ===
def rewrite_answer(question, context, section, llm):
    prompt = (
        "You are an assistant for military members completing PCS travel.\n"
        "Rewrite the regulation context below into a clear answer to the user's question.\n"
        "Use plain language. Avoid regulation numbers. Give helpful bullet points if needed.\n\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Answer:"
    )
    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.5,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Error generating response: {e}"

# === Batch Processor ===
def process_prompts(input_file, output_file, llm, chunks):
    with open(input_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    results = []
    for prompt in prompts:
        try:
            logger.info(f"🔍 Processing: {prompt}")
            keywords = extract_keywords_from_question(prompt)
            top_chunks = find_top_chunks_by_keyword_overlap(keywords, chunks, top_k=1)

            if not top_chunks:
                logger.warning(f"⚠️ No match found for: {prompt}")
                results.append((prompt, "⚠️ No keyword match found."))
                continue

            top = top_chunks[0]
            answer = rewrite_answer(prompt, top["text"], top["metadata"].get("section", "N/A"), llm)
            results.append((prompt, answer))
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            results.append((prompt, "⚠️ Error during processing."))

    # Save to CSV
    with open(output_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Prompt", "Response"])
        writer.writerows(results)
    logger.info(f"✅ Results saved to {output_file}")

# === Run Batch Mode ===
def main():
    INPUT_FILE = "test_prompts.txt"
    OUTPUT_FILE = "hybrid_results.csv"

    logger.info("🚀 Loading regulation chunks...")
    chunks = (
        load_jsonl("data/jtr.jsonl") +
        load_jsonl("data/dafi.jsonl") +
        load_jsonl("data/dts.jsonl") +
        load_jsonl("data/gtc.jsonl")
    )
    logger.info(f"📚 Loaded {len(chunks)} chunks.")

    logger.info("🧠 Initializing TinyLLaMA...")
    llm = Llama(model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", n_ctx=2048, n_threads=4)

    process_prompts(INPUT_FILE, OUTPUT_FILE, llm, chunks)

if __name__ == "__main__":
    main()
