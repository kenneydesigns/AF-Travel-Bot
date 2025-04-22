# main.py

import time
import argparse
from src.utils import (
    extract_keywords_from_question,
    load_jsonl,
    dynamic_keyword_match_score,
    embed_text,
    load_question_pool,
    find_nearest_common_question,
    evaluate_summary, generate_best_response
)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


# === SETTINGS ===
model_name = "microsoft/phi-1_5"   # or whatever model you prefer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# from llama_cpp import Llama

# === Keyword Overlap Retrieval ===
def find_top_chunks_by_keyword_overlap(keywords, chunks, top_k=3):
    ranked = []
    for entry in chunks:
        content = entry.get("text", "").lower()
        overlap = sum(1 for kw in keywords if kw in content)
        if overlap > 0:
            ranked.append((overlap, entry))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in ranked[:top_k]]

# === Rewrite Answer ===
def rewrite_answer(question, context, section, pipe):
    prompt = (
        "You are a helpful assistant for **military members** preparing for or completing a PCS move.\n"
        "Use the regulation excerpt below to answer the user's question.\n"
        "✅ Assume the user is **active duty military**, not civilian, unless explicitly stated.\n"
        "✅ Write in plain, friendly English.\n"
        "❌ Do NOT repeat the regulation text or use phrases like 'Regulation Excerpt'.\n"
        "✅ Focus on clear advice — what the member should do — and use bullet points if helpful.\n\n"
        f"Question: {question}\n"
        f"Regulation Excerpt: {context}\n\n"
        "Answer:"
    )

    result = pipe(prompt, max_new_tokens=256, temperature=0.7, do_sample=True)
    return result[0]["generated_text"].split("Answer:")[-1].strip()
    
# === MAIN CLI APP ===
def main():
    parser = argparse.ArgumentParser(description="AF TravelBot: PCS QA Helper")
    args = parser.parse_args()

    print("✈️ Welcome to AF TravelBot. Ask a PCS question below:")
    question = input("❓ ")

    start = time.time()
    keywords = extract_keywords_from_question(question)
    print(f"\n🔍 Extracted Keywords: {keywords}")

    # Try to match from question pool first
    question_pool = load_question_pool("data/question_pool.jsonl")
    match, score = find_nearest_common_question(question, question_pool, threshold=0.75)

    if match:
        print(f"\n✅ Matched known question (score {score:.2f})")
        print(f"📘 Source: {match['source']} | Section: {match['section']}")
        print(f"📝 {match['summary']}")
        print(f"\n⏱️ Done in {time.time() - start:.2f} seconds.")
        return

    # Load all chunks
    all_chunks = (
        load_jsonl("data/jtr.jsonl") +
        load_jsonl("data/dafi.jsonl") +
        load_jsonl("data/dts.jsonl") +
        load_jsonl("data/gtc.jsonl")
    )
    print(f"📚 Loaded {len(all_chunks)} total chunks.")

    # Match by keyword overlap
    top_chunks = find_top_chunks_by_keyword_overlap(keywords, all_chunks, top_k=3)
    if not top_chunks:
        print("⚠️ No matching chunk found with keywords.")
        return

    top = top_chunks[0]
    section = top["metadata"].get("section", "N/A")
    print(f"\n🎯 Top Match Section: {section}")
    print(f"📘 Title: {top.get('section_title', '')}")
    print(f"🧠 Content Preview:\n{top['text'][:300]}...")

    # Load TinyLLaMA
    # llm = Llama(model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", n_ctx=2048, n_threads=4)

    print("\n✍️ Generating response...")
    answer, best_score, all_scores = generate_best_response(question, top["text"], section, )


    print("\n💬 Final Answer:")
    print(f"📘 From {top['metadata'].get('source', '')}, Section {section}")
    print(answer)

    print("\n📊 All Attempts:")
    for i, (conf, resp, scores) in enumerate(all_scores, 1):
        print(f"\n🔁 Attempt {i} — Confidence: {conf:.2f}")
        print(resp)

    # === Evaluation ===
    print("\n📊 Response Evaluation:")
    print(f"📖 Readability:      {best_score['readability']:.2f}")
    print(f"📝 Grammar:          {best_score['grammar']:.2f}")
    if best_score['keyword_coverage'] is not None:
        print(f"🔑 Keyword Coverage: {best_score['keyword_coverage']:.2f}")
    print(f"✅ Confidence Score: {best_score['confidence']:.2f}")


    print(f"\n⏱️ Done in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    main()
