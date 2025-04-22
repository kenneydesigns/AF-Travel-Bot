# === Imports ===
import os
import re
import json
import openai
import numpy as np
import torch
import textstat
import language_tool_python
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text
from llama_cpp import Llama

# === Setup for Tools ===
tool = language_tool_python.LanguageTool('en-US')
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
openai.api_key = "hf_placeholder"  
openai.api_base = "http://localhost:8080/v1"
model_name = "microsoft/phi-1_5" 
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # device -1 = CPU

# === FILE I/O UTILITIES ===
# Used by: ALL scripts
def load_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

def convert_pdf_to_text(pdf_path, txt_path):
    """Convert PDF to plain text if not already done."""
    if not os.path.exists(txt_path):
        print(f"📄 Converting {pdf_path} → {txt_path}")
        text = extract_text(pdf_path)
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(f"✅ Skipping existing: {txt_path}")


# === TEXT CHUNKING ===
# Used by: process_data.py
def split_into_windowed_chunks(text, max_words, overlap_words):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_words]
        chunk_text = " ".join(chunk)
        if i + max_words < len(words):
            chunk_text += " …continued"
        chunks.append(chunk_text)
        i += max_words - overlap_words
    return chunks


# === KEYWORD EXTRACTION ===
# Used by: main.py, process_data.py
COMMON_STOPWORDS = {
    "chapter", "service", "authorized", "may", "section", "funds", "regulations",
    "accordance", "provided", "includes", "appendix", "use", "travel", "transportation",
    "pds", "dependent", "government", "member", "documents", "order", "expense"
}

def extract_keywords_from_question(question):
    words = re.findall(r"\b\w+\b", question.lower())
    return [w for w in words if w not in COMMON_STOPWORDS]

def extract_keywords(text, top_n=5):
    named_terms = re.findall(r"\b[A-Z]{2,}\b", text)
    return list(dict.fromkeys(named_terms))[:top_n]


# === GLOSSARY / ACRONYM UTILITIES ===
# Used by: process_data.py
def clean_definition(def_text):
    clean = def_text.strip().split("\n")[0]
    clean = re.sub(r"[^a-zA-Z0-9() ,\-]", "", clean)
    return clean[:120].strip().capitalize()

def extract_acronym_definitions(text, glossary):
    acronyms_found = re.findall(r"\b[A-Z]{2,}\b", text)
    definitions = []
    for acronym in set(acronyms_found):
        if re.search(rf"{acronym}\s*\(\s*[A-Z]{{2,}}\s*\)", text):
            continue
        match = re.search(rf"([A-Z][a-zA-Z\- ]{{8,}})\s*\(\s*{acronym}\s*\)", text)
        if match:
            definition = match.group(1).strip()
            if not definition.isupper() and len(definition.split()) >= 2:
                definitions.append({"acronym": acronym, "definition": definition})
                continue
        if acronym in glossary:
            definitions.append({"acronym": acronym, "definition": glossary[acronym]})
    return definitions

def safe_load_glossary(path="data/glossary_master.json"):
    if not os.path.exists(path):
        print(f"⚠️ Glossary file {path} not found.")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading glossary: {e}")
        return {}

ACRONYM_GLOSSARY = safe_load_glossary()


# === TEXT EVALUATION METRICS ===
# Used by: main.py, process_data.py
def readability_score(text):
    return max(0.0, min(textstat.flesch_reading_ease(text) / 100, 1.0))

def grammar_error_score(text):
    errors = len(tool.check(text))
    if errors == 0: return 1.0
    elif errors < 3: return 0.7
    elif errors < 5: return 0.4
    else: return 0.2

def overall_confidence(text, source=""):
    grammar = grammar_error_score(text)
    readability = readability_score(text)
    keyword = keyword_coverage(source, text)
    
    return round((0.4 * grammar + 0.3 * readability + 0.3 * keyword), 2)

def keyword_coverage(source, summary):
    key_terms = ["transportation", "allowance", "dependent", "PCS", "government", "mileage"]
    hits = [kw for kw in key_terms if kw in summary.lower()]
    return len(hits) / len(key_terms)

def evaluate_summary(summary, source=""):
    return {
        "readability": readability_score(summary),
        "grammar": grammar_error_score(summary),
        "keyword_coverage": keyword_coverage(source, summary) if source else None,
        "confidence": overall_confidence(summary),
    }
def rank_responses_with_llm(question, responses, llm):
    prompt = (
        f"You are helping a military member who asked this question:\n\n"
        f"\"{question}\"\n\n"
        f"You’ve written {len(responses)} possible answers. Choose the best one that:\n"
        f"- clearly explains the rule in plain English\n"
        f"- does not repeat regulation text\n"
        f"- is helpful and easy to follow\n\n"
    )

    for i, r in enumerate(responses, 1):
        prompt += f"Answer {i}:\n{r}\n\n"

    prompt += "Reply ONLY with the number of the best answer: 1, 2, or 3"

    try:
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.3,
        )
        raw_output = result["choices"][0]["message"]["content"].strip()
        match = re.search(r"[1-3]", raw_output)
        return responses[int(match.group()) - 1] if match else responses[0]
    except Exception as e:
        print(f"⚠️ Error during ranking: {e}")
        return responses[0]


# === MAIN.PY RESPONSE REWRITE UTILITY ===
# Used by: main.py
def generate_best_response(question, context, section, llm, top_n=3):
    """Run multiple answer rewrites using rewrite_answer() and return the best-scoring one."""
    from .utils import rewrite_answer, evaluate_summary

    candidates = []
    for _ in range(top_n):
        response = rewrite_answer(question, context, section, llm)
        score = evaluate_summary(response, source=context)
        candidates.append((score["confidence"], response, score))

    all_responses = [resp for _, resp, _ in candidates]
    best_answer = rank_responses_with_llm(question, all_responses, llm)
    best_score = next(score for _, resp, score in candidates if resp == best_answer)
    return best_answer, best_score, candidates


def rewrite_answer(question, context, section):
    prompt = (
        "You are a helpful assistant for military members moving PCS.\n"
        "Answer the user's question clearly using the regulation excerpt below.\n"
        "Use friendly, plain English. DO NOT repeat the regulation text or say 'Regulation Excerpt'.\n"
        "Assume the member is not civilian unless specified.\n"
        "Explain what the member should do, using simple language and bullets if needed.\n\n"
        f"Question: {question}\n"
        f"Regulation Excerpt: {context}\n\n"
        "Answer:"
    )

    try:
        result = text_gen(prompt, max_new_tokens=256, temperature=0.7, do_sample=True)
        return result[0]["generated_text"].split("Answer:")[-1].strip()
    except Exception as e:
        return f"⚠️ Error generating response: {e}"
     
# === PROCESS_DATA.PY SUMMARIZATION UTILITY ===
# Used by: process_data.py
def summarize_best_of_n(text, llm, attempts=3, threshold=None):
    threshold = threshold or {
        "readability": 0.5,
        "grammar": 0.7,
        "keyword_coverage": 0.4,
        "confidence": 0.65,
    }

    best_summary = ""
    best_score = {"readability": 0, "grammar": 0, "keyword_coverage": 0, "confidence": 0}
    for _ in range(attempts):
        summary = summarize_with_pipe(text, pipe)
        score = evaluate_summary(summary, text)

        if (
            score["confidence"] + (score.get("keyword_coverage") or 0)
            > best_score["confidence"] + (best_score.get("keyword_coverage") or 0)
        ):
            best_summary = summary
            best_score = score

        if all(score.get(k, 0) >= threshold[k] for k in threshold):
            break

    return best_summary, best_score


# === TINYLLAMA SUMMARIZATION CORE ===
# Used by: summarize_best_of_n()
def summarize_with_pipe(text, pipe):
    prompt = (
        "You are a helpful assistant. Summarize this military regulation text in plain English.\n"
        "Avoid repeating legal jargon. Be short and clear.\n\n"
        f"Text: {text.strip()}\n\nSummary:"
    )
    result = pipe(prompt, max_new_tokens=256, temperature=0.7, do_sample=True)
    return result[0]["generated_text"].split("Summary:")[-1].strip()

# === KEYWORD MATCHING + SIMILARITY ===
# Used by: evaluation tools
def dynamic_keyword_match_score(question, summary):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([question, summary])
        return (tfidf_matrix[0] @ tfidf_matrix[1].T).A[0][0]
    except:
        return 0.0


# === EMBEDDING SUPPORT FOR SEMANTIC SEARCH ===
# Used by: main.py, batch processing
def embed_text(text):
    return embed_model.encode([text])[0]

def find_most_similar(question, entries, top_k=3):
    q_vec = embed_text(question)
    scores = []
    for entry in entries:
        emb = np.array(entry.get("embedding", []))
        if emb.size == 0:
            continue
        sim = cosine_similarity([q_vec], [emb])[0][0]
        scores.append((sim, entry))
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:top_k]

def load_question_pool(filepath="data/question_pool.jsonl"):
    return load_jsonl(filepath)

def find_nearest_common_question(user_question, question_pool, threshold=0.75):
    user_vec = embed_text(user_question)
    best_match = None
    best_score = 0.0

    for q in question_pool:
        q_vec = embed_text(q["question"])
        score = cosine_similarity([user_vec], [q_vec])[0][0]
        if score > best_score and score >= threshold:
            best_score = score
            best_match = q

    return best_match, best_score
