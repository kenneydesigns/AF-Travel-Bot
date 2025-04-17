import json
import os
import textstat
import language_tool_python

def load_jsonl(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]

def save_jsonl(filepath, data):
    with open(filepath, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

from pdfminer.high_level import extract_text

def convert_pdf_to_text(pdf_path, txt_path):
    """
    Converts a PDF file to plain text using pdfminer if the txt file doesn't already exist.
    """
    if not os.path.exists(txt_path):
        print(f"📄 Converting {pdf_path} to {txt_path}...")
        text = extract_text(pdf_path)
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(f"✅ {txt_path} already exists. Skipping conversion.")

def split_into_windowed_chunks(text, max_words, overlap_words):
    """
    Splits the input text into overlapping word chunks.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + max_words]
        chunk_text = " ".join(chunk_words)
        if i + max_words < len(words):
            chunk_text += " …continued"
        chunks.append(chunk_text)
        i += max_words - overlap_words
    return chunks

tool = language_tool_python.LanguageTool('en-US')

def readability_score(text: str) -> float:
    """Flesch-Kincaid readability score scaled to 0–1"""
    score = textstat.flesch_reading_ease(text)
    return max(0.0, min(score / 100, 1.0))  # Normalize to 0-1

def grammar_error_score(text: str) -> float:
    """Returns score 1.0 = no grammar issues, 0.0 = major issues"""
    matches = tool.check(text)
    errors = len(matches)
    if errors == 0:
        return 1.0
    elif errors < 3:
        return 0.7
    elif errors < 5:
        return 0.4
    else:
        return 0.2

def overall_confidence(text: str) -> float:
    if not text.strip():
        return 0.0
    read = readability_score(text)
    grammar = grammar_error_score(text)
    return round((read + grammar) / 2, 2)

def keyword_coverage(source, summary):
    key_terms = ["transportation", "allowance", "dependent", "PCS", "government", "mileage"]
    hits = [kw for kw in key_terms if kw in summary.lower()]
    return len(hits) / len(key_terms)

def evaluate_summary(summary: str, source: str = "") -> dict:
    return {
        "readability": readability_score(summary),
        "grammar": grammar_error_score(summary),
        "keyword_coverage": keyword_coverage(source, summary) if source else None,
        "confidence": overall_confidence(summary),
    }
