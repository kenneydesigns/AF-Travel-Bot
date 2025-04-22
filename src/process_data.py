# process_data.py

import os
import json
from src.utils import (
    convert_pdf_to_text,
    split_into_windowed_chunks,
    extract_keywords,
    extract_acronym_definitions,
    ACRONYM_GLOSSARY, summarize_best_of_n, summarize_with_pipe,
    embed_text  # Optional: for embeddings
)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# === SETTINGS ===
model_name = "microsoft/phi-1_5" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

WORD_LIMIT = 500
OVERLAP_WORDS = 50
DATA_DIR = "data"
# llm = Llama(model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", n_ctx=2048, n_threads=4)

# === Processing Function ===
def process_doc(source_name, pdf_path, txt_path, output_jsonl, section_header_fn, chapter_map):
    convert_pdf_to_text(pdf_path, txt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sections = []
    current = None
    matched_count = 0

    for line in lines:
        line = line.strip()
        is_header, section_id, title = section_header_fn(line)

        if is_header:
            matched_count += 1
            if current:
                sections.append(current)

            chapter = chapter_map.get(section_id.split('.')[0].zfill(2), "Unknown")
            current = {
                "chunk_id": f"{source_name.upper()}_{section_id.replace('.', '_')}",
                "section_title": title,
                "metadata": {
                    "source": source_name.upper(),
                    "chapter": chapter,
                    "section": section_id
                },
                "content": ""
            }
        elif current:
            current["content"] += line + " "

    if current:
        sections.append(current)

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for section in sections:
            chunks = split_into_windowed_chunks(section["content"], WORD_LIMIT, OVERLAP_WORDS)
            for i, chunk_text in enumerate(chunks):
                keywords_main = extract_keywords(chunk_text)
                acronym_defs = extract_acronym_definitions(chunk_text, ACRONYM_GLOSSARY)
                acronym_keywords = [item["acronym"] for item in acronym_defs]
                all_keywords = list(set(keywords_main + acronym_keywords))

                for item in acronym_defs:
                    ACRONYM_GLOSSARY[item["acronym"]] = item["definition"]
                summary, scores = summarize_best_of_n(chunk_text, )
                chunk_entry = {
                    "chunk_id": f"{section['chunk_id']}_chunk{i}",
                    "section_title": section["section_title"],
                    "metadata": section["metadata"],
                    "chunk_index": i,
                    "text": chunk_text,
                    "keywords": all_keywords,
                    "glossary": acronym_defs,
                    "summary": summary,
                    "metrics": scores
                }
                f.write(json.dumps(chunk_entry) + "\n")

    print(f"✅ {source_name.upper()}: {matched_count} sections parsed into {output_jsonl}")

# === Header Matchers ===

def match_jtr(line):
    if len(line) >= 8 and line[:6].isdigit() and line.startswith("05") and line[6] in ['.', ' ']:
        return True, line[:6], line[7:].strip()
    return False, None, None

def match_dafi(line):
    if line.startswith("Chapter") and any(c.isdigit() for c in line):
        return True, "00", line
    if "." in line and re.match(r"^\d{1,2}(\.\d+)+\s+", line):
        section_id = line.split()[0]
        title = line[len(section_id):].strip()
        return True, section_id, title
    return False, None, None

def match_dts(line):
    match = re.match(r"^(03\d{2,4})\.\s+(.*)", line)
    if match:
        return True, match.group(1), match.group(2).strip()
    match_alt = re.match(r"^(030[0-9])\s+(.*)", line)
    if match_alt:
        return True, match_alt.group(1), match_alt.group(2).strip()
    return False, None, None

def match_gtc(line):
    match = re.match(r"^(04\d{4})\.\s*$", line.strip())
    return (True, match.group(1)) if match else (False, None)

# === Chapter Maps ===

CHAPTERS_DAFI = {
    "01": "Chapter 1 – Overview",
    "02": "Chapter 2 – Managing the Leave Program",
    "03": "Chapter 3 – Chargeable Leave",
    "04": "Chapter 4 – Non-Chargeable Leave",
    "05": "Chapter 5 – Regular and Special Passes",
    "06": "Chapter 6 – Special Leave Accrual",
    "07": "Chapter 7 – Unique Leave Provisions",
    "08": "Chapter 8 – Post Deployment/Mobilization Respite Absence (PDMRA)"
}

CHAPTERS_JTR = {
    "05": "Chapter 5 – Permanent Duty Travel"
}

# === Run All Documents ===

if __name__ == "__main__":
    process_doc(
        "jtr",
        f"{DATA_DIR}/pdfs/jtr.pdf",
        f"{DATA_DIR}/txts/jtr.txt",
        f"{DATA_DIR}/jtr.jsonl",
        match_jtr,
        CHAPTERS_JTR
    )

    process_doc(
        "dafi",
        f"{DATA_DIR}/pdfs/dafi36-3003.pdf",
        f"{DATA_DIR}/txts/dafi36-3003.txt",
        f"{DATA_DIR}/dafi.jsonl",
        match_dafi,
        CHAPTERS_DAFI
    )

    process_doc(
        "dts",
        f"{DATA_DIR}/pdfs/dts.pdf",
        f"{DATA_DIR}/txts/dts.txt",
        f"{DATA_DIR}/dts.jsonl",
        match_dts,
        {"030": "DTS PCS"}
    )

    process_doc(
        "gtc",
        f"{DATA_DIR}/pdfs/gtc.pdf",
        f"{DATA_DIR}/txts/gtc.txt",
        f"{DATA_DIR}/gtc.jsonl",
        match_gtc,
        {"04": "GTC PCS"}
    )
