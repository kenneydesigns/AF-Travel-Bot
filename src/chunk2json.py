import os
import re
import json
from tqdm import tqdm
from src.utils import convert_pdf_to_text, split_into_windowed_chunks

# === SETTINGS ===
WORD_LIMIT = 500
OVERLAP_WORDS = 50

# === Helper: JTR Section Header ===
def is_jtr_section_header(line):
    line = line.strip()
    if len(line) >= 8 and line[:6].isdigit() and line.startswith("05") and line[6] in ['.', ' ']:
        section_id = line[:6]
        title = line[7:].strip()
        return True, section_id, title
    return False, None, None

# === JTR Parser ===
def process_jtr(input_txt, output_jsonl):
    chapter_map = {
        "05": "Chapter 5 – Permanent Duty Travel"
    }

    with open(input_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sections = []
    current = None
    matched_count = 0

    for line in lines:
        line = line.strip()
        is_header, section_id, title = is_jtr_section_header(line)

        if is_header:
            matched_count += 1
            if current:
                sections.append(current)

            chapter = chapter_map.get(section_id[:2], "Unknown")
            current = {
                "chunk_id": f"JTR_{section_id}",
                "section_title": title,
                "metadata": {
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
            total = len(chunks)
            for i, chunk_text in enumerate(chunks):
                chunk_entry = {
                    "chunk_id": f"{section['chunk_id']}_chunk{i}",
                    "section_title": section["section_title"],
                    "metadata": section["metadata"],
                    "chunk_index": i,
                    "text": chunk_text
                }
                f.write(json.dumps(chunk_entry) + "\n")

    print(f"✅ JTR: {matched_count} sections parsed and chunked into {output_jsonl}")


# === Helper: DAFI Section Header ===
def is_dafi_section_header(line):
    line = line.strip()
    if line.startswith("Chapter") and any(c.isdigit() for c in line):
        return True, "00", line
    if re.match(r"^\d{1,2}(\.\d+)+\s+", line):
        section_id = line.split()[0]
        title = line[len(section_id):].strip()
        return True, section_id, title
    return False, None, None

# === DAFI Parser ===
def process_dafi(input_txt, output_jsonl):
    chapter_map = {
        "01": "Chapter 1 – Overview",
        "02": "Chapter 2 – Managing the Leave Program",
        "03": "Chapter 3 – Chargeable Leave",
        "04": "Chapter 4 – Non-Chargeable Leave",
        "05": "Chapter 5 – Regular and Special Passes",
        "06": "Chapter 6 – Special Leave Accrual",
        "07": "Chapter 7 – Unique Leave Provisions",
        "08": "Chapter 8 – Post Deployment/Mobilization Respite Absence (PDMRA)"
    }

    with open(input_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sections = []
    current = None
    matched_count = 0

    for line in lines:
        line = line.strip()
        is_header, section_id, title = is_dafi_section_header(line)

        if is_header:
            matched_count += 1
            if current:
                sections.append(current)

            chapter = chapter_map.get(section_id.split('.')[0].zfill(2), "Unknown")
            current = {
                "chunk_id": f"DAFI_{section_id.replace('.', '_')}",
                "section_title": title,
                "metadata": {
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
            total = len(chunks)
            for i, chunk_text in enumerate(chunks):
                chunk_entry = {
                    "chunk_id": f"{section['chunk_id']}_chunk{i}",
                    "section_title": section["section_title"],
                    "metadata": section["metadata"],
                    "chunk_index": i,
                    "text": chunk_text
                }
                f.write(json.dumps(chunk_entry) + "\n")

    print(f"✅ DAFI: {matched_count} sections parsed and chunked into {output_jsonl}")

# === Helper: DTS Section Header ===
def is_dts_section_header(line):
    """
    Matches DTS headers like '030401. Authorizing Official (AO)'
    or top-level section numbers like '0305 DTS TRAVEL DOCUMENTS'
    """
    line = line.strip()
    match = re.match(r"^(03\d{2,4})\.\s+(.*)", line)
    if match:
        return True, match.group(1), match.group(2).strip()
    
    match_alt = re.match(r"^(030[0-9])\s+(.*)", line)
    if match_alt:
        return True, match_alt.group(1), match_alt.group(2).strip()
    
    return False, None, None

# === DTS Parser ===
def process_dts(input_txt, output_jsonl):
    print("📘 Processing DTS Regulations...")
    with open(input_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sections = []
    current = None
    matched_count = 0

    for line in lines:
        line = line.strip()
        is_header, section_id, title = is_dts_section_header(line)

        if is_header:
            matched_count += 1
            if current:
                sections.append(current)

            current = {
                "chunk_id": f"DTS_{section_id}",
                "section_title": title,
                "metadata": {
                    "chapter": "DTS PCS",
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
                f.write(json.dumps({
                    "chunk_id": f"{section['chunk_id']}_chunk{i}",
                    "section_title": section["section_title"],
                    "metadata": section["metadata"],
                    "chunk_index": i,
                    "text": chunk_text
                }) + "\n")

    print(f"✅ DTS: {matched_count} sections parsed and chunked into {output_jsonl}")

# === Helper: GTC Section Header ===
def is_gtc_section_header(line):
    """
    Matches headers like '041401.' and captures the title from the next line.
    """
    line = line.strip()
    match = re.match(r"^(04\d{4})\.\s*$", line)  # Matches '041401.' exactly
    if match:
        return True, match.group(1)
    return False, None

# === GTC Parser ===
def process_gtc(input_txt, output_jsonl):
    print("📘 Processing GTCC Regulations...")
    with open(input_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sections = []
    current = None
    matched_count = 0
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        is_header, section_id = is_gtc_section_header(line)
        if is_header and (i + 1 < len(lines)):
            matched_count += 1

            # Save previous section if exists
            if current:
                sections.append(current)

            title = lines[i + 1].strip()
            current = {
                "chunk_id": f"GTC_{section_id}",
                "section_title": title,
                "metadata": {
                    "chapter": "GTC PCS",
                    "section": section_id
                },
                "content": ""
            }
            i += 2  # Skip title line
            continue

        if current:
            current["content"] += line + " "

        i += 1

    if current:
        sections.append(current)

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for section in sections:
            chunks = split_into_windowed_chunks(section["content"], WORD_LIMIT, OVERLAP_WORDS)
            for i, chunk_text in enumerate(chunks):
                f.write(json.dumps({
                    "chunk_id": f"{section['chunk_id']}_chunk{i}",
                    "section_title": section["section_title"],
                    "metadata": section["metadata"],
                    "chunk_index": i,
                    "text": chunk_text
                }) + "\n")

    print(f"✅ GTC: {matched_count} real sections parsed and chunked into {output_jsonl}")

# === Main Execution ===
if __name__ == "__main__":
    convert_pdf_to_text("data/pdfs/jtr.pdf", "data/txts/jtr.txt")
    process_jtr("data/txts/jtr.txt", "data/jtr.jsonl")

    convert_pdf_to_text("data/pdfs/dafi36-3003.pdf", "data/txts/dafi36-3003.txt")
    process_dafi("data/txts/dafi36-3003.txt", "data/dafi.jsonl")

    convert_pdf_to_text("data/pdfs/dts.pdf", "data/txts/dts.txt")
    process_dts("data/txts/dts.txt", "data/dts.jsonl")

    convert_pdf_to_text("data/pdfs/gtc.pdf", "data/txts/gtc.txt")
    process_gtc("data/txts/gtc.txt", "data/gtc.jsonl")
