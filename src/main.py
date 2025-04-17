from src.utils import load_jsonl
from src.api_interface import choose_best_source
from src.summarizer import summarize_chunk

def main():
    question = input("Ask a travel question: ")

    # Load summaries from metadata (simulate preview of jsonl data)
    data = load_jsonl("data/summary_index.jsonl")
    summaries = [d["summary"] for d in data]

    chosen_index = int(choose_best_source(question, summaries)) - 1
    chosen_data = data[chosen_index]

    full_chunk = chosen_data["content"]
    summary = summarize_chunk(full_chunk)

    print("\n📘 Summary:\n", summary)

if __name__ == "__main__":
    main()
