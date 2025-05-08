import os
import json
import re
import numpy as np
import fitz  # PyMuPDF
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATA_DIR = "app/data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")

model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_metadata(text, filename):
    # Try to extract case title and date from text; fallback to filename
    title_match = re.search(r"(.*?) v\. (.*?)\n", text, re.IGNORECASE)
    date_match = re.search(r"(\d{1,2} \w+ \d{4})", text)
    title = title_match.group(0).strip() if title_match else filename
    date = date_match.group(0) if date_match else "Unknown"
    return title, date


def chunk_text(text, max_length=500):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""
    seen = set()
    for para in paragraphs:
        if para in seen:
            continue  # skip duplicate paragraphs
        seen.add(para)
        if len(current_chunk) + len(para) < max_length:
            current_chunk += " " + para
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def index_pdfs():
    all_chunks = []
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        path = os.path.join(PDF_DIR, filename)
        text = extract_text_from_pdf(path)
        title, date = extract_metadata(text, filename)
        chunks = chunk_text(text)
        for c in chunks:
            all_chunks.append(
                {
                    "filename": filename,
                    "case_title": title,
                    "judgment_date": date,
                    "text": c,
                }
            )

    print(f"Extracted {len(all_chunks)} total chunks. Generating embeddings...")

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "w") as f:
        json.dump(all_chunks, f)

    print(f"✅ Indexed {len(all_chunks)} chunks from {len(pdf_files)} PDFs.")


if __name__ == "__main__":
    index_pdfs()
