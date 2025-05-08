import streamlit as st
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import nltk

for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)


from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

# Setup paths
DATA_DIR = "app/data"
INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
summarizer = LexRankSummarizer()


# Load model + data
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_index_and_chunks():
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "r") as f:
        chunks = json.load(f)
    return index, chunks


model = load_model()
index, chunks = load_index_and_chunks()


def summarize(text, num_sentences=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(s) for s in summary)


# Search function
def search(query, initial_top_k=10, min_score=0.45, summarize_results=True):
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, initial_top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        score = 1 / (1 + dist)
        if score < min_score:
            continue
        item = chunks[idx]
        summary = summarize(item["text"]) if summarize_results else item["text"]
        results.append(
            {
                "filename": item["filename"],
                "case_title": item.get("case_title", ""),
                "judgment_date": item.get("judgment_date", ""),
                "similarity": float(score),
                "content": summary,
            }
        )
    return results


# Streamlit UI
st.title("⚖️ Legal Case Search Demo")
st.write(
    "Enter a legal query below to find similar past cases from your knowledge base."
)

query = st.text_input("🔍 Your query:", "")
show_summary = st.toggle("Show summary only", value=True)

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a query to search.")
    else:
        with st.spinner("Searching..."):
            results = search(query, summarize_results=show_summary)
        if results:
            st.success(f"Found {len(results)} matching results:")
            for r in results:
                st.markdown(
                    f"**📄 {r['case_title']}**  \n"
                    f"📅 {r['judgment_date']}  \n"
                    f"**Similarity:** {r['similarity']:.2f}  \n"
                    f"*Filename:* {r['filename']}"
                )
                st.write(r["content"])
                st.markdown("---")
        else:
            st.info("No relevant cases found. Try a different query.")
