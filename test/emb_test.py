import pickle
import os
import re
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def load_chunks(pkl_path: str) -> List[Dict[str, any]]:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"❌ Pickle file not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def match_subsection_regex(target_subsection: str, chunks: List[Dict[str, any]], section: str) -> List[Dict[str, any]]:
    pattern = re.compile(re.escape(target_subsection.strip()), re.IGNORECASE)
    return [
        chunk for chunk in chunks
        if chunk.get("section", "").strip().upper() == section.strip().upper()
        and pattern.search(chunk.get("subsection", ""))
    ]


def score_subsection_semantically(
    target_subsection: str,
    chunks: List[Dict[str, any]],
    section: str,
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 5
) -> List[Dict[str, any]]:
    model = SentenceTransformer(model_name)
    query_vec = model.encode([target_subsection])[0]

    filtered = [
        chunk for chunk in chunks
        if chunk.get("section", "").strip().upper() == section.strip().upper()
        and "embedding_vector" in chunk
    ]

    scored = []
    for chunk in filtered:
        score = cosine_similarity([query_vec], [chunk["embedding_vector"]])[0][0]
        chunk_with_score = chunk.copy()
        chunk_with_score["similarity"] = score
        scored.append(chunk_with_score)

    return sorted(scored, key=lambda x: x["similarity"], reverse=True)[:top_k]


# 🔧 Example usage
if __name__ == "__main__":
    pkl_path = r"C:\Users\2000166072\Desktop\dhrp2\pickles\pro_spdrdowjonesindustrialaverageetf_embedded.pkl"
    section = "Federal Income Taxes"
    target_subsection = "Transactions with Affiliates of the Trustee and Sponsor"

    chunks = load_chunks(pkl_path)

    print("\n🔍 Regex match:")
    regex_hits = match_subsection_regex(target_subsection, chunks, section)
    for c in regex_hits:
        print(f"📄 {c['chunk_id']} | {c['subsection']} | Pages {c['pdf_start_page']}–{c['pdf_end_page']}")

    print("\n🧠 Semantic match:")
    semantic_hits = score_subsection_semantically(target_subsection, chunks, section)
    for c in semantic_hits:
        print(f"📄 {c['chunk_id']} | {c['subsection']} | Score: {c['similarity']:.4f}")
