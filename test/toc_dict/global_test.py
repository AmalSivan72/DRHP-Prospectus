
import json
import difflib
import unicodedata
import re
from html import unescape
from typing import Dict, List, Tuple, Optional, Any

# --- Sentence embeddings ---
from sentence_transformers import SentenceTransformer, util

from utility.helpers import load_local_model
#C:\Users\2000166072\Documents\DRHP Prospectus\dhrp2\utils\helpers.py
#utils\helpers.py



# --- Normalization ---
def normalize(text: str) -> str:
    """
    Normalize text for matching:
    - HTML unescape
    - Unicode NFKC
    - lowercase
    - trim
    - remove punctuation except '&'
    - collapse whitespace
    """
    if text is None:
        return ""
    text = unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    # Keep '&' only (fix for previous pattern "[^\\w\\s&amp;]")
    text = re.sub(r"[^\w\s&]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Fuzzy match ---
def fuzzy_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


# --- Embedding similarity (Sentence Transformers cosine) ---
def encode_sentences(texts: List[str], model: SentenceTransformer):
    # Normalize before encoding to reduce noise
    texts_norm = [normalize(t) for t in texts]
    # Normalize embeddings improves cosine stability
    return model.encode(texts_norm, normalize_embeddings=True)

def semantic_similarity_embeddings(
    a: str,
    b_list: List[str],
    model: SentenceTransformer,
    precomputed_b: Optional[Any] = None
) -> List[float]:
    """
    Compute cosine similarity between 'a' and list of 'b' using sentence embeddings.
    If precomputed_b is provided (Embeddings tensor/array), reuse it for speed.
    """
    a_emb = encode_sentences([a], model)
    if precomputed_b is None:
        b_emb = encode_sentences(b_list, model)
    else:
        b_emb = precomputed_b
    sims = util.cos_sim(a_emb, b_emb).cpu().numpy().ravel().tolist()
    return sims


# --- Prepare taxonomy embeddings once ---
def prepare_taxonomy_embeddings(
    global_taxonomy: Dict[str, List[str]],
    model: SentenceTransformer
) -> Dict[str, Dict[str, Any]]:
    """
    Returns {key: {"examples": original_texts, "embeddings": np.ndarray, "norm_texts": normalized}}
    """
    prepared = {}
    for key, examples in global_taxonomy.items():
        norm_examples = [normalize(e) for e in examples]
        emb = model.encode(norm_examples, normalize_embeddings=True)
        prepared[key] = {
            "examples": examples,
            "norm_texts": norm_examples,
            "embeddings": emb
        }
    return prepared


# --- Match subsections (embeddings + optional fuzzy) ---
def match_subsections(
    input_subsections: List[str],
    global_taxonomy: Dict[str, List[str]],
    *,
    threshold: float = 0.7,
    alpha: float = 0.85,   # favor semantic heavily
    return_details: bool = False,
    model: Optional[SentenceTransformer] = None,
    prepared_taxonomy: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Match each input subsection to taxonomy keys using sentence embeddings.
    Scoring:
      combined = alpha * semantic + (1 - alpha) * fuzzy
    Args:
      threshold: minimum combined score to accept a match (typical 0.65–0.8).
      alpha: weight for semantic vs fuzzy similarity.
      model: preloaded SentenceTransformer model (e.g., all-MiniLM-L6-v2).
      prepared_taxonomy: optional precomputed embeddings for taxonomy (from prepare_taxonomy_embeddings).
    """
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    # Precompute taxonomy embeddings when not supplied
    if prepared_taxonomy is None:
        prepared_taxonomy = prepare_taxonomy_embeddings(global_taxonomy, model)

    results: Dict[str, Any] = {}
    for raw_local in input_subsections:
        local_norm = normalize(raw_local)

        best_key = None
        best_example = None
        best_combined = -1.0
        best_sem = 0.0
        best_fuzzy = 0.0

        for key, pack in prepared_taxonomy.items():
            examples = pack["examples"]
            norm_examples = pack["norm_texts"]
            b_emb = pack["embeddings"]

            # Semantic: cosine between local and each candidate example
            sem_scores = semantic_similarity_embeddings(local_norm, norm_examples, model, precomputed_b=b_emb)
            max_sem = max(sem_scores)

            # Fuzzy: char-level ratio
            fuzzy_scores = [fuzzy_ratio(local_norm, ne) for ne in norm_examples]
            max_fuzzy = max(fuzzy_scores)

            # Combined score
            combined_per_example = [
                alpha * s + (1 - alpha) * f
                for s, f in zip(sem_scores, fuzzy_scores)
            ]
            idx = max(range(len(combined_per_example)), key=lambda i: combined_per_example[i])

            combined = combined_per_example[idx]

            if combined > best_combined:
                best_combined = combined
                best_key = key
                best_example = examples[idx]
                best_sem = sem_scores[idx]
                best_fuzzy = fuzzy_scores[idx]

        matched_key = best_key if best_combined >= threshold else "unmapped"

        if return_details:
            results[raw_local] = {
                "matched_key": matched_key,
                "combined_score": round(best_combined, 4),
                "semantic_score": round(best_sem, 4),
                "fuzzy_score": round(best_fuzzy, 4),
                "best_example": best_example,
                "threshold": threshold,
                "alpha": alpha,
            }
        else:
            results[raw_local] = matched_key

    return results


# --- Example usage ---
if __name__ == "__main__":
    global_taxonomy = {
        "special_tax_benefits": [
            "STATEMENT OF POSSIBLE SPECIAL TAX BENEFITS",
            "STATEMENT ON SPECIAL TAX BENEFITS",
            "STATEMENT OF SPECIAL TAX BENEFITS"
        ],
        "conventions_financial_industry_market_data_currency": [
            "CERTAIN CONVENTIONS, PRESENTATION OF FINANCIAL, INDUSTRY AND MARKET DATA AND CURRENCY OF PRESENTATION",
            "CERTAIN CONVENTIONS, PRESENTATION OF FINANCIAL, INDUSTRY AND MARKET DATA",
            "CERTAIN CONVENTIONS, PRESENTATION OF FINANCIAL, INDUSTRY AND MARKET DATA AND CURRENCY OF PRESENTATION"
        ],
        "summary_financial_information": [
            "SUMMARY OF FINANCIAL INFORMATION",
            "SUMMARY OF RESTATED CONSOLIDATED FINANCIAL INFORMATION"
        ],
    }

    input_subsections = [
        "STATEMENT OF POSSIBLE SPECIAL TAX BENEFITS",
        "CERTAIN CONVENTIONS, USE OF FINANCIAL INFORMATION AND MARKET DATA AND CURRENCY OF PRESENTATION",
        "SUMMARY FINANCIAL STATEMENTS"
    ]

    model_path = r"C:\Users\2000166072\Documents\DRHP Prospectus\dhrp2\Embedding_model"
    model = load_local_model(model_path, dim=768)
    prepared = prepare_taxonomy_embeddings(global_taxonomy, model)

    result = match_subsections(
        input_subsections,
        global_taxonomy,
        threshold=0.68,    # slightly lenient for paraphrases
        alpha=0.85,        # rely mostly on semantic
        return_details=True,
        model=model,
        prepared_taxonomy=prepared
    )
    print(json.dumps(result, indent=2))

