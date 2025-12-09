import logging
import os
import re
import textwrap
import requests
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Any, Optional, Set, Tuple
from sentence_transformers import SentenceTransformer
import json
import time 
import math
from services.azure_llm import AzureOpenAIClient
from services.prompt_builder import build_llm_prompt_for_chunk_extraction
from utility.helpers import load_local_model


def cosine_similarity(a, b) -> float:
    a = np.array(a) / np.linalg.norm(a)
    b = np.array(b) / np.linalg.norm(b)
    return float(np.dot(a, b))



MISSING_SENTINELS = {"", "nan", "none", "null"}

def normalize(value: Any) -> Optional[str]:
    """
    Missing-safe normalization:
    - Treat None, float NaN, 'nan', 'none', 'null', '' as missing (return None)
    - Lowercase, strip, collapse whitespace, remove punctuation (keep spaces)
    Returns:
        Normalized string or None if missing sentinel.
    """
    if value is None:
        return None
    try:
        # Handle pandas NaN or any float NaN
        if isinstance(value, float) and math.isnan(value):
            return None
    except Exception:
        pass

    s = str(value).strip().lower()
    if s in MISSING_SENTINELS:
        return None

    s = re.sub(r"\s+", " ", s)       # collapse multiple spaces to one space
    s = re.sub(r"[^\w\s]", "", s)    # remove punctuation, keep spaces
    return s

def normalize_compact(value: Any) -> Optional[str]:
    """
    Compact normalization: same as normalize() but removes spaces.
    Use sparingly (e.g., for substring checks), not for logging/debug.
    """
    s = normalize(value)
    if s is None:
        return None
    return re.sub(r"\s+", "", s)

def intersects_pages(start: Optional[int], end: Optional[int], expected_pages: Optional[Set[int]]) -> bool:
    """
    Returns True if the chunk's page range intersects with expected_pages.
    If expected_pages is None or start/end are missing, returns True (no guardrail).
    """
    if not expected_pages or start is None or end is None:
        return True
    try:
        return any(p in expected_pages for p in range(int(start), int(end) + 1))
    except Exception:
        return True  # be permissive if values are malformed

# Optional: allowlist of subsections per section (normalized strings)
SECTION_ALLOWED_SUBSECTIONS: Dict[str, Set[Optional[str]]] = {
    # For auditor's report: only FULL or None should be accepted
    "report of independent registered public accounting firm": {"full", None},
    # Add more sections if needed
}

def allowed_subsection_for_section(section_norm: Optional[str], sub_norm: Optional[str]) -> bool:
    if section_norm is None:
        return True
    allowed = SECTION_ALLOWED_SUBSECTIONS.get(section_norm)
    if not allowed:
        return True
    return sub_norm in allowed



def match_subsection_regex(target_subsection: str, chunks: list, section: str = None):
    if not target_subsection:
        return []
    target_norm = normalize(target_subsection)
    matched = []
    for chunk in chunks:
        sec_norm = normalize(chunk.get("section", ""))
        sub_norm = normalize(chunk.get("subsection", ""))
        if section and sec_norm != normalize(section):
            continue
        if target_norm in sub_norm:   # normalized substring match
            matched.append(chunk)
    logging.info("Regex subsection match found %d chunks (target='%s')", len(matched), target_norm)
    return matched

def score_and_return_top_k(query_vector, chunk_list, k: int, context: str = ""):
    scored = []
    for chunk in chunk_list:
        if "embedding_vector" not in chunk or chunk["embedding_vector"] is None:
            continue
        score = cosine_similarity(query_vector, chunk["embedding_vector"])
        chunk_with_score = chunk.copy()
        chunk_with_score["similarity"] = score
        scored.append(chunk_with_score)

    top_k = sorted(scored, key=lambda x: x["similarity"], reverse=True)[:k]

    logging.info("Top %d chunks selected (%s):", len(top_k), context)
    for i, c in enumerate(top_k, 1):
        logging.info(
            "\n--- Chunk %d ---\n"
            "Section    : %s (norm='%s')\n"
            "Subsection : %s (norm='%s')\n"
            "Similarity : %.4f\n"
            "Page Range : %s-%s\n"
            "Char Count : %s\n"
            "Source     : %s\n"
            "Text       :\n%s\n",
            i,
            c.get("section", ""),
            normalize(c.get("section", "")),
            c.get("subsection", ""),
            normalize(c.get("subsection", "")),
            c.get("similarity", 0.0),
            c.get("pdf_start_page", ""),
            c.get("pdf_end_page", ""),
            c.get("char_count", ""),
            c.get("source", ""),
            c.get("text", "").strip()
        )
    return top_k


def get_chunks_with_fallback(
    question: str,
    chunks: list,
    k: int = 10,
    model_path: str = r"C:\Users\2000166072\Documents\DRHP Prospectus\dhrp2\Embedding_model",  
    section: str = None,
    subsection: str = None
) -> Tuple[List[Dict[str, Any]], str, str, int]:
    """
    Returns top-k chunks plus section/subsection info and raw match count.
    Uses a local SentenceTransformer model by default.
    """

    # Load local model with pooling dimension
    model = load_local_model(model_path, dim=768)
    logging.info("Embedding dimension: %d", model.get_sentence_embedding_dimension())

    # Encode query
    query_vector = model.encode([question])[0]

    section_norm = normalize(section) if section else None
    subsection_norm = normalize(subsection) if subsection else None

    logging.info("Normalized CSV section='%s', subsection='%s'", section_norm, subsection_norm)

    # 1️⃣ Exact normalized match
    exact_filtered = [
        c for c in chunks
        if (section_norm is None or normalize(c.get("section", "")) == section_norm)
        and (subsection_norm is None or normalize(c.get("subsection", "")) == subsection_norm)
    ]
    logging.info("Exact filter matched %d chunks", len(exact_filtered))
    if exact_filtered:
        top_k = score_and_return_top_k(query_vector, exact_filtered, k, context="section+subsection exact")
        return top_k, section or "", subsection or "", len(exact_filtered)

    # 2️⃣ Regex/substring subsection match
    if subsection_norm:
        regex_filtered = [
            c for c in chunks
            if (section_norm is None or normalize(c.get("section", "")) == section_norm)
            and subsection_norm in normalize(c.get("subsection", ""))
        ]
        logging.info("Regex filter matched %d chunks", len(regex_filtered))
        if regex_filtered:
            top_k = score_and_return_top_k(query_vector, regex_filtered, k, context="section+subsection regex")
            return top_k, section or "", subsection or "", len(regex_filtered)

    # 3️⃣ Section-only fallback
    if section_norm:
        section_filtered = [c for c in chunks if normalize(c.get("section", "")) == section_norm]
        logging.info("Section-only fallback matched %d chunks", len(section_filtered))
        if section_filtered:
            top_k = score_and_return_top_k(query_vector, section_filtered, k, context="section-only fallback")
            return top_k, section, subsection, len(section_filtered)

    logging.warning(f"No chunks matched for question='{question}' (section={section}, subsection={subsection})")
    return [], section or "", subsection or "", 0

def _collect_gemini_api_keys() -> List[Tuple[str, str]]:
    keys = []

    # Base key first (highest priority)
    base_key = os.environ.get("GEMINI_API_KEY")
    if base_key:
        keys.append(("GEMINI_API_KEY", base_key))

    # Collect numbered keys, sorted by suffix number (1..N)
    numbered = []
    for k, v in os.environ.items():
        if k.startswith("GEMINI_API_KEY") and k != "GEMINI_API_KEY":
            suffix = k[len("GEMINI_API_KEY"):]
            if suffix.isdigit():
                numbered.append((int(suffix), k, v))
    numbered.sort(key=lambda x: x[0])

    keys.extend((name, val) for _, name, val in numbered)

    return keys


def evaluate_with_gemini(
    prompt: str,
    cache_path: str,
    max_retries: int = 3,
    backoff_factor: int = 2,
    request_timeout: int = 30
) -> dict:
    
    api_keys = _collect_gemini_api_keys()
    if not api_keys:
        logging.error("No Gemini API keys found in environment. Set GEMINI_API_KEY or GEMINI_API_KEY1..N.")
        return {"answer": "[API Error: missing GEMINI_API_KEY(s)]", "reasoning_steps": [], "validation_steps": []}

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-live:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    summary_filename = os.path.basename(cache_path)

    # Try each key in priority order
    for idx, (env_name, api_key) in enumerate(api_keys, start=1):
        logging.info("Using Gemini API key #%d (%s)", idx, env_name)
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": api_key,
        }

        # Attempt up to max_retries with exponential backoff
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=request_timeout)
            except requests.RequestException as e:
                wait = (backoff_factor ** attempt) + (0.25 * (attempt + 1))
                logging.warning("Network/Request error with %s: %s (attempt %d/%d). Retrying in %.2fs...",
                                env_name, str(e), attempt + 1, max_retries, wait)
                time.sleep(wait)
                continue

            status = response.status_code

            # ✅ Success
            if status == 200:
                try:
                    data = response.json()
                    raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()

                    # Strip Markdown code block fencing if present
                    if raw.startswith("```json"):
                        raw = raw[len("```json"):].strip()
                    if raw.endswith("```"):
                        raw = raw[:-3].strip()

                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        f.write(raw)

                    print(f" Summary saved to: {summary_filename}")
                    return json.loads(raw)

                except Exception as e:
                    # If parsing error, return gracefully
                    logging.error("Failed to parse Gemini response with %s: %s\nRaw: %s",
                                  env_name, str(e), response.text[:500])
                    return {"answer": f"[JSON parsing error: {str(e)}]", "reasoning_steps": [], "validation_steps": []}

            # ✅ Retry on 429 or 5xx
            elif status == 429 or 500 <= status < 600:
                # If 429 (rate limit/quota), likely key-specific; retry a couple times then rotate to next key
                wait = (backoff_factor ** attempt) + (0.25 * (attempt + 1))  # small jitter
                logging.warning("Gemini API error %d with %s: %s. Retrying in %.2fs (attempt %d/%d)...",
                                status, env_name, response.text[:200], wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue

            # ❌ Non-retryable error (e.g., 400 bad request / auth issues)
            else:
                logging.error("Non-retryable Gemini API error %d with %s: %s",
                              status, env_name, response.text[:200])
                # Move to next key (if any)
                break

        # This key exhausted retries; rotate to next
        logging.info("Key %s exhausted retries or non-retryable error; switching to next if available.", env_name)

    # ❌ All keys failed
    logging.error("All Gemini API keys failed or exhausted.")
    return {"answer": "[API Error: all API keys exhausted]", "reasoning_steps": [], "validation_steps": []}


def process_csv_and_evaluate(
    csv_path: str,
    output_path: str,
    chunks: list,
    model_path: str = r"C:\Users\2000166072\Documents\DRHP Prospectus\dhrp2\Embedding_model",  
    base: str = None,
    provider: str = "azure"   # choose "gemini" or "azure"
) -> pd.DataFrame:
    """
    Process a CSV of questions, evaluate with Gemini or Azure, and save results.
    Produces:
      - step1 JSON: {data_field: answer}
      - step2 JSON: {data_field: {answer, reasoning_steps, validation_steps}}
      - answered CSV with 'Answer' column
    """

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    step1_output, step2_output, results = {}, {}, []

    if base is None:
        base = os.path.splitext(os.path.basename(csv_path))[0]

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    outputs_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    step1_path = os.path.join(outputs_dir, f"{base}_step1_answers.json")
    step2_path = os.path.join(outputs_dir, f"{base}_step2_reasoning.json")

    # Load local model oncemodel = load_local_model(model_path, dim=768)
    model = load_local_model(model_path, dim=768)
    logging.info("Embedding dimension: %d", model.get_sentence_embedding_dimension())

    for idx, row in df.iterrows():
        data_field = str(row.get("Data Fields", "")).strip()
        particulars = str(row.get("Particulars", "")).strip()
        section = str(row.get("Heading", "")).strip()
        subsection = str(row.get("Sub- Heading", "")).strip()

        user_question = particulars if particulars and particulars.upper() != "N/A" else data_field

        print(f"\n Processing row {idx+1}: {data_field} - {particulars[:50]}...")

        # Use local model path in chunk retrieval
        top_k_chunks, final_section, final_subsection, match_count = get_chunks_with_fallback(
            user_question, chunks, k=10, model_path=model_path,
            section=section, subsection=subsection
        )

        logging.info(
            "📊 Question %d: '%s'\n"
            "   Total chunks available: %d\n"
            "   Section/subsection matched before scoring: %d\n"
            "   Top-K chunks selected: %d\n"
            "   Section: %s | Subsection: %s",
            idx+1, user_question, len(chunks),
            match_count, len(top_k_chunks),
            final_section, final_subsection
        )

        prompt = build_llm_prompt_for_chunk_extraction(data_field, particulars, top_k_chunks)

        if provider == "azure":
            cache_path = f"cache/azure_summary_row_{idx+1}.txt"
            azure_client = AzureOpenAIClient()
            summary = azure_client.evaluate(prompt, cache_path)
            parsed = parse_gemini_response(summary)
        else:
            cache_path = f"cache/gemini_summary_row_{idx+1}.txt"
            summary = evaluate_with_gemini(prompt, cache_path)
            parsed = parse_gemini_response(summary)

        if not parsed.get("validation_steps"):
            parsed["validation_steps"] = ["No validation steps returned by model"]

        step1_output[data_field] = parsed["answer"]
        step2_output[data_field] = {
            "answer": parsed["answer"],
            "reasoning_steps": parsed.get("reasoning_steps", []),
            "validation_steps": parsed.get("validation_steps", [])
        }
        results.append(parsed)

        with open(step1_path, "w", encoding="utf-8") as f1:
            json.dump(step1_output, f1, indent=2)
        with open(step2_path, "w", encoding="utf-8") as f2:
            json.dump(step2_output, f2, indent=2)

    df["Answer"] = [p["answer"] for p in results]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✅ All rows processed. Outputs saved to:\n- {output_path}\n- {step1_path}\n- {step2_path}")
    return df


def parse_gemini_response(summary: str) -> Dict[str, Any]:
    """
    Parse Gemini's JSON response safely.
    Accepts raw string or dict, extracts 'answer' and optional steps.
    """
    data = {}
    if isinstance(summary, str):
        try:
            data = json.loads(summary)
        except Exception as e:
            logging.error(f"Failed to parse Gemini response: {e}")
            return {"answer": "", "reasoning_steps": [], "validation_steps": ["Parse error"]}
    elif isinstance(summary, dict):
        data = summary

    answer = str(data.get("answer", "")).strip()
    if answer.lower().startswith("answer:"):
        answer = answer[len("answer:"):].strip()

    reasoning = data.get("reasoning_steps", [])
    validation = data.get("validation_steps", [])
    if not validation:
        validation = ["No validation steps returned by model"]

    return {
        "answer": answer,
        "reasoning_steps": reasoning,
        "validation_steps": validation
    }
