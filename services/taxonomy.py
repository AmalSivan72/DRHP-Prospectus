
# taxonomy.py

import re
import json
import difflib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import os

from utility.helpers import load_local_model



model_path= r"C:\Users\2000166072\Documents\DRHP Prospectus\dhrp2\Embedding_model"
embedding_model=load_local_model(model_path, dim=768)

# Optional TF-IDF (semantic-ish) matcher
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

_SECTION_PREFIX_RE = re.compile(
    r"^\s*SECTION\s*(?:[:\-–—]\s*)?(?P<roman>[IVXLCDM]+)\s*(?:[:\-–—]\s*)?",
    re.IGNORECASE,
)

def normalize_text(s: str) -> str:
    s = (s or "").strip().replace("\u200b", "").replace("\u2026", "...")
    s = re.sub(r"\s+", " ", s).lower()
    return s

def strip_section_prefix(title: str) -> str:
    t = normalize_text(title)
    m = _SECTION_PREFIX_RE.match(t)
    if m:
        t = t[m.end():].strip()
    return t

def normalize_title_for_match(title: str) -> str:
    t = strip_section_prefix(title)
    t = re.sub(r"[:\-–—]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def lexical_similarity(a: str, b: str) -> float:
    na = normalize_title_for_match(a)
    nb = normalize_title_for_match(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.95
    if na.startswith(nb) or nb.startswith(na):
        return 0.9
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    overlap = len(ta & tb) / max(1, len(ta | tb))
    return max(ratio, overlap)




def embedding_cosine(a: str, aliases: List[str]) -> float:
    if not aliases:
        return 0.0
    texts = [normalize_title_for_match(x) for x in aliases + [a]]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    query_emb = embeddings[-1]
    alias_embs = embeddings[:-1]
    # Compute cosine similarities
    sims = np.dot(alias_embs, query_emb) / (np.linalg.norm(alias_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8)
    return float(np.max(sims)) if len(sims) > 0 else 0.0

def alias_score(a: str, aliases: List[str]) -> float:
    lex = max((lexical_similarity(a, al) for al in aliases), default=0.0)
    cos = embedding_cosine(a, aliases)
    return max(lex, cos)


def load_taxonomy(path: Optional[str] = None) -> Dict[str, Any]:
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError("taxonomy.json was not found. Provide a valid path.")

def _all_aliases(node: Dict[str, Any]) -> List[str]:
    return [node["title"]] + node.get("aliases", [])

def _index_taxonomy(tax: Dict[str, Any]) -> Dict[str, Any]:
    idx: Dict[str, Any] = {"sections": [], "subsections": {}, "sub_to_parent": {}}
    for sec in tax.get("sections", []):
        idx["sections"].append({"node": sec, "aliases": _all_aliases(sec)})
        sec_slug = sec["slug"]
        idx["subsections"][sec_slug] = []
        for sub in sec.get("subsections", []):
            idx["subsections"][sec_slug].append({"node": sub, "aliases": _all_aliases(sub)})
            idx["sub_to_parent"][sub["slug"]] = sec_slug
    return idx

def _best_match(name: str, candidates: List[Dict[str, Any]], threshold: float = 0.7) -> Optional[Dict[str, Any]]:
    best = None
    best_score = 0.0
    for cand in candidates:
        score = alias_score(name, cand["aliases"])
        if score > best_score:
            best_score = score
            best = cand["node"]
    return best if best_score >= threshold else None

def _contains_range(parent: Dict[str, Any], child: Dict[str, Any]) -> bool:
    """Return True if parent [start_page, end_page] fully contains child's [start_page, end_page]."""
    sp_p, ep_p = parent.get("start_page"), parent.get("end_page")
    sp_c, ep_c = child.get("start_page"), child.get("end_page")
    if not (isinstance(sp_p, int) and isinstance(ep_p, int) and isinstance(sp_c, int) and isinstance(ep_c, int)):
        return False
    return sp_p <= sp_c <= ep_p and sp_p <= ep_c <= ep_p

def map_toc_to_taxonomy(
    hierarchy: List[Dict[str, Any]],
    taxonomy: Dict[str, Any],
    threshold_section: float = 0.7,
    threshold_subsection: float = 0.7,
    allow_cross_section_subs: bool = True
) -> Dict[str, Dict[str, Any]]:
    idx = _index_taxonomy(taxonomy)
    mapped: Dict[str, Dict[str, Any]] = {}

    def ensure_section(section_slug: str, matched_title: Optional[str], start_page: Optional[int], end_page: Optional[int]):
        if section_slug not in mapped:
            mapped[section_slug] = {
                "matched_title": matched_title,
                "start_page": start_page,
                "end_page": end_page,
                "subsections": {},
            }
        else:
            if mapped[section_slug]["start_page"] is None and start_page is not None:
                mapped[section_slug]["start_page"] = start_page
            if mapped[section_slug]["end_page"] is None and end_page is not None:
                mapped[section_slug]["end_page"] = end_page

    # 1) Match sections
    for sec in hierarchy:
        sec_title = sec.get("section_title") or ""
        sec_match = _best_match(sec_title, idx["sections"], threshold_section)
        if not sec_match:
            # Keep unmapped; we'll still place cross-matched subsections under canonical parents
            continue
        s_slug = sec_match["slug"]
        ensure_section(s_slug, sec_title, sec.get("start_page"), sec.get("end_page"))

        # 2) Match subsections under this matched section
        sub_candidates = idx["subsections"].get(s_slug, [])
        for sub in sec.get("subsections", []):
            sub_title = sub.get("title") or ""
            sub_match = _best_match(sub_title, sub_candidates, threshold_subsection)

            if not sub_match and allow_cross_section_subs:
                cross_candidates: List[Dict[str, Any]] = []
                for lst in idx["subsections"].values():
                    cross_candidates.extend(lst)
                sub_match = _best_match(sub_title, cross_candidates, threshold_subsection)

            if not sub_match:
                continue

            sub_slug = sub_match["slug"]
            canonical_parent_slug = idx["sub_to_parent"].get(sub_slug, s_slug)

            ensure_section(canonical_parent_slug, mapped.get(canonical_parent_slug, {}).get("matched_title"), None, None)
            mapped[canonical_parent_slug]["subsections"][sub_slug] = {
                "matched_title": sub_title,
                "start_page": sub.get("start_page"),
                "end_page": sub.get("end_page"),
            }

    # 3) Post-validation: move subsections to the section whose page range contains them
    #    if their current canonical parent does not.
    for parent_slug, parent_info in list(mapped.items()):
        subs = list(parent_info["subsections"].items())
        for sub_slug, sub_info in subs:
            # Skip if parent contains range
            if _contains_range(parent_info, sub_info):
                continue
            # Try to find any mapped section that contains this sub range
            for candidate_slug, candidate_info in mapped.items():
                # Only consider moving if taxonomy allows the subsection under candidate
                allowed_subs = [n["node"]["slug"] for n in idx["subsections"].get(candidate_slug, [])]
                if sub_slug in allowed_subs and _contains_range(candidate_info, sub_info):
                    # Move subsection
                    del mapped[parent_slug]["subsections"][sub_slug]
                    mapped[candidate_slug]["subsections"][sub_slug] = sub_info
                    break

    return mapped


import json

def toc_json_to_hierarchy(toc_json: dict) -> list:
    hierarchy = []
    for section_title, section_info in toc_json.items():
        section = {
            "section_title": section_title,
            "start_page": int(section_info.get("start_page", 0)),
            "end_page": int(section_info.get("end_page", 0)),
            "subsections": []
        }
        for sub_title, sub_info in section_info.get("subsections", {}).items():
            subsection = {
                "title": sub_title,
                "start_page": int(sub_info.get("start_page", 0)),
                "end_page": int(sub_info.get("end_page", 0))
            }
            section["subsections"].append(subsection)
        hierarchy.append(section)
    return hierarchy


def process_toc(base: str):
    filename = f"{base}.pdf"
    toc_path = os.path.join('toc', f"{base}.json")
    with open(toc_path, "r", encoding="utf-8") as f:
        toc_json = json.load(f)
    hierarchy = toc_json_to_hierarchy(toc_json)
    print(json.dumps(hierarchy, indent=2))