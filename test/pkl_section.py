import pickle
import os

def find_section_for_text(pkl_path: str, query: str, limit: int = 3):
    if not os.path.exists(pkl_path):
        print(f"❌ File not found: {pkl_path}")
        return

    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)

    matches = []
    for chunk in chunks:
        text = chunk.get("text", "")
        if query.lower() in text.lower():
            matches.append({
                "section": chunk.get("section"),
                "subsection": chunk.get("subsection"),
                "page_range": f"{chunk.get('pdf_start_page')}–{chunk.get('pdf_end_page')}",
                "preview": text[:2000] + "...",
            })
            if len(matches) >= limit:
                break

    if not matches:
        print(f"⚠️ No matches found for: {query}")
    else:
        print(f"✅ Found {len(matches)} matches for: {query}\n")
        for i, m in enumerate(matches, 1):
            print(f"--- Match {i} ---")
            print(f"Section    : {m['section']}")
            print(f"Subsection : {m['subsection']}")
            print(f"Pages      : {m['page_range']}")
            print(f"Preview    : {m['preview']}")
            print()

# Example usage
pkl_path = r"C:\Users\2000166072\Desktop\dhrp2\pickles\pro_spdrdowjonesindustrialaverageetf.pkl"
find_section_for_text(pkl_path, "Transactions with Affiliates of the Trustee and Sponsor ")
