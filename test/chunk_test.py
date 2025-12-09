import pickle
import os
from typing import List, Dict, Optional


def extract_chunks_by_heading(pkl_path: str, section: str, subsection: Optional[str] = None) -> List[Dict[str, any]]:
    if not os.path.exists(pkl_path):
        print(f"❌ Pickle file not found: {pkl_path}")
        return []

    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)

    filtered = [
        chunk for chunk in chunks
        if chunk["section"] == section and (subsection is None or chunk["subsection"] == subsection)
    ]

    print(f"\n🔍 Found {len(filtered)} chunks for section '{section}'" +
          (f" → subsection '{subsection}'" if subsection else ""))

    for chunk in filtered:
        print(f"\n📄 Chunk {chunk['chunk_index']} | Pages {chunk['pdf_start_page']}–{chunk['pdf_end_page']}")
        print(f"🆔 ID: {chunk['chunk_id']}")
        print(f"🔤 Characters: {chunk['char_count']}")
        print(f"📝 Preview:\n{chunk['text'][:500].strip()}...\n{'-'*60}")

    return filtered


# 🔧 Example usage
if __name__ == "__main__":
    # Replace with your actual file path and section/subsection
    pkl_file = r"C:\Users\2000166072\Desktop\dhrp2\pickles\pro_spdrdowjonesindustrialaverageetf.pkl"
    section_name = "Summary"
    subsection_name = "Summary"  # or None if you want all chunks under the section

    extract_chunks_by_heading(pkl_file, section_name, subsection_name)
