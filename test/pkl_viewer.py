import pickle
import os
from pprint import pprint

def view_chunks(pkl_path: str, limit: int = 5):
    if not os.path.exists(pkl_path):
        print(f"❌ File not found: {pkl_path}")
        return

    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"✅ Loaded {len(chunks)} chunks from {pkl_path}\n")

    for i, chunk in enumerate(chunks[:limit]):
        print(f"--- Chunk {i+1} ---")
        print(f"chunk_id       : {chunk.get('chunk_id')}")
        print(f"section        : {chunk.get('section')}")
        print(f"subsection     : {chunk.get('subsection')}")
        print(f"chunk_index    : {chunk.get('chunk_index')}")
        print(f"pdf_start_page : {chunk.get('pdf_start_page')}")
        print(f"pdf_end_page   : {chunk.get('pdf_end_page')}")
        print(f"char_count     : {chunk.get('char_count')}")
        print(f"text           : {chunk.get('text')[:300]}... [truncated]")
        print(f"source         : {chunk.get('source')}")
        print()

# Example usage
pkl_path = r"C:\Users\2000166072\Desktop\dhrp2\pickles\pro_spdrdowjonesindustrialaverageetf.pkl"
view_chunks(pkl_path, limit=30)
