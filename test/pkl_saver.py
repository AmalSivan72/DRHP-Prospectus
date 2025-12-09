import pickle
import os

def save_chunks_to_txt(pkl_path: str, txt_path: str):
    if not os.path.exists(pkl_path):
        print(f"❌ File not found: {pkl_path}")
        return

    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"✅ Loaded {len(chunks)} chunks from {pkl_path}")

    with open(txt_path, "w", encoding="utf-8") as out:
        out.write(f"✅ Loaded {len(chunks)} chunks from {pkl_path}\n\n")
        for i, chunk in enumerate(chunks):
            out.write(f"--- Chunk {i+1} ---\n")
            out.write(f"chunk_id       : {chunk.get('chunk_id')}\n")
            out.write(f"section        : {chunk.get('section')}\n")
            out.write(f"subsection     : {chunk.get('subsection')}\n")
            out.write(f"chunk_index    : {chunk.get('chunk_index')}\n")
            out.write(f"pdf_start_page : {chunk.get('pdf_start_page')}\n")
            out.write(f"pdf_end_page   : {chunk.get('pdf_end_page')}\n")
            out.write(f"char_count     : {chunk.get('char_count')}\n")
            out.write(f"source         : {chunk.get('source')}\n")
            out.write("text           :\n")
            out.write(chunk.get("text", "").strip() + "\n")
            out.write("\n")

    print(f"📄 Saved all chunks with metadata to {txt_path}")

# Example usage
pkl_path = r"C:\Users\2000166072\Documents\DRHP Prospectus\dhrp2\pickles\tata_capital.pkl"
txt_path = r"C:\Users\2000166072\Documents\DRHP Prospectus\dhrp2\test\tata_capital_chunks.txt"
save_chunks_to_txt(pkl_path, txt_path)
