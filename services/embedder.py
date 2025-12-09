import pickle
from sentence_transformers import SentenceTransformer
import os
import logging
from typing import List, Dict, Any

from utility.helpers import load_local_model

def embed_chunks_optimal(
    input_pkl_path: str,
    output_pkl_path: str,
    model_path: str = r"C:\Users\2000166072\Documents\DRHP Prospectus\dhrp2\Embedding_model"
) -> List[Dict[str, Any]]:
    # Load chunks
    with open(input_pkl_path, "rb") as f:
        chunks = pickle.load(f)

    logging.info(f"📦 Loaded {len(chunks)} chunks from {input_pkl_path}")

    model = load_local_model(model_path, dim=768)

    embedding_inputs = []
    valid_indices = []

    # Build embedding inputs
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        if text.strip():
            # Only embed the text itself
            chunk["embedding_input"] = text.strip()
            embedding_inputs.append(chunk["embedding_input"])
            valid_indices.append(i)
        else:
            chunk["embedding_input"] = ""
            chunk["embedding_vector"] = None
            logging.warning(f"⚠️ Skipping empty chunk at index {i} (section={chunk.get('section')})")

    if not embedding_inputs:
        logging.warning("⚠️ No valid chunks found for embedding.")
        return chunks

    # Generate embeddings
    logging.info("⚙️ Generating embeddings...")
    embeddings = model.encode(embedding_inputs, show_progress_bar=True)

    # Attach vectors back to chunks
    for j, i in enumerate(valid_indices):
        chunks[i]["embedding_vector"] = embeddings[j]
        logging.debug(f"Embedded chunk {i} (section={chunks[i].get('section')})")

    # Save updated pickle
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(chunks, f)

    logging.info(f"✅ Saved embedded chunks to {output_pkl_path}")
    return chunks
