from typing import List, Dict, Any

def build_llm_prompt_for_chunk_extraction(
    data_field: str,
    particulars: str,
    top_chunks: List[Dict[str, Any]]
) -> str:
    
    def block_repr(i: int, c: Dict[str, Any]) -> str:
        sec = c.get("section", "")
        sub = c.get("subsection", "")
        page = c.get("page", "")
        return f"[Excerpt {i+1} | Section: {sec} | Subsection: {sub} | Page: {page}]\n{c.get('text','').strip()}"

    combined_text = "\n\n".join(block_repr(i, c) for i, c in enumerate(top_chunks))

    prompt = f"""
You are given text chunks.

Your primary objective is to extract the value for the Data Field.
- First, try to identify the Data Field value directly from the chunk text.
- If the Data Field cannot be found explicitly, use the Particulars as supporting context to infer the answer.
- If neither the Data Field nor the Particulars provide a clear match, analyze the chunk carefully and return the nearest relevant answer that best fits the intent.

Data Field: {data_field}
Particulars: {particulars}

Document Content:
{combined_text}

Respond strictly in JSON with keys:
{{
  "data_field": "<repeat the Data Field>",
  "answer": "<the extracted value>"
}}
""".strip()


    return prompt
