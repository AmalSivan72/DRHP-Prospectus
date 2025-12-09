from typing import List, Dict, Any

def build_llm_prompt_drhp(
    particulars: str,
    what_ai_search_for_in_drhp: str,
    top_chunks: List[Dict[str, Any]]
) -> str:
    """
    Build a prompt where 'particulars' is the question
    and 'What AI search for in DRHP' is the guiding instruction.
    """

    def block_repr(c: Dict[str, Any]) -> str:
        sec = c.get("section", "")
        sub = c.get("subsection", "")
        page = c.get("page", "")
        return f"(Section: {sec} | Subsection: {sub} | Page: {page})\n{c.get('text','').strip()}"

    combined_text = "\n\n".join(block_repr(c) for c in top_chunks)

    prompt = f"""
You are given text from a Draft Red Herring Prospectus (DRHP).

Question:
{particulars}

Guidance:
{what_ai_search_for_in_drhp}

Relevant Document Content:
{combined_text}

Your primary objective is to extract the value for the particulars
- First, try to identify the particulars directly from the chunk text.
- If the particulars cannot be found explicitly, use the 'What AI search for in DRHP' as supporting context  to infer the answer.
- If the question seems to look like a one word answer then provide that or else a concise summary.
- Sometimes the answer may not be directly available in the document, use your best judgment to infer the answer based on the provided content.
- Dont inculde the words 'As per the document or chunks or excerpts' in answer.
- Dont give me long paragraphs you can use short summarized explanations but only after the answer.

- Respond strictly in JSON with keys:
{{
  "question": "<repeat the Particulars>",
  "answer": "<short factual answer or concise summary>"
}}
""".strip()

    return prompt
