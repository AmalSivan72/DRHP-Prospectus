import os, json, logging, re, pandas as pd

from utility.helpers import load_local_model
from services.core_logic_drhp import evaluate_with_gemini, get_chunks_with_fallback, parse_gemini_response
from services.prompt_builder_drhp import build_llm_prompt_drhp
from models import db
from models.dhrp_entry import FundEvaluation   

def strip_emojis(text: str) -> str:
    """Remove emoji characters from a string for frontend safety."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  
        "\U0001F300-\U0001F5FF"  
        "\U0001F680-\U0001F6FF"  
        "\U0001F1E0-\U0001F1FF"  
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def stream_csv_evaluation(
    base: str,
    chunks: list,
    csv_path: str,
    output_path: str,
    model_path: str = r"C:\Users\2000166072\Documents\DRHP Prospectus\dhrp2\Embedding_model"
):
    
    embedding_model = load_local_model(model_path, dim=768)
    logging.info("Embedding dimension: %d", embedding_model.get_sentence_embedding_dimension())
    status_path = os.path.join('status', f"{base}.json")
    os.makedirs('status', exist_ok=True)
    status_data = {
        "answers": {},
        "review_status": {},   
        "done": False
    }

    def save_status():
        with open(status_path, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, indent=2)

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    answers, scores, exact_counts = [], [], []

    logging.info(f"🚀 Starting evaluation for base={base}, rows={len(df)}")

    for idx, row in df.iterrows():
        particulars = str(row.get("Particulars", "")).strip()
        what_ai_search_for = str(row.get("What AI search for in DRHP", "")).strip()
        section = str(row.get("Heading", "")).strip()
        subsection = str(row.get("Sub-Heading", "")).strip()

        user_question = particulars if particulars and particulars.upper() != "N/A" else "Unnamed Question"

        top_k_chunks, final_section, final_subsection,exact_count = get_chunks_with_fallback(
            user_question, chunks, k=5, model_path=model_path,
            section=section, subsection=subsection
        )
        logging.info(f"🔎 Row {idx+1}: selected {len(top_k_chunks)} top chunks (section={final_section}, subsection={final_subsection})")
        exact_counts.append(exact_count)
        # Use the new prompt builder
        prompt = build_llm_prompt_drhp(
            particulars,
            what_ai_search_for,
            top_k_chunks
        )
        logging.info(f"📝 Row {idx+1}: built LLM prompt for question={user_question}")
        logging.info(f"--- Prompt for Row {idx+1} ({user_question}) ---\n{prompt}\n")

        cache_path = f"cache/gemini_summary_row_{idx+1}.txt"
        summary = evaluate_with_gemini(prompt, cache_path)
        parsed = parse_gemini_response(summary)

        logging.info(f"🤖 Row {idx+1}: parsed answer='{parsed.get('answer','')}'")

        # Save parsed results
        status_data["answers"][user_question] = parsed.get("answer", "")
        status_data["review_status"][user_question] = "pending"
        save_status()

        answers.append(parsed.get("answer", ""))

        # Save into DB (only question + answer + review_status)
        record = FundEvaluation(
            base=base,
            data_field=user_question,
            answer=parsed.get("answer", ""),
            review_status="pending"
        )
        db.session.add(record)

    # Save CSV with answers
    df["Answer"] = answers
    df["Review Status"] = ["pending"] * len(df)
    df["Exact Match Count"] = exact_counts
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"✅ CSV evaluation complete: {output_path}")

    status_data["done"] = True
    save_status()

    # Commit DB changes
    try:
        db.session.commit()
        logging.info("💾 DB commit successful")
    except Exception as e:
        db.session.rollback()
        logging.error(f"❌ DB commit failed: {e}")
