import os, json
import pandas as pd
from models import db
from models.dhrp_entry import FundEvaluation   

def get_csv_path(base: str) -> str:
    """Resolve the CSV path for a given base (always expects _analysis.csv)."""
    return os.path.join("answered_csv", f"{base}_analysis.csv")

def get_status_path(base: str) -> str:
    """Resolve the JSON status path for a given base."""
    return os.path.join("status", f"{base}.json")

def load_status(base: str) -> dict:
    path = get_status_path(base)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No status file for {base}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_status(base: str, data: dict):
    path = get_status_path(base)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def update_review(base: str, field: str, action: str, edited_answer: str = "") -> dict:
    """
    Approve or reject a field. Updates JSON, CSV, and DB.
    Adds a 'review_status' column in CSV and JSON.
    """
    status_data = load_status(base)
    csv_path = get_csv_path(base)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No CSV file for {base}")

    if field not in status_data["answers"]:
        raise KeyError(f"Field {field} not found")

    score_info = status_data.get("scores", {}).get(field, {})
    score = score_info.get("score", 0)
    if score >= 75:
        raise ValueError(f"Field {field} has score {score}, review not required.")

    df = pd.read_csv(csv_path)

    if action == "approve":
        final_answer = edited_answer if edited_answer else status_data["answers"][field]
        status_data["answers"][field] = final_answer
        status_data.setdefault("review_status", {})[field] = "approved"

        if "Data Fields" in df.columns:
            df.loc[df["Data Fields"].str.strip() == field, "Answer"] = final_answer
            df.loc[df["Data Fields"].str.strip() == field, "Review Status"] = "approved"

        # ✅ Update DB
        record = FundEvaluation.query.filter_by(base=base, data_field=field).first()
        if record:
            record.answer = final_answer
            record.review_status = "approved"
            db.session.add(record)

    elif action == "reject":
        status_data["answers"][field] = "[REJECTED]"
        status_data.setdefault("review_status", {})[field] = "rejected"

        if "Data Fields" in df.columns:
            df.loc[df["Data Fields"].str.strip() == field, "Answer"] = "[REJECTED]"
            df.loc[df["Data Fields"].str.strip() == field, "Review Status"] = "rejected"

        # ✅ Update DB
        record = FundEvaluation.query.filter_by(base=base, data_field=field).first()
        if record:
            record.answer = "[REJECTED]"
            record.review_status = "rejected"
            db.session.add(record)

    else:
        raise ValueError("Invalid action. Use 'approve' or 'reject'.")

    # Ensure Review Status column exists
    if "Review Status" not in df.columns:
        df["Review Status"] = ""

    df.to_csv(csv_path, index=False)
    save_status(base, status_data)

    # ✅ Commit DB changes
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        raise

    return {"success": True, "message": f"Answer for {field} {action}d."}
