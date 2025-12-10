import os, logging
from models.dhrp_entry import DhrpEntry, ProcessingStatus, RiskSummary, TocSection
from utility.helpers import normalize_name, get_base_name
from models import db

def delete_document(doc: str):
    """Delete files, Gemini cache, status JSON, and DB records for a given document."""
    base = normalize_name(get_base_name(doc))
    filename = f"{base}.pdf"

    # Files to delete
    paths = [
        f"uploads/{base}.pdf",
        f"risk_summary/{base}.json",
        f"answered_csv/{base}_analysis.csv",
        f"status/{base}.json"  # ✅ status JSON
    ]

    # Delete Gemini cache files (gemini_summary_row_X.txt)
    cache_dir = "cache"
    if os.path.exists(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.startswith("gemini_summary_row_") and fname.endswith(".txt"):
                os.remove(os.path.join(cache_dir, fname))
                logging.info(f"🗑️ Gemini cache deleted: {fname}")

    # Delete listed paths
    for path in paths:
        if os.path.exists(path):
            os.remove(path)
            logging.info(f"🗑️ File deleted: {path}")
        else:
            logging.warning(f"⚠️ File not found for deletion: {path}")

    # DB cleanup
    entry = DhrpEntry.query.filter_by(pdf_filename=filename).first()
    if entry:
        TocSection.query.filter_by(dhrp_id=entry.id).delete()
        RiskSummary.query.filter_by(dhrp_id=entry.id).delete()
        ProcessingStatus.query.filter_by(dhrp_id=entry.id).delete()
        db.session.delete(entry)
        db.session.commit()
        logging.info(f"🗂️ DB entry and related records removed for: {filename} — Company: {entry.company}")
        return True, f"{doc} deleted successfully"
    else:
        logging.warning(f"⚠️ No matching DB entry found for base: {base}")
        return False, f"No matching DB entry found for {doc}"
