import logging
from models import db
from models.dhrp_entry import DhrpEntry


def save_entry(company, bse_code, upload_date, uploader_name, promoter, filename):
    """
    Save a new DHRP entry if not duplicate.
    Duplicate check is based only on bse_code (identity number).
    Promoter differences are ignored.
    """
    # 🔎 Check for duplicates by bse_code only
    existing = DhrpEntry.query.filter_by(bse_code=bse_code).first()

    if existing:
        logging.warning(
            f"⚠️ Duplicate entry blocked: company={company}, bse_code={bse_code}"
        )
        raise ValueError("File already available in list: An entry with the same BSE code exists.")

    # ✅ Save new entry
    entry = DhrpEntry(
        company=company,
        bse_code=bse_code,
        upload_date=upload_date,
        uploader_name=uploader_name,
        promoter=promoter,
        pdf_filename=filename,
        status="Processing"
    )
    db.session.add(entry)
    db.session.commit()
    logging.info(f"📥 DHRP entry saved to database: {filename}")
    return entry
