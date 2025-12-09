import os, logging
from flask import current_app
from werkzeug.utils import secure_filename
from utility.helpers import normalize_name, get_base_name

def save_pdf(pdf, uploader_name, company):
    original_filename = secure_filename(pdf.filename)
    base = normalize_name(get_base_name(original_filename))
    filename = f"{base}.pdf"

    # Use current_app to access config
    upload_folder = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)  # ensure folder exists

    pdf_path = os.path.join(upload_folder, filename)
    pdf.save(pdf_path)

    logging.info(f"📄 PDF saved: {filename} by {uploader_name} for {company}")
    return filename, pdf_path, base
