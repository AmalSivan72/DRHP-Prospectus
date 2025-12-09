from flask import Blueprint, request, jsonify
import logging
from sqlalchemy.exc import IntegrityError
from services import file_service, db_services, index_service, toc_service_drhp
from utility.helpers import save_toc_to_db

upload_bp = Blueprint("upload", __name__)

@upload_bp.route("/upload", methods=["POST"])
def upload_dhrp():
    try:
        company = request.form.get("company")
        bse_code = request.form.get("bse_code")
        upload_date = request.form.get("upload_date")
        uploader_name = request.form.get("uploader_name")
        promoter = request.form.get("promoter")
        pdf = request.files.get("pdf")

        if not pdf or pdf.filename == "":
            logging.warning(f"Upload failed — No PDF provided by {uploader_name} for {company}")
            return jsonify({"success": False, "message": "No PDF uploaded"}), 400

        filename, pdf_path, base = file_service.save_pdf(pdf, uploader_name, company)

        try:
            db_services.save_entry(company, bse_code, upload_date, uploader_name, promoter, filename)
        except ValueError as dup_err:
            return jsonify({"success": False, "message": str(dup_err)}), 409  # Conflict
        except IntegrityError as db_err:
            logging.warning(f"⚠️ Integrity error: {db_err}")
            return jsonify({"success": False, "message": "File already exists in the system."}), 409

        index_service.update_index(company, bse_code, upload_date, uploader_name, promoter, filename)

        hierarchy = toc_service_drhp.extract_and_save_toc(pdf_path, base)
        if hierarchy:
            save_toc_to_db(base, hierarchy)

        return jsonify({
            "success": True,
            "message": "DHRP uploaded successfully",
            "entry": {
                "company": company,
                "bse_code": bse_code,
                "upload_date": upload_date,
                "uploader_name": uploader_name,
                "promoter": promoter,
                "pdf_filename": filename,
                "status": "New"
            },
            "base": base
        }), 200

    except Exception as e:
        logging.error(f"❌ Upload error for {request.form.get('company', 'Unknown')}: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500
