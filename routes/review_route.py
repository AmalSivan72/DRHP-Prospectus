from flask import Blueprint, request, jsonify
from services.review_service import load_status, update_review, get_csv_path
import os
review_bp = Blueprint("review", __name__)

@review_bp.route("/review/<base>", methods=["GET", "POST"])
def review(base):
    try:
        if request.method == "GET":
            data = load_status(base)
            answers = {}
            for field, answer in data.get("answers", {}).items():
                score_info = data.get("scores", {}).get(field, {})
                score = score_info.get("score", 0)
                if score < 75:
                    answers[field] = {
                        "answer": answer,
                        "score": score,
                        "explanation": score_info.get("explanation", ""),
                        "review_status": data.get("review_status", {}).get(field, "pending")
                    }
            return jsonify({
                "success": True,
                "base": base,
                "csv_file": os.path.basename(get_csv_path(base)),
                "answers": answers
            }), 200

        if request.method == "POST":
            payload = request.get_json(force=True)
            field = payload.get("field")
            action = payload.get("action")
            edited_answer = payload.get("edited_answer", "").strip()
            result = update_review(base, field, action, edited_answer)
            return jsonify(result), 200

    except FileNotFoundError as e:
        return jsonify({"success": False, "message": str(e)}), 404
    except KeyError as e:
        return jsonify({"success": False, "message": str(e)}), 404
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
