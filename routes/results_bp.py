import os, json
from flask import Blueprint, jsonify

results_status_bp = Blueprint("results_status", __name__)

@results_status_bp.route("/status/<base>/details", methods=["GET"])
def get_status_details(base):
    """Return full details: answers, reasoning, validation, milestones, plus scores."""
    path = os.path.join("status", f"{base}.json")
    if not os.path.exists(path):
        return jsonify({"success": False, "message": f"No status file for {base}"}), 404
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ✅ Include scores explicitly alongside the rest of the data
    return jsonify({
        "success": True,
        "base": base,
        "data": data,
        "scores": data.get("scores", {})
    }), 200


@results_status_bp.route("/status/<base>/answers", methods=["GET"])
def get_completed_answers(base):
    """Return only data_field + answer for fields that are completed."""
    path = os.path.join("status", f"{base}.json")
    if not os.path.exists(path):
        return jsonify({"success": False, "message": f"No status file for {base}"}), 404
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    completed = {}
    
    return jsonify({"success": True, "base": base, "answers": completed}), 200
