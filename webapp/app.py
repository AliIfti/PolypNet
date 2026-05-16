"""
PolypNet Flask Web Application
"""

import os
import json
import uuid
from pathlib import Path

from flask import (
    Flask, request, jsonify, render_template,
    send_from_directory, session
)
from werkzeug.utils import secure_filename

# Add project root to path so webapp/ can import src/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webapp.config import (
    UPLOAD_FOLDER, RESULTS_FOLDER,
    ALLOWED_EXTENSIONS, SECRET_KEY, MAX_CONTENT_LENGTH
)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Ensure directories exist
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    Expects: multipart/form-data with field "image"
    Returns: JSON with all pipeline results
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # Save upload
    filename  = secure_filename(file.filename)
    unique_fn = f"{uuid.uuid4().hex[:8]}_{filename}"
    save_path = Path(UPLOAD_FOLDER) / unique_fn
    file.save(str(save_path))

    # Run pipeline (imported lazily to avoid slow startup)
    try:
        from webapp.pipeline import run_pipeline
        result = run_pipeline(str(save_path))
        result["upload_filename"] = unique_fn
        return jsonify(result)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Pipeline error: {str(exc)}"}), 500


@app.route("/results")
def results():
    return render_template("results.html")


# ─────────────────────────────────────────────────────────────────────────────
# Static helpers (uploads & results served explicitly)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PolypNet Web Application")
    print("  http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
