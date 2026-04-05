#!/usr/bin/env python3
# =============================================================================
#   KKR Gen AI Innovations — Document Scanner AI
#   Flask Web Application
#
#   A simple web interface to scan documents from any browser.
#
#   Website : https://kkrgenaiinnovations.com/
#   Email   : info@kkrgenaiinnovations.com
#   WhatsApp: +1 470-861-6312
# =============================================================================
"""
app.py
------
Start the web server:

  python app.py

Then open your browser and visit:  http://localhost:5000
"""

import os
import uuid
import base64

import cv2
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify

from utils.image_processor import DocumentProcessor
from utils.enhancer import ImageEnhancer
from utils.pdf_converter import PDFConverter


# ===========================================================================
#  Flask app setup
# ===========================================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

UPLOAD_FOLDER  = "static/uploads"
OUTPUT_FOLDER  = "static/output"
ALLOWED_EXTS   = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


# ===========================================================================
#  Routes
# ===========================================================================

@app.route("/")
def index():
    """Home page — upload form."""
    return render_template("index.html")


@app.route("/scan", methods=["POST"])
def scan():
    """
    Receive an uploaded image, scan it, and return the result page.
    """
    # ── Validate upload ────────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTS)}"}), 400

    mode = request.form.get("mode", "adaptive")
    save_pdf_flag = request.form.get("save_pdf") == "on"

    # ── Save upload ────────────────────────────────────────────────────────
    uid       = uuid.uuid4().hex[:8]
    ext       = file.filename.rsplit(".", 1)[1].lower()
    orig_name = f"{uid}_original.{ext}"
    orig_path = os.path.join(UPLOAD_FOLDER, orig_name)
    file.save(orig_path)

    # ── Process ────────────────────────────────────────────────────────────
    image = cv2.imread(orig_path)
    if image is None:
        return jsonify({"error": "Cannot read image file."}), 400

    processor = DocumentProcessor()
    enhancer  = ImageEnhancer()

    corners = processor.detect_corners(image)
    if corners is not None:
        scanned = processor.four_point_transform(image, corners)
        corners_detected = True
    else:
        scanned = image.copy()
        corners_detected = False

    enhanced = enhancer.full_enhance(scanned, mode=mode)

    # ── Save result ────────────────────────────────────────────────────────
    out_name = f"{uid}_scanned.jpg"
    out_path = os.path.join(OUTPUT_FOLDER, out_name)
    cv2.imwrite(out_path, enhanced)

    pdf_name = None
    if save_pdf_flag:
        pdf_name = f"{uid}_scanned.pdf"
        PDFConverter.image_to_pdf(enhanced, os.path.join(OUTPUT_FOLDER, pdf_name))

    # ── Build base64 preview strings ──────────────────────────────────────
    orig_b64   = _img_to_b64(image)
    result_b64 = _img_to_b64(enhanced)

    return render_template(
        "result.html",
        orig_b64=orig_b64,
        result_b64=result_b64,
        mode=mode,
        corners_detected=corners_detected,
        out_name=out_name,
        pdf_name=pdf_name,
        orig_size=f"{image.shape[1]}×{image.shape[0]}",
        scan_size=f"{enhanced.shape[1] if enhanced.ndim > 1 else enhanced.shape[0]}×{enhanced.shape[0]}",
    )


@app.route("/download/<filename>")
def download(filename: str):
    """Serve a file from the output folder for download."""
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.isfile(path):
        return "File not found", 404
    return send_file(path, as_attachment=True)


# ===========================================================================
#  Helper
# ===========================================================================

def _img_to_b64(image: np.ndarray) -> str:
    """Encode a numpy image as a base64 JPEG string for inline HTML display."""
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


# ===========================================================================
#  Entry point
# ===========================================================================

if __name__ == "__main__":
    print("\n  KKR Gen AI Innovations — Document Scanner AI (Web)")
    print("  Open your browser: http://localhost:5000\n")
    app.run(debug=True, port=5000)
