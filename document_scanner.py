#!/usr/bin/env python3
# =============================================================================
#
#   ██╗  ██╗██╗  ██╗██████╗      ██████╗ ███████╗███╗   ██╗     █████╗ ██╗
#   ██║ ██╔╝██║ ██╔╝██╔══██╗    ██╔════╝ ██╔════╝████╗  ██║    ██╔══██╗██║
#   █████╔╝ █████╔╝ ██████╔╝    ██║  ███╗█████╗  ██╔██╗ ██║    ███████║██║
#   ██╔═██╗ ██╔═██╗ ██╔══██╗    ██║   ██║██╔══╝  ██║╚██╗██║    ██╔══██║██║
#   ██║  ██╗██║  ██╗██║  ██║    ╚██████╔╝███████╗██║ ╚████║    ██║  ██║██║
#   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═══╝    ╚═╝  ╚═╝╚═╝
#
#   Document Scanner AI  —  Command-Line Interface
#   Empowering Tomorrow with AI-Powered Solutions
#
#   Company  : KKR Gen AI Innovations
#   Website  : https://kkrgenaiinnovations.com/
#   Email    : info@kkrgenaiinnovations.com
#   WhatsApp : +1 470-861-6312
#   Twitter  : https://x.com/kkr_genai_
#   Facebook : https://www.facebook.com/kkrgenaiinnovations
#   Instagram: https://www.instagram.com/kkrgenaiinnovations/
#   LinkedIn : https://www.linkedin.com/company/kkr-genai-innovations/
# =============================================================================
"""
document_scanner.py
-------------------
Single-image document scanner — command-line interface.

USAGE EXAMPLES
--------------
  # Basic scan (adaptive threshold — recommended for photos)
  python document_scanner.py --input samples/receipt.jpg

  # Save as PDF
  python document_scanner.py --input samples/letter.jpg --pdf

  # Choose enhancement mode
  python document_scanner.py --input samples/invoice.jpg --mode color

  # Show every processing step (great for learning!)
  python document_scanner.py --input samples/doc.jpg --debug

  # Specify output file name
  python document_scanner.py --input samples/doc.jpg --output output/my_scan.jpg
"""

import argparse
import sys
import os
import cv2
import numpy as np

# ── Local imports ──────────────────────────────────────────────────────────
from utils.image_processor import DocumentProcessor
from utils.enhancer import ImageEnhancer
from utils.pdf_converter import PDFConverter


# ===========================================================================
#  Branding helpers
# ===========================================================================

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║         KKR Gen AI Innovations — Document Scanner AI            ║
║              Empowering Tomorrow with AI Innovation             ║
║  Web : https://kkrgenaiinnovations.com/  |  WA: +1 470-861-6312 ║
╚══════════════════════════════════════════════════════════════════╝
"""


def print_banner():
    print(BANNER)


# ===========================================================================
#  Core scanning function
# ===========================================================================

def scan_document(
    input_path: str,
    output_path: str | None = None,
    mode: str = "adaptive",
    save_pdf: bool = False,
    debug: bool = False,
) -> str:
    """
    Scan a single document image.

    Parameters
    ----------
    input_path  : path to the input image
    output_path : where to save the result  (auto-generated if None)
    mode        : enhancement mode — "adaptive" | "otsu" | "color" | "grayscale"
    save_pdf    : also export a PDF version
    debug       : show intermediate steps in pop-up windows

    Returns
    -------
    str – path to the saved output image
    """
    # ── Load image ─────────────────────────────────────────────────────────
    print(f"  [1/5]  Loading image  →  {input_path}")
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Cannot open image: {input_path}")

    print(f"         Image size: {image.shape[1]}×{image.shape[0]} px")

    # ── Detect document corners ────────────────────────────────────────────
    print("  [2/5]  Detecting document boundaries …")
    processor = DocumentProcessor()
    corners = processor.detect_corners(image)

    if corners is None:
        print("  ⚠️   No document boundary found.")
        print("         Tip: make sure the document is on a contrasting background.")
        print("         Falling back to full image (no perspective correction).")
        scanned = image.copy()
    else:
        print(f"         Corners detected: {corners.astype(int).tolist()}")
        if debug:
            debug_img = processor.draw_corners(image, corners)
            _show("DEBUG: Detected Corners", debug_img)

        # ── Perspective warp ───────────────────────────────────────────────
        print("  [3/5]  Applying perspective transformation …")
        scanned = processor.four_point_transform(image, corners)
        if debug:
            _show("DEBUG: After Perspective Warp", scanned)

    # ── Enhance image ──────────────────────────────────────────────────────
    print(f"  [4/5]  Enhancing image  (mode={mode}) …")
    enhancer = ImageEnhancer()
    enhanced = enhancer.full_enhance(scanned, mode=mode)
    if debug:
        _show("DEBUG: Enhanced Output", enhanced)

    # ── Save result ────────────────────────────────────────────────────────
    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", f"{base}_scanned.jpg")

    print(f"  [5/5]  Saving result  →  {output_path}")
    cv2.imwrite(output_path, enhanced)
    print(f"\n  ✅  Scan complete!  Saved: {output_path}")

    # ── Optional PDF ───────────────────────────────────────────────────────
    if save_pdf:
        pdf_path = os.path.splitext(output_path)[0] + ".pdf"
        PDFConverter.image_to_pdf(enhanced, pdf_path)
        print(f"  📄  PDF saved     :  {pdf_path}")

    return output_path


# ===========================================================================
#  Debug helper
# ===========================================================================

def _show(title: str, image: np.ndarray):
    """Display an image in a named window.  Press any key to continue."""
    # Scale down very large images so they fit on screen
    h, w = image.shape[:2]
    scale = min(1.0, 900 / max(h, w, 1))
    display = cv2.resize(image, (int(w * scale), int(h * scale)))
    cv2.imshow(title, display)
    print(f"         [DEBUG] Showing '{title}' — press any key to continue …")
    cv2.waitKey(0)
    cv2.destroyWindow(title)


# ===========================================================================
#  CLI argument parser
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="document_scanner.py",
        description=(
            "KKR Gen AI Innovations — Document Scanner AI\n"
            "Transform a photo of a document into a clean, flat, high-contrast scan.\n"
            "Website: https://kkrgenaiinnovations.com/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ENHANCEMENT MODES
  adaptive  (default) – Adaptive Gaussian threshold; best for camera photos
                         with uneven lighting or shadows.
  otsu                – Otsu's global threshold; great for clean, flat scans.
  color               – Keep colours; boost contrast with CLAHE.
  grayscale           – Simple grayscale, no threshold.

EXAMPLES
  python document_scanner.py --input samples/receipt.jpg
  python document_scanner.py --input samples/letter.jpg --mode color --pdf
  python document_scanner.py --input samples/invoice.jpg --debug

Contact: info@kkrgenaiinnovations.com  |  WA: +1 470-861-6312
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the input image (JPG, PNG, BMP, TIFF …)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path for the output image (default: output/<name>_scanned.jpg)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["adaptive", "otsu", "color", "grayscale"],
        default="adaptive",
        help="Enhancement mode (default: adaptive)"
    )
    parser.add_argument(
        "--pdf", action="store_true",
        help="Also export the result as a PDF"
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Show intermediate processing steps in pop-up windows"
    )
    return parser


# ===========================================================================
#  Entry point
# ===========================================================================

def main():
    print_banner()
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"\n  ❌  Error: File not found — '{args.input}'")
        print("     Please check the path and try again.\n")
        sys.exit(1)

    try:
        scan_document(
            input_path=args.input,
            output_path=args.output,
            mode=args.mode,
            save_pdf=args.pdf,
            debug=args.debug,
        )
    except Exception as exc:
        print(f"\n  ❌  Error: {exc}\n")
        sys.exit(1)

    print("\n  Powered by KKR Gen AI Innovations — https://kkrgenaiinnovations.com/\n")


if __name__ == "__main__":
    main()
