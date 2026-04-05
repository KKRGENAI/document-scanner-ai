#!/usr/bin/env python3
# =============================================================================
#   KKR Gen AI Innovations — Document Scanner AI
#   Batch Processing Script
#
#   Scan an entire folder of document images in one go.
#   Supports parallel processing for speed.
#
#   Website : https://kkrgenaiinnovations.com/
#   Email   : info@kkrgenaiinnovations.com
#   WhatsApp: +1 470-861-6312
# =============================================================================
"""
batch_scanner.py
----------------
Scan every image in a folder (or a list of files).

USAGE EXAMPLES
--------------
  # Scan all images in a folder
  python batch_scanner.py --input samples/

  # Scan and export combined PDF
  python batch_scanner.py --input samples/ --pdf

  # Use 4 CPU workers for speed
  python batch_scanner.py --input samples/ --workers 4

  # Choose enhancement mode
  python batch_scanner.py --input samples/ --mode otsu
"""

import argparse
import os
import sys
import time
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.image_processor import DocumentProcessor
from utils.enhancer import ImageEnhancer
from utils.pdf_converter import PDFConverter

# Supported image file extensions
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║     KKR Gen AI Innovations — Document Scanner AI (Batch)        ║
║         Empowering Tomorrow with AI-Powered Solutions           ║
║  Web: https://kkrgenaiinnovations.com/  | WA: +1 470-861-6312   ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ===========================================================================
#  Single-file worker  (called in thread pool)
# ===========================================================================

def process_one(
    input_path: str,
    output_dir: str,
    mode: str,
) -> dict:
    """
    Scan a single image and save the result.

    Returns a dict with status info for the progress report.
    """
    result = {"file": input_path, "status": "ok", "output": "", "error": ""}
    try:
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError("cv2.imread returned None — unsupported file?")

        processor = DocumentProcessor()
        enhancer  = ImageEnhancer()

        # Detect + warp
        corners = processor.detect_corners(image)
        if corners is not None:
            scanned = processor.four_point_transform(image, corners)
        else:
            scanned = image  # fallback: use full image

        # Enhance
        enhanced = enhancer.full_enhance(scanned, mode=mode)

        # Save
        stem = Path(input_path).stem
        out_path = os.path.join(output_dir, f"{stem}_scanned.jpg")
        cv2.imwrite(out_path, enhanced)
        result["output"] = out_path

    except Exception as exc:
        result["status"] = "error"
        result["error"]  = str(exc)

    return result


# ===========================================================================
#  Batch runner
# ===========================================================================

def batch_scan(
    input_dir: str,
    output_dir: str,
    mode: str = "adaptive",
    save_pdf: bool = False,
    workers: int = 2,
) -> list[dict]:
    """
    Scan all supported images in input_dir and save results to output_dir.

    Parameters
    ----------
    input_dir  : folder containing input images
    output_dir : folder to save scanned images
    mode       : enhancement mode
    save_pdf   : combine all results into one PDF
    workers    : number of parallel threads

    Returns
    -------
    list of result dicts
    """
    # Collect image files
    files = [
        str(p) for p in Path(input_dir).iterdir()
        if p.suffix.lower() in SUPPORTED
    ]
    if not files:
        print(f"  ⚠️   No supported images found in '{input_dir}'")
        print(f"       Supported types: {', '.join(SUPPORTED)}")
        return []

    files.sort()
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Found {len(files)} image(s) to process.")
    print(f"  Output folder : {output_dir}")
    print(f"  Enhancement   : {mode}")
    print(f"  Workers       : {workers}")
    print(f"  {'─' * 60}")

    results = []
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_one, f, output_dir, mode): f
            for f in files
        }
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            res = future.result()
            status_icon = "✅" if res["status"] == "ok" else "❌"
            filename = os.path.basename(res["file"])
            if res["status"] == "ok":
                print(f"  {status_icon}  [{done_count:>3}/{len(files)}]  {filename:<35} → {os.path.basename(res['output'])}")
            else:
                print(f"  {status_icon}  [{done_count:>3}/{len(files)}]  {filename:<35} ERROR: {res['error']}")
            results.append(res)

    elapsed = time.perf_counter() - start
    ok_count  = sum(1 for r in results if r["status"] == "ok")
    err_count = len(results) - ok_count

    print(f"\n  {'─' * 60}")
    print(f"  Batch complete in {elapsed:.1f}s")
    print(f"  ✅  Success : {ok_count}   ❌  Failed : {err_count}")

    # ── Optional: combined PDF ──────────────────────────────────────────────
    if save_pdf and ok_count > 0:
        print("\n  Building combined PDF …")
        ok_outputs = sorted(r["output"] for r in results if r["status"] == "ok")
        scanned_images = []
        for path in ok_outputs:
            img = cv2.imread(path)
            if img is not None:
                scanned_images.append(img)

        if scanned_images:
            pdf_path = os.path.join(output_dir, "batch_scan.pdf")
            PDFConverter.images_to_pdf(scanned_images, pdf_path)
            print(f"  📄  Combined PDF : {os.path.abspath(pdf_path)}")

    return results


# ===========================================================================
#  CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batch_scanner.py",
        description=(
            "KKR Gen AI Innovations — Document Scanner AI (Batch Mode)\n"
            "Scan an entire folder of document images at once.\n"
            "Website: https://kkrgenaiinnovations.com/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  python batch_scanner.py --input samples/
  python batch_scanner.py --input samples/ --mode color --pdf --workers 4

Contact: info@kkrgenaiinnovations.com  |  WA: +1 470-861-6312
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Folder containing input images"
    )
    parser.add_argument(
        "--output", "-o", default="output/batch",
        help="Folder to save scanned images (default: output/batch)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["adaptive", "otsu", "color", "grayscale"],
        default="adaptive",
        help="Enhancement mode (default: adaptive)"
    )
    parser.add_argument(
        "--pdf", action="store_true",
        help="Combine all results into one multi-page PDF"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=2,
        help="Number of parallel worker threads (default: 2)"
    )
    return parser


def main():
    print(BANNER)
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"\n  ❌  Error: Not a directory — '{args.input}'\n")
        sys.exit(1)

    batch_scan(
        input_dir=args.input,
        output_dir=args.output,
        mode=args.mode,
        save_pdf=args.pdf,
        workers=args.workers,
    )
    print("\n  Powered by KKR Gen AI Innovations — https://kkrgenaiinnovations.com/\n")


if __name__ == "__main__":
    main()
