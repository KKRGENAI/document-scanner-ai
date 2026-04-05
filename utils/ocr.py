# =============================================================================
#  KKR Gen AI Innovations — Document Scanner AI
#  OCR Module  (Optical Character Recognition)
#
#  Extracts selectable / copyable text from a scanned document image
#  using Google's Tesseract OCR engine via the pytesseract wrapper.
#
#  REQUIREMENTS
#  ────────────
#  1. pip install pytesseract
#  2. Install Tesseract engine:
#       Windows : https://github.com/UB-Mannheim/tesseract/wiki
#                 (installer — choose "Add to PATH" during setup)
#       Mac     : brew install tesseract
#       Linux   : sudo apt install tesseract-ocr
#
#  Website : https://kkrgenaiinnovations.com/
#  Email   : info@kkrgenaiinnovations.com
#  WhatsApp: +1 470-861-6312
# =============================================================================

from __future__ import annotations

import os
import platform
import cv2
import numpy as np


# Common Tesseract install paths on Windows — checked automatically
_WIN_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(
        os.environ.get("USERNAME", "")
    ),
]


def _find_tesseract_windows() -> str | None:
    """Return the first existing Tesseract path on Windows."""
    for path in _WIN_TESSERACT_PATHS:
        if os.path.isfile(path):
            return path
    return None


def _configure_tesseract():
    """
    Import pytesseract and make sure it can find the Tesseract binary.
    Raises a clear ImportError / RuntimeError with install instructions.
    """
    try:
        import pytesseract
    except ImportError:
        raise ImportError(
            "pytesseract is not installed.\n"
            "  Fix: pip install pytesseract\n"
            "  Then also install the Tesseract engine:\n"
            "    Windows : https://github.com/UB-Mannheim/tesseract/wiki\n"
            "    Mac     : brew install tesseract\n"
            "    Linux   : sudo apt install tesseract-ocr"
        )

    # On Windows, auto-detect the install path if not already on PATH
    if platform.system() == "Windows":
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            win_path = _find_tesseract_windows()
            if win_path:
                pytesseract.pytesseract.tesseract_cmd = win_path
            else:
                raise RuntimeError(
                    "Tesseract OCR engine not found on this Windows machine.\n\n"
                    "Download and install it from:\n"
                    "  https://github.com/UB-Mannheim/tesseract/wiki\n\n"
                    "During installation, tick 'Add to PATH'.\n"
                    "Then restart this application."
                )

    return pytesseract


class OCRProcessor:
    """
    Wraps pytesseract to extract text from scanned document images.

    Usage
    -----
        ocr    = OCRProcessor()
        text   = ocr.extract_text(scanned_image)
        words  = ocr.extract_words(scanned_image)   # with bounding boxes
    """

    def __init__(self):
        self._tess = _configure_tesseract()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_text(self, image: np.ndarray, lang: str = "eng") -> str:
        """
        Extract plain text from the image.

        Parameters
        ----------
        image : scanned image (grayscale or BGR)
        lang  : Tesseract language code (default "eng" = English)
                Other examples: "fra" (French), "deu" (German), "hin" (Hindi)

        Returns
        -------
        str – extracted text, stripped of leading/trailing whitespace
        """
        prepared = self._prepare(image)
        config   = "--oem 3 --psm 6"   # OEM 3 = LSTM; PSM 6 = assume uniform block of text
        text     = self._tess.image_to_string(prepared, lang=lang, config=config)
        return text.strip()

    def extract_words(self, image: np.ndarray, lang: str = "eng") -> list[dict]:
        """
        Extract individual words with their bounding-box coordinates.

        Returns
        -------
        list of dicts:
          { "text": str, "x": int, "y": int, "w": int, "h": int, "conf": float }
        Only words with confidence > 30 are returned.
        """
        prepared = self._prepare(image)
        data     = self._tess.image_to_data(
            prepared, lang=lang, output_type=self._tess.Output.DICT
        )
        words = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            conf = float(data["conf"][i])
            if conf < 30:
                continue
            words.append({
                "text": text,
                "x"   : data["left"][i],
                "y"   : data["top"][i],
                "w"   : data["width"][i],
                "h"   : data["height"][i],
                "conf": conf,
            })
        return words

    def draw_word_boxes(self, image: np.ndarray, lang: str = "eng") -> np.ndarray:
        """
        Return a copy of the image with bounding boxes drawn around each word.
        Useful for debugging OCR quality.
        """
        output = image.copy()
        if output.ndim == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        for w in self.extract_words(image, lang=lang):
            x, y, ww, hh = w["x"], w["y"], w["w"], w["h"]
            cv2.rectangle(output, (x, y), (x + ww, y + hh), (0, 200, 0), 1)
        return output

    def is_available(self) -> bool:
        """Return True if Tesseract is installed and reachable."""
        try:
            self._tess.get_tesseract_version()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare(image: np.ndarray) -> np.ndarray:
        """
        Pre-process image for best OCR accuracy.

        1. Convert to grayscale.
        2. Upscale small images — Tesseract works best at ~300 DPI.
           If the image is narrower than 1000 px, upscale ×2.
        3. Light bilateral filter to smooth noise while keeping edges sharp.
        4. Otsu threshold to get clean binary (black text / white background).
        """
        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

        # Upscale small images
        h, w = gray.shape
        if w < 1000:
            gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        # Light smoothing (preserves edges better than Gaussian)
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Make sure text is dark-on-white (invert if needed)
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        return binary
