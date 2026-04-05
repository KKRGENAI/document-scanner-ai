# =============================================================================
#  KKR Gen AI Innovations — Document Scanner AI
#  PDF Converter Module
#
#  Converts one or more scanned images into a multi-page PDF.
#  Uses only Pillow (PIL) — no heavyweight PDF library required.
#
#  Website : https://kkrgenaiinnovations.com/
#  Email   : info@kkrgenaiinnovations.com
#  WhatsApp: +1 470-861-6312
# =============================================================================

from __future__ import annotations

import os
import cv2
import numpy as np
from pathlib import Path


class PDFConverter:
    """
    Save scanned document images as PDF files.

    Requires Pillow:  pip install Pillow
    """

    @staticmethod
    def image_to_pdf(image: np.ndarray, output_path: str) -> str:
        """
        Save a single OpenCV image (numpy array) as a PDF.

        Parameters
        ----------
        image       : np.ndarray  – grayscale or BGR image
        output_path : str         – e.g. "output/scan.pdf"

        Returns
        -------
        str – absolute path to the saved PDF
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Pillow is required for PDF export.\n"
                "Install it with:  pip install Pillow"
            )

        pil_img = PDFConverter._cv2_to_pil(image)
        output_path = str(Path(output_path).with_suffix(".pdf"))
        pil_img.save(output_path, "PDF", resolution=200)
        return os.path.abspath(output_path)

    @staticmethod
    def images_to_pdf(images: list[np.ndarray], output_path: str) -> str:
        """
        Save a list of scanned images as a single multi-page PDF.

        Parameters
        ----------
        images      : list of np.ndarray  – one per page
        output_path : str                 – e.g. "output/batch.pdf"

        Returns
        -------
        str – absolute path to the saved PDF
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Pillow is required for PDF export.\n"
                "Install it with:  pip install Pillow"
            )

        if not images:
            raise ValueError("images list is empty — nothing to save.")

        pil_images = [PDFConverter._cv2_to_pil(img) for img in images]
        first, rest = pil_images[0], pil_images[1:]

        output_path = str(Path(output_path).with_suffix(".pdf"))
        first.save(output_path, "PDF", resolution=200, save_all=True, append_images=rest)
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _cv2_to_pil(image: np.ndarray):
        """Convert an OpenCV numpy array to a Pillow Image (RGB or L)."""
        from PIL import Image  # local import keeps module importable without Pillow

        if image.ndim == 2:
            # Grayscale
            return Image.fromarray(image, mode="L")
        # BGR → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb, mode="RGB")
