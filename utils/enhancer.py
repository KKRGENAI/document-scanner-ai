# =============================================================================
#  KKR Gen AI Innovations — Document Scanner AI
#  Image Enhancement Module
#
#  Website : https://kkrgenaiinnovations.com/
#  Email   : info@kkrgenaiinnovations.com
#  WhatsApp: +1 470-861-6312
# =============================================================================

import cv2
import numpy as np


class ImageEnhancer:
    """
    Post-processing pipeline → clean, high-contrast scanner-like output.

    full_enhance() auto-detects whether the image is a real photo
    (taken with a camera) or a digital document (screenshot, PDF export)
    and applies the right processing for each case.
    """

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def full_enhance(
        image: np.ndarray,
        mode: str = "adaptive",
        sharpen: bool = True,
        denoise: bool = True,
        remove_shadow: bool = True,
    ) -> np.ndarray:
        """
        Full enhancement pipeline.

        Parameters
        ----------
        image         : warped BGR (or grayscale) image
        mode          : "adaptive" | "otsu" | "color" | "grayscale"
        sharpen       : apply gentle unsharp mask
        denoise       : remove noise before thresholding
        remove_shadow : flatten uneven illumination (skip for digital images)

        Auto-behaviour
        --------------
        If the image is already high-contrast (a digital screenshot / PDF),
        shadow-removal and heavy denoising are skipped automatically to
        avoid degrading quality.
        """
        img        = image.copy()
        is_digital = ImageEnhancer._is_digital_image(img)

        # Skip shadow removal and denoising for already-clean digital images
        if remove_shadow and not is_digital:
            img = ImageEnhancer.remove_shadow(img)

        if denoise and not is_digital:
            img = ImageEnhancer.denoise(img)

        if sharpen:
            img = ImageEnhancer.sharpen(img)

        if mode == "adaptive":
            img = ImageEnhancer.adaptive_threshold(img)
        elif mode == "otsu":
            img = ImageEnhancer.otsu_threshold(img)
        elif mode == "color":
            img = ImageEnhancer.enhance_color(img)
        elif mode == "grayscale":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Choose: adaptive | otsu | color | grayscale"
            )

        return img

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------

    @staticmethod
    def adaptive_threshold(image: np.ndarray) -> np.ndarray:
        """
        Adaptive Gaussian Threshold — best for photos with uneven lighting.

        Calculates a separate threshold per small region (blockSize × blockSize)
        so shadows don't turn entire sections solid black.

        blockSize=21 : neighbourhood size (must be odd)
        C=12         : constant subtracted from the local mean
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        return cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=21,
            C=12,
        )

    @staticmethod
    def otsu_threshold(image: np.ndarray) -> np.ndarray:
        """
        Otsu's automatic global threshold.

        Finds the single best threshold by minimising intra-class variance.
        Works best for evenly-lit scans with clear text/background separation.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result

    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        """
        Gentle Unsharp Mask.

        Formula: sharpened = original*(1+a) − blur*a
        amount=0.6 is crisp without the halo/outline artefact that a
        raw convolution kernel produces on already-sharp digital text.
        """
        amount  = 0.6
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2)
        return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)

    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """
        Non-Local Means Denoising — removes camera grain while preserving edges.
        Only applied to real photographs (digital images skip this automatically).

        h=7 : conservative filter strength to avoid over-smoothing text.
        """
        if image.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, None, h=7, hColor=7,
                templateWindowSize=7, searchWindowSize=21
            )
        return cv2.fastNlMeansDenoising(
            image, None, h=7,
            templateWindowSize=7, searchWindowSize=21
        )

    @staticmethod
    def remove_shadow(image: np.ndarray) -> np.ndarray:
        """
        Flatten uneven illumination / remove shadows (photos only).

        Per channel:
          1. Dilate → background brightness estimate
          2. Divide original by background → normalised illumination
        """
        channels       = cv2.split(image) if image.ndim == 3 else [image]
        result_channels = []
        for ch in channels:
            bg   = cv2.dilate(ch, np.ones((7, 7), np.uint8), iterations=4)
            bg   = cv2.GaussianBlur(bg, (21, 21), 0)
            diff = cv2.divide(ch.astype(np.float32), bg.astype(np.float32) + 1e-6)
            norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            result_channels.append(norm)
        if image.ndim == 3:
            return cv2.merge(result_channels)
        return result_channels[0]

    @staticmethod
    def enhance_color(image: np.ndarray) -> np.ndarray:
        """
        Boost contrast while keeping colours via CLAHE on the L channel (LAB space).
        clipLimit=3.0 and tileGridSize=(8,8) work well for most documents.
        """
        lab              = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b          = cv2.split(lab)
        clahe            = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe          = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def adjust_brightness_contrast(
        image: np.ndarray,
        brightness: int = 0,
        contrast: int = 0,
    ) -> np.ndarray:
        """Manual brightness/contrast. Values in [-127, 127]."""
        img = image.copy().astype(np.int16)
        if contrast != 0:
            factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
            img    = factor * (img - 128) + 128
        img = img + brightness
        return np.clip(img, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _is_digital_image(image: np.ndarray) -> bool:
        """
        Heuristic: decide if this image is a clean digital document
        (screenshot, PDF render) rather than a real camera photograph.

        A digital document typically has:
          - Very high contrast  (std-dev of grayscale > 80)
          - Many pure-white pixels (>30 % of area near 255)

        If true → skip shadow removal and heavy denoising.
        """
        gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        std_dev   = float(np.std(gray))
        white_pct = float(np.mean(gray > 230))
        return std_dev > 60 or white_pct > 0.25
