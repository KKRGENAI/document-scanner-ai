# =============================================================================
#  KKR Gen AI Innovations — Document Scanner AI
#  Image Enhancement Module
#
#  This module handles:
#    - Grayscale conversion
#    - Adaptive thresholding  → clean black-and-white scan look
#    - Sharpening, brightness, and contrast adjustments
#    - Shadow/background removal
#    - Noise reduction (denoising)
#
#  Website : https://kkrgenaiinnovations.com/
#  Email   : info@kkrgenaiinnovations.com
#  WhatsApp: +1 470-861-6312
# =============================================================================

import cv2
import numpy as np


class ImageEnhancer:
    """
    Post-processing pipeline that turns a warped document image into a
    clean, high-contrast scanner-like output.

    Each method can be used independently, or call `full_enhance()`
    for the recommended defaults.
    """

    # ------------------------------------------------------------------
    # Full pipeline (recommended)
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
        Apply the complete enhancement pipeline.

        Parameters
        ----------
        image         : warped BGR image from DocumentProcessor
        mode          : "adaptive" (default) | "otsu" | "color" | "grayscale"
                        -------------------------------------------------------
                        adaptive  → best for uneven lighting (camera photos)
                        otsu      → clean scans with uniform background
                        color     → keep colors, only sharpen/contrast-boost
                        grayscale → simple grayscale, no thresholding
        sharpen       : apply unsharp masking to make text crisper
        denoise       : remove salt-and-pepper / camera noise first
        remove_shadow : attempt to flatten uneven illumination

        Returns
        -------
        np.ndarray – enhanced image (grayscale or BGR depending on mode)
        """
        img = image.copy()

        # 1. Optional shadow removal (works on BGR image)
        if remove_shadow:
            img = ImageEnhancer.remove_shadow(img)

        # 2. Optional denoising
        if denoise:
            img = ImageEnhancer.denoise(img)

        # 3. Sharpening (before threshold for best results)
        if sharpen:
            img = ImageEnhancer.sharpen(img)

        # 4. Thresholding / mode selection
        if mode == "adaptive":
            img = ImageEnhancer.adaptive_threshold(img)
        elif mode == "otsu":
            img = ImageEnhancer.otsu_threshold(img)
        elif mode == "color":
            img = ImageEnhancer.enhance_color(img)
        elif mode == "grayscale":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose: adaptive | otsu | color | grayscale")

        return img

    # ------------------------------------------------------------------
    # Individual enhancement steps
    # ------------------------------------------------------------------

    @staticmethod
    def adaptive_threshold(image: np.ndarray) -> np.ndarray:
        """
        Convert to grayscale then apply Adaptive Gaussian Thresholding.

        WHY adaptive?
        Normal (global) thresholding uses ONE brightness cutoff for the
        whole image.  If part of the document is in shadow, global
        thresholding will turn the dark area completely black.

        Adaptive thresholding calculates a DIFFERENT threshold for small
        regions of the image — so text in both bright and dark areas
        comes out clean.

        blockSize=11 → neighbourhood size (must be odd)
        C=10         → constant subtracted from the mean (fine-tuning)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        result = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,
            C=10,
        )
        return result

    @staticmethod
    def otsu_threshold(image: np.ndarray) -> np.ndarray:
        """
        Grayscale + Otsu's automatic global threshold.

        Otsu's algorithm finds the OPTIMAL global threshold by minimising
        intra-class variance.  Works brilliantly for clean, evenly-lit
        scans but may struggle with shadows.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result

    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        """
        Sharpen using an Unsharp Mask.

        Formula: sharpened = original + (original − blurred) × amount
        This makes edges pop and text appear crisp.
        """
        # A strong sharpening kernel
        kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0],
        ], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """
        Remove noise using Non-Local Means Denoising.

        This is more powerful than simple Gaussian blur because it
        preserves edges while smoothing out random noise.

        h=10 → filter strength (higher = more smoothing, less detail)
        """
        if image.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10,
                                                   templateWindowSize=7, searchWindowSize=21)
        return cv2.fastNlMeansDenoising(image, None, h=10,
                                        templateWindowSize=7, searchWindowSize=21)

    @staticmethod
    def remove_shadow(image: np.ndarray) -> np.ndarray:
        """
        Attempt to flatten uneven lighting / remove shadow.

        Technique (channel-wise):
          1. Split into channels.
          2. Dilate each channel — this blurs text and keeps the
             background illumination.
          3. Divide the original channel by the background estimate.
             Where background is bright, the result normalises to ~1.
             Where background is dark (shadow), values get boosted.
        This effectively "flattens" the illumination across the image.
        """
        channels = cv2.split(image) if image.ndim == 3 else [image]
        result_channels = []
        for ch in channels:
            bg = cv2.dilate(ch, np.ones((7, 7), np.uint8), iterations=4)
            bg = cv2.GaussianBlur(bg, (21, 21), 0)
            # Normalise: divide original by background, scale to [0,255]
            diff = cv2.divide(ch.astype(np.float32), bg.astype(np.float32) + 1e-6)
            norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            result_channels.append(norm)

        if image.ndim == 3:
            return cv2.merge(result_channels)
        return result_channels[0]

    @staticmethod
    def enhance_color(image: np.ndarray) -> np.ndarray:
        """
        Boost contrast and saturation while keeping the document in colour.

        Uses CLAHE (Contrast Limited Adaptive Histogram Equalisation) on
        the Luminance channel of the LAB colour space — so only brightness
        is adjusted, not hue.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_merged = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def adjust_brightness_contrast(
        image: np.ndarray,
        brightness: int = 0,
        contrast: int = 0,
    ) -> np.ndarray:
        """
        Manual brightness and contrast adjustment.

        Parameters
        ----------
        brightness : int in [-127, 127]  –  positive = brighter
        contrast   : int in [-127, 127]  –  positive = more contrast
        """
        img = image.copy().astype(np.int16)
        # Contrast: scale pixel values around 128
        if contrast != 0:
            factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
            img = factor * (img - 128) + 128
        # Brightness: shift all pixel values
        img = img + brightness
        return np.clip(img, 0, 255).astype(np.uint8)
