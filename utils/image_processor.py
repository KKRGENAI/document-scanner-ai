# =============================================================================
#  KKR Gen AI Innovations — Document Scanner AI
#  Core Image Processing Module
#
#  Website : https://kkrgenaiinnovations.com/
#  Email   : info@kkrgenaiinnovations.com
#  WhatsApp: +1 470-861-6312
# =============================================================================

import cv2
import numpy as np


class DocumentProcessor:
    """
    Handles all OpenCV operations:
      1. Pre-process (resize, blur, edge-detect)
      2. Find the largest 4-sided contour that is the document boundary
      3. Order the four corner points
      4. Perspective-warp to a flat bird's-eye view
    """

    WORKING_WIDTH = 800   # resize to this width before processing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, image: np.ndarray) -> np.ndarray:
        """Full pipeline: detect → warp. Falls back to full image."""
        corners = self.detect_corners(image)
        if corners is None:
            return image.copy()
        return self.four_point_transform(image, corners)

    def detect_corners(self, image: np.ndarray) -> np.ndarray | None:
        """
        Detect the four corner points of a document in the image.

        Tries two strategies:
          1. Canny edge → contour (works well for real photos)
          2. Thresholded white-region detection (works well for screenshots)

        Returns np.ndarray shape (4, 2) or None.
        """
        orig_h, orig_w = image.shape[:2]
        scale = self.WORKING_WIDTH / orig_w
        resized = cv2.resize(image, (self.WORKING_WIDTH, int(orig_h * scale)))
        img_area = resized.shape[0] * resized.shape[1]

        # Strategy 1 — Canny edges (good for camera photos)
        corners = self._canny_strategy(resized, img_area)

        # Strategy 2 — Bright-region threshold (good for screenshots)
        if corners is None:
            corners = self._white_region_strategy(resized, img_area)

        if corners is None:
            return None

        # Scale corners back to original resolution
        corners = corners.reshape(4, 2).astype(np.float32)
        corners /= scale
        return corners

    # ------------------------------------------------------------------
    # Detection strategies
    # ------------------------------------------------------------------

    def _canny_strategy(self, resized: np.ndarray, img_area: int) -> np.ndarray | None:
        """Edge-based contour detection (standard approach)."""
        gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Try two Canny thresholds — stricter first, looser as fallback
        for lo, hi in [(75, 200), (30, 100)]:
            edges  = cv2.Canny(blurred, lo, hi)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges  = cv2.dilate(edges, kernel, iterations=1)

            contours, _ = cv2.findContours(
                edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

            # The document contour must cover ≥ 15 % of the image.
            # This rejects small inner rectangles (text highlights, etc.)
            min_area = 0.15 * img_area

            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    break   # remaining are all smaller
                peri   = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:
                    return approx

        return None

    def _white_region_strategy(self, resized: np.ndarray, img_area: int) -> np.ndarray | None:
        """
        Find the largest bright (document-coloured) region.
        Useful for screenshots where the page is a white/light rectangle
        on a dark app background.
        """
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Threshold: pixels brighter than 200 → white (document area)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Close small holes (text on the page)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Pick largest bright blob
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 0.10 * img_area:
            return None   # blob too small

        peri   = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
        if len(approx) == 4:
            return approx

        # Fallback: use the bounding rectangle of the blob as 4 corners
        x, y, w, h = cv2.boundingRect(largest)
        return np.array([
            [[x,     y    ]],
            [[x + w, y    ]],
            [[x + w, y + h]],
            [[x,     y + h]],
        ], dtype=np.int32)

    # ------------------------------------------------------------------
    # Perspective transform
    # ------------------------------------------------------------------

    @staticmethod
    def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Warp the document to a flat rectangle (bird's-eye view).

        Given four corner points in any order, compute the 3×3 perspective
        matrix M and apply it so the document fills the output image.
        """
        rect = DocumentProcessor._order_points(pts)
        tl, tr, br, bl = rect

        widthA    = np.linalg.norm(br - bl)
        widthB    = np.linalg.norm(tr - tl)
        max_width = int(max(widthA, widthB))

        heightA    = np.linalg.norm(tr - br)
        heightB    = np.linalg.norm(tl - bl)
        max_height = int(max(heightA, heightB))

        dst = np.array([
            [0,             0            ],
            [max_width - 1, 0            ],
            [max_width - 1, max_height - 1],
            [0,             max_height - 1],
        ], dtype=np.float32)

        M      = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        return warped

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Sort four points → [top-left, top-right, bottom-right, bottom-left]."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s    = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left     (smallest x+y)
        rect[2] = pts[np.argmax(s)]   # bottom-right (largest  x+y)
        diff    = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    @staticmethod
    def draw_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Draw detected document outline on a copy (for debug/preview)."""
        output = image.copy()
        pts    = corners.reshape(4, 2).astype(int)
        cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        for i, (x, y) in enumerate(pts):
            cv2.circle(output, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(output, f"P{i+1}", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return output
