# =============================================================================
#  KKR Gen AI Innovations — Document Scanner AI
#  Core Image Processing Module
#
#  This module handles:
#    - Edge detection to find the document boundary
#    - Contour detection to identify the four corners
#    - Perspective transformation (bird's-eye view / flat scan)
#
#  Website : https://kkrgenaiinnovations.com/
#  Email   : info@kkrgenaiinnovations.com
#  WhatsApp: +1 470-861-6312
# =============================================================================

import cv2
import numpy as np


class DocumentProcessor:
    """
    Handles all the heavy-lifting OpenCV work:
      1. Pre-process the image (resize, blur, edge detect)
      2. Find the largest 4-sided contour  →  the document boundary
      3. Order the four corner points
      4. Apply perspective warp to flatten the document
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, image: np.ndarray) -> np.ndarray | None:
        """
        Full pipeline: detect document → warp to flat scan.

        Parameters
        ----------
        image : np.ndarray
            BGR image loaded by cv2.imread()

        Returns
        -------
        np.ndarray or None
            Warped (flattened) BGR image, or None if no document found.
        """
        corners = self.detect_corners(image)
        if corners is None:
            return None
        return self.four_point_transform(image, corners)

    def detect_corners(self, image: np.ndarray) -> np.ndarray | None:
        """
        Detect the four corners of a document in the image.

        Steps (easy to follow!):
          1. Resize to a working width so processing is fast.
          2. Convert to grayscale.
          3. Blur slightly to remove noise.
          4. Run Canny edge detection.
          5. Find all contours and keep the biggest 4-sided one.
          6. Scale the corners back to the original image size.

        Returns
        -------
        np.ndarray shape (4, 2) or None
        """
        orig_h, orig_w = image.shape[:2]

        # --- Step 1 : Resize for faster processing ----------------------
        WORKING_WIDTH = 800
        scale = WORKING_WIDTH / orig_w
        resized = cv2.resize(image, (WORKING_WIDTH, int(orig_h * scale)))

        # --- Step 2 : Grayscale -----------------------------------------
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # --- Step 3 : Gaussian blur (reduces false edges from texture) ---
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # --- Step 4 : Canny edge detection --------------------------------
        #   threshold1=75  – weak edges below this are discarded
        #   threshold2=200 – strong edges above this are always kept
        edges = cv2.Canny(blurred, 75, 200)

        # Dilate edges to close small gaps in the document boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # --- Step 5 : Find contours and pick the biggest rectangle -------
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        # Sort contours by area, largest first — keep top 10 candidates
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        # The document must cover at least 15 % of the working image area.
        # This rejects small inner rectangles (e.g. text highlights, stamps)
        # that often have crisper edges than the page border itself.
        img_area    = resized.shape[0] * resized.shape[1]
        min_area    = 0.15 * img_area

        doc_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                # All remaining contours will be even smaller — stop early
                break

            # Approximate the contour to a polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # We want exactly 4 vertices → quadrilateral (the document)
            if len(approx) == 4:
                doc_contour = approx
                break

        if doc_contour is None:
            return None  # No large-enough document boundary found

        # --- Step 6 : Scale corners back to original resolution ----------
        corners = doc_contour.reshape(4, 2).astype(np.float32)
        corners /= scale          # undo the resize scaling
        return corners

    @staticmethod
    def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Apply a perspective warp so the document appears flat
        (as if photographed from directly above — bird's-eye view).

        How it works
        ------------
        Given four corner points of the document (in any order),
        we compute the transformation matrix M that maps them onto
        a perfect rectangle of the same dimensions.

        Parameters
        ----------
        image : np.ndarray  – original BGR image
        pts   : np.ndarray  – four corner points, shape (4, 2)

        Returns
        -------
        np.ndarray – warped (flattened) image
        """
        rect = DocumentProcessor._order_points(pts)
        (tl, tr, br, bl) = rect   # top-left, top-right, bottom-right, bottom-left

        # Calculate the width of the output image
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        max_width = int(max(widthA, widthB))

        # Calculate the height of the output image
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        max_height = int(max(heightA, heightB))

        # Destination points — a perfect rectangle
        dst = np.array([
            [0,             0            ],
            [max_width - 1, 0            ],
            [max_width - 1, max_height - 1],
            [0,             max_height - 1],
        ], dtype=np.float32)

        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        return warped

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """
        Sort four points into a consistent order:
            [top-left, top-right, bottom-right, bottom-left]

        Trick:
          - top-left     → smallest (x + y) sum
          - bottom-right → largest  (x + y) sum
          - top-right    → smallest (y - x) difference
          - bottom-left  → largest  (y - x) difference
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left
        rect[2] = pts[np.argmax(s)]   # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    @staticmethod
    def draw_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Draw detected document corners on a COPY of the image.
        Useful for debugging / showing the student what was detected.
        """
        output = image.copy()
        pts = corners.reshape(4, 2).astype(int)
        cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        for i, (x, y) in enumerate(pts):
            cv2.circle(output, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(
                output, f"P{i+1}", (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )
        return output
