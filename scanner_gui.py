#!/usr/bin/env python3
# =============================================================================
#   KKR Gen AI Innovations — Document Scanner AI
#   Tkinter Desktop GUI
#
#   A beginner-friendly graphical interface to scan documents
#   without typing any commands.
#
#   Website : https://kkrgenaiinnovations.com/
#   Email   : info@kkrgenaiinnovations.com
#   WhatsApp: +1 470-861-6312
# =============================================================================
"""
scanner_gui.py
--------------
Launch the desktop GUI:

  python scanner_gui.py

Features
--------
  • Browse and load an image
  • Live preview of original and scanned result side-by-side
  • Select enhancement mode from a dropdown
  • Save as JPG or PDF with one click
  • Status bar with progress messages
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np

from utils.image_processor import DocumentProcessor
from utils.enhancer import ImageEnhancer
from utils.pdf_converter import PDFConverter


# ===========================================================================
#  Utility: convert OpenCV image → Tkinter PhotoImage
# ===========================================================================

def cv2_to_tk(image: np.ndarray, max_w: int = 520, max_h: int = 640):
    """Resize and convert a numpy BGR/gray image for display in a Label."""
    try:
        from PIL import Image, ImageTk
    except ImportError:
        raise ImportError("Pillow is needed for the GUI.\n  pip install Pillow")

    if image.ndim == 2:
        pil_img = Image.fromarray(image, mode="L").convert("RGB")
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

    # Fit inside max_w × max_h while keeping aspect ratio
    pil_img.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(pil_img)


# ===========================================================================
#  Main Application Window
# ===========================================================================

class DocumentScannerApp(tk.Tk):

    # Brand colours
    PRIMARY   = "#1a1a2e"   # dark navy
    SECONDARY = "#16213e"   # slightly lighter navy
    ACCENT    = "#0f3460"   # blue
    HIGHLIGHT = "#e94560"   # red-pink
    TEXT      = "#eaeaea"
    SUCCESS   = "#4caf50"
    WARNING   = "#ff9800"

    def __init__(self):
        super().__init__()
        self.title("Document Scanner AI — KKR Gen AI Innovations")
        self.configure(bg=self.PRIMARY)
        self.resizable(True, True)
        self.minsize(1100, 700)

        # State
        self._orig_image: np.ndarray | None = None
        self._scanned_image: np.ndarray | None = None
        self._orig_tk  = None   # keep references (prevent GC)
        self._scan_tk  = None

        self._build_ui()
        self._center_window()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Top banner ──────────────────────────────────────────────────
        banner_frame = tk.Frame(self, bg=self.ACCENT, pady=8)
        banner_frame.pack(fill="x")
        tk.Label(
            banner_frame,
            text="  📄  KKR Gen AI Innovations — Document Scanner AI",
            font=("Segoe UI", 14, "bold"),
            bg=self.ACCENT, fg=self.TEXT,
        ).pack(side="left", padx=16)
        tk.Label(
            banner_frame,
            text="https://kkrgenaiinnovations.com/  ",
            font=("Segoe UI", 9),
            bg=self.ACCENT, fg="#aad4f5",
        ).pack(side="right", padx=16)

        # ── Main content ────────────────────────────────────────────────
        content = tk.Frame(self, bg=self.PRIMARY, padx=12, pady=10)
        content.pack(fill="both", expand=True)

        # Left column: controls
        left = tk.Frame(content, bg=self.SECONDARY, width=280, padx=14, pady=14)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        self._build_controls(left)

        # Right column: preview panels
        right = tk.Frame(content, bg=self.PRIMARY)
        right.pack(side="left", fill="both", expand=True)
        self._build_previews(right)

        # ── Status bar ──────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Ready. Load an image to begin.")
        status_bar = tk.Label(
            self, textvariable=self._status_var,
            font=("Segoe UI", 9), bg=self.SECONDARY,
            fg=self.TEXT, anchor="w", padx=12, pady=5,
        )
        status_bar.pack(fill="x", side="bottom")

    def _build_controls(self, parent):
        lbl_style = {"bg": self.SECONDARY, "fg": self.TEXT, "font": ("Segoe UI", 10)}
        sep_style = {"bg": self.ACCENT, "height": 1}

        # Section: Load
        tk.Label(parent, text="1.  LOAD IMAGE", font=("Segoe UI", 10, "bold"),
                 bg=self.SECONDARY, fg=self.HIGHLIGHT).pack(anchor="w", pady=(0, 4))
        tk.Button(
            parent, text="📂  Browse …",
            font=("Segoe UI", 10, "bold"),
            bg=self.HIGHLIGHT, fg="white", relief="flat",
            activebackground="#c73652", activeforeground="white",
            padx=10, pady=6, cursor="hand2",
            command=self._load_image,
        ).pack(fill="x", pady=(0, 4))
        self._file_lbl = tk.Label(parent, text="No file selected",
                                  wraplength=230, justify="left",
                                  font=("Segoe UI", 8), **lbl_style)
        self._file_lbl.pack(anchor="w", pady=(0, 8))
        tk.Frame(parent, **sep_style).pack(fill="x", pady=6)

        # Section: Scan settings
        tk.Label(parent, text="2.  SCAN SETTINGS", font=("Segoe UI", 10, "bold"),
                 bg=self.SECONDARY, fg=self.HIGHLIGHT).pack(anchor="w", pady=(0, 4))

        tk.Label(parent, text="Enhancement Mode:", **lbl_style).pack(anchor="w")
        self._mode_var = tk.StringVar(value="adaptive")
        mode_box = ttk.Combobox(
            parent, textvariable=self._mode_var,
            values=["adaptive", "otsu", "color", "grayscale"],
            state="readonly", font=("Segoe UI", 10),
        )
        mode_box.pack(fill="x", pady=(2, 6))

        self._denoise_var  = tk.BooleanVar(value=True)
        self._shadow_var   = tk.BooleanVar(value=True)
        self._sharpen_var  = tk.BooleanVar(value=True)

        for var, text in [
            (self._denoise_var,  "Reduce noise"),
            (self._shadow_var,   "Remove shadows"),
            (self._sharpen_var,  "Sharpen text"),
        ]:
            tk.Checkbutton(
                parent, text=text, variable=var,
                bg=self.SECONDARY, fg=self.TEXT,
                selectcolor=self.ACCENT, activebackground=self.SECONDARY,
                font=("Segoe UI", 9),
            ).pack(anchor="w")

        tk.Frame(parent, **sep_style).pack(fill="x", pady=8)

        # Section: Scan
        tk.Label(parent, text="3.  SCAN", font=("Segoe UI", 10, "bold"),
                 bg=self.SECONDARY, fg=self.HIGHLIGHT).pack(anchor="w", pady=(0, 4))
        tk.Button(
            parent, text="▶  Scan Document",
            font=("Segoe UI", 11, "bold"),
            bg=self.SUCCESS, fg="white", relief="flat",
            activebackground="#388e3c", activeforeground="white",
            padx=10, pady=8, cursor="hand2",
            command=self._scan_threaded,
        ).pack(fill="x", pady=(0, 8))
        tk.Frame(parent, **sep_style).pack(fill="x", pady=6)

        # Section: Save
        tk.Label(parent, text="4.  SAVE RESULT", font=("Segoe UI", 10, "bold"),
                 bg=self.SECONDARY, fg=self.HIGHLIGHT).pack(anchor="w", pady=(0, 4))
        tk.Button(
            parent, text="💾  Save as JPG",
            font=("Segoe UI", 10),
            bg=self.ACCENT, fg="white", relief="flat",
            activebackground="#0a2540", activeforeground="white",
            padx=10, pady=6, cursor="hand2",
            command=lambda: self._save("jpg"),
        ).pack(fill="x", pady=(0, 4))
        tk.Button(
            parent, text="📄  Save as PDF",
            font=("Segoe UI", 10),
            bg=self.ACCENT, fg="white", relief="flat",
            activebackground="#0a2540", activeforeground="white",
            padx=10, pady=6, cursor="hand2",
            command=lambda: self._save("pdf"),
        ).pack(fill="x", pady=(0, 8))
        tk.Frame(parent, **sep_style).pack(fill="x", pady=6)

        # Footer links
        tk.Label(parent, text="KKR Gen AI Innovations",
                 font=("Segoe UI", 8, "bold"),
                 bg=self.SECONDARY, fg=self.HIGHLIGHT).pack(anchor="w", pady=(4, 0))
        tk.Label(parent, text="kkrgenaiinnovations.com",
                 font=("Segoe UI", 8), bg=self.SECONDARY, fg="#aad4f5").pack(anchor="w")
        tk.Label(parent, text="WA: +1 470-861-6312",
                 font=("Segoe UI", 8), bg=self.SECONDARY, fg="#aad4f5").pack(anchor="w")

    def _build_previews(self, parent):
        row1 = tk.Frame(parent, bg=self.PRIMARY)
        row1.pack(fill="both", expand=True)

        # Original preview
        orig_frame = tk.LabelFrame(
            row1, text="  Original Image  ",
            font=("Segoe UI", 10, "bold"),
            bg=self.PRIMARY, fg=self.TEXT,
            bd=2, relief="ridge",
        )
        orig_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self._orig_panel = tk.Label(orig_frame, bg=self.SECONDARY,
                                    text="Load an image to preview",
                                    fg="#888", font=("Segoe UI", 11))
        self._orig_panel.pack(fill="both", expand=True, padx=4, pady=4)

        # Scanned preview
        scan_frame = tk.LabelFrame(
            row1, text="  Scanned Output  ",
            font=("Segoe UI", 10, "bold"),
            bg=self.PRIMARY, fg=self.TEXT,
            bd=2, relief="ridge",
        )
        scan_frame.pack(side="left", fill="both", expand=True)
        self._scan_panel = tk.Label(scan_frame, bg=self.SECONDARY,
                                    text="Scan result will appear here",
                                    fg="#888", font=("Segoe UI", 11))
        self._scan_panel.pack(fill="both", expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Open Document Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", f"Cannot open:\n{path}")
            return

        self._orig_image = image
        self._scanned_image = None
        self._file_lbl.config(text=os.path.basename(path))
        self._set_status(f"Loaded: {os.path.basename(path)}")

        # Show original preview
        self._orig_tk = cv2_to_tk(image)
        self._orig_panel.config(image=self._orig_tk, text="")
        self._scan_panel.config(image="", text="Press ▶ Scan Document")

    def _scan_threaded(self):
        """Run scanning in a background thread so the UI stays responsive."""
        if self._orig_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        self._set_status("Scanning … please wait")
        threading.Thread(target=self._do_scan, daemon=True).start()

    def _do_scan(self):
        try:
            processor = DocumentProcessor()
            enhancer  = ImageEnhancer()

            corners = processor.detect_corners(self._orig_image)
            if corners is not None:
                scanned = processor.four_point_transform(self._orig_image, corners)
            else:
                scanned = self._orig_image.copy()

            enhanced = enhancer.full_enhance(
                scanned,
                mode=self._mode_var.get(),
                sharpen=self._sharpen_var.get(),
                denoise=self._denoise_var.get(),
                remove_shadow=self._shadow_var.get(),
            )
            self._scanned_image = enhanced
            # Update UI on main thread
            self.after(0, self._show_scan_result)
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Scan Error", str(exc)))
            self.after(0, lambda: self._set_status(f"Error: {exc}"))

    def _show_scan_result(self):
        self._scan_tk = cv2_to_tk(self._scanned_image)
        self._scan_panel.config(image=self._scan_tk, text="")
        self._set_status("Scan complete!  Save using the buttons on the left.")

    def _save(self, fmt: str):
        if self._scanned_image is None:
            messagebox.showwarning("Nothing to Save", "Please scan a document first.")
            return

        if fmt == "jpg":
            path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")],
                title="Save Scanned Image",
            )
            if path:
                cv2.imwrite(path, self._scanned_image)
                self._set_status(f"Saved: {os.path.basename(path)}")
                messagebox.showinfo("Saved", f"Image saved:\n{path}")
        else:
            path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF Document", "*.pdf")],
                title="Save as PDF",
            )
            if path:
                PDFConverter.image_to_pdf(self._scanned_image, path)
                self._set_status(f"PDF saved: {os.path.basename(path)}")
                messagebox.showinfo("Saved", f"PDF saved:\n{path}")

    def _set_status(self, msg: str):
        self._status_var.set(f"  {msg}")

    def _center_window(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")


# ===========================================================================
#  Entry point
# ===========================================================================

def main():
    app = DocumentScannerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
