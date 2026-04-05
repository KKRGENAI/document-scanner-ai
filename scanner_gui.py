#!/usr/bin/env python3
# =============================================================================
#   KKR Gen AI Innovations — Document Scanner AI
#   Tkinter Desktop GUI  (v2 — with OCR + Word export)
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
  • Load any image via Browse or Drag-and-Drop
  • Side-by-side preview: Original | Scanned
  • Enhancement mode selector (adaptive / otsu / color / grayscale)
  • Save as JPG, PDF, or Word (.docx)
  • OCR: Extract selectable text from the scanned document
  • Copy OCR text to clipboard with one click
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
from utils.word_converter import WordConverter


# ===========================================================================
#  Helper: OpenCV image → Tkinter PhotoImage
# ===========================================================================

def cv2_to_tk(image: np.ndarray, max_w: int = 500, max_h: int = 560):
    try:
        from PIL import Image, ImageTk
    except ImportError:
        raise ImportError("Pillow is needed for the GUI.\n  pip install Pillow")
    if image.ndim == 2:
        pil = Image.fromarray(image, mode="L").convert("RGB")
    else:
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(pil)


# ===========================================================================
#  Main Application
# ===========================================================================

class DocumentScannerApp(tk.Tk):

    # Brand colours
    BG       = "#0f1117"
    PANEL    = "#1e2235"
    SIDEBAR  = "#16213e"
    ACCENT   = "#0f3460"
    RED      = "#e94560"
    TEXT     = "#eaeaea"
    MUTED    = "#a0aec0"
    SUCCESS  = "#4caf50"
    WARNING  = "#f6ad55"

    def __init__(self):
        super().__init__()
        self.title("Document Scanner AI — KKR Gen AI Innovations")
        self.configure(bg=self.BG)
        self.minsize(1200, 780)
        self.resizable(True, True)

        # State
        self._orig_image:    np.ndarray | None = None
        self._scanned_image: np.ndarray | None = None
        self._ocr_text: str = ""
        self._orig_tk  = None
        self._scan_tk  = None

        self._build_ui()
        self._center()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Top navbar ─────────────────────────────────────────────────
        nav = tk.Frame(self, bg=self.ACCENT, pady=9)
        nav.pack(fill="x")
        tk.Label(nav, text="  📄  KKR Gen AI Innovations — Document Scanner AI",
                 font=("Segoe UI", 13, "bold"), bg=self.ACCENT, fg=self.TEXT
                 ).pack(side="left", padx=16)
        tk.Label(nav, text="kkrgenaiinnovations.com  ",
                 font=("Segoe UI", 9), bg=self.ACCENT, fg="#aad4f5"
                 ).pack(side="right", padx=14)

        # ── Main content (sidebar + workspace) ─────────────────────────
        body = tk.Frame(self, bg=self.BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # Left sidebar
        sidebar = tk.Frame(body, bg=self.SIDEBAR, width=260)
        sidebar.pack(side="left", fill="y", padx=(0, 8))
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        # Right workspace
        workspace = tk.Frame(body, bg=self.BG)
        workspace.pack(side="left", fill="both", expand=True)
        self._build_workspace(workspace)

        # ── Status bar ─────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Ready — load an image to begin.")
        tk.Label(self, textvariable=self._status_var,
                 font=("Segoe UI", 9), bg=self.PANEL,
                 fg=self.TEXT, anchor="w", padx=12, pady=5
                 ).pack(fill="x", side="bottom")

    # ── Sidebar ────────────────────────────────────────────────────────

    def _build_sidebar(self, parent):
        pad = {"padx": 12, "pady": 4}

        def section(text):
            tk.Label(parent, text=text,
                     font=("Segoe UI", 9, "bold"),
                     bg=self.SIDEBAR, fg=self.RED
                     ).pack(anchor="w", padx=12, pady=(10, 2))
            tk.Frame(parent, bg=self.ACCENT, height=1).pack(fill="x", padx=12, pady=(0, 6))

        def btn(parent, text, color, command, pady=4):
            tk.Button(parent, text=text,
                      font=("Segoe UI", 9, "bold"),
                      bg=color, fg="white", relief="flat",
                      activebackground=color, activeforeground="white",
                      padx=8, pady=6, cursor="hand2",
                      command=command
                      ).pack(fill="x", padx=12, pady=(0, pady))

        # ── 1. LOAD ─────────────────────────────────────────────────────
        section("1.  LOAD IMAGE")
        btn(parent, "📂  Browse …", self.RED, self._load_image)
        self._file_lbl = tk.Label(parent, text="No file selected",
                                  wraplength=220, justify="left",
                                  font=("Segoe UI", 8),
                                  bg=self.SIDEBAR, fg=self.MUTED)
        self._file_lbl.pack(anchor="w", padx=12, pady=(0, 2))

        # ── 2. SCAN SETTINGS ────────────────────────────────────────────
        section("2.  SCAN SETTINGS")
        tk.Label(parent, text="Enhancement Mode",
                 font=("Segoe UI", 8), bg=self.SIDEBAR, fg=self.MUTED
                 ).pack(anchor="w", padx=12)
        self._mode_var = tk.StringVar(value="adaptive")
        ttk.Combobox(parent, textvariable=self._mode_var,
                     values=["adaptive", "otsu", "color", "grayscale"],
                     state="readonly", font=("Segoe UI", 9)
                     ).pack(fill="x", padx=12, pady=(2, 6))

        self._denoise_var = tk.BooleanVar(value=True)
        self._shadow_var  = tk.BooleanVar(value=True)
        self._sharpen_var = tk.BooleanVar(value=True)
        for var, label in [
            (self._denoise_var, "Reduce noise"),
            (self._shadow_var,  "Remove shadows"),
            (self._sharpen_var, "Sharpen text"),
        ]:
            tk.Checkbutton(parent, text=label, variable=var,
                           bg=self.SIDEBAR, fg=self.TEXT,
                           selectcolor=self.ACCENT,
                           activebackground=self.SIDEBAR,
                           font=("Segoe UI", 9)
                           ).pack(anchor="w", padx=12)

        # ── 3. SCAN ─────────────────────────────────────────────────────
        section("3.  SCAN")
        btn(parent, "▶  Scan Document", self.SUCCESS, self._scan_threaded, pady=2)
        self._corners_lbl = tk.Label(parent, text="",
                                     font=("Segoe UI", 8),
                                     bg=self.SIDEBAR, fg=self.MUTED)
        self._corners_lbl.pack(anchor="w", padx=12, pady=(0, 4))

        # ── 4. OCR ──────────────────────────────────────────────────────
        section("4.  EXTRACT TEXT (OCR)")
        btn(parent, "🔍  Extract Text (OCR)", "#7b2ff7", self._ocr_threaded)
        btn(parent, "📋  Copy Text to Clipboard", self.ACCENT, self._copy_text, pady=8)

        # ── 5. SAVE ─────────────────────────────────────────────────────
        section("5.  SAVE RESULT")
        btn(parent, "💾  Save as JPG",  self.ACCENT,   lambda: self._save("jpg"))
        btn(parent, "📄  Save as PDF",  "#c73652",     lambda: self._save("pdf"))
        btn(parent, "📝  Save as Word", "#1565c0",     lambda: self._save("word"), pady=8)

        # ── Branding footer ─────────────────────────────────────────────
        tk.Frame(parent, bg=self.ACCENT, height=1).pack(fill="x", padx=12, pady=(8, 4))
        for line in [
            ("KKR Gen AI Innovations", ("Segoe UI", 8, "bold"), self.RED),
            ("kkrgenaiinnovations.com", ("Segoe UI", 8), "#aad4f5"),
            ("WA: +1 470-861-6312",    ("Segoe UI", 8), "#aad4f5"),
        ]:
            tk.Label(parent, text=line[0], font=line[1],
                     bg=self.SIDEBAR, fg=line[2]
                     ).pack(anchor="w", padx=12, pady=1)

    # ── Workspace ──────────────────────────────────────────────────────

    def _build_workspace(self, parent):
        # Top row: image previews
        top = tk.Frame(parent, bg=self.BG)
        top.pack(fill="both", expand=True)

        for attr, title, placeholder in [
            ("_orig_panel",  "Original Image",  "Load an image to preview"),
            ("_scan_panel",  "Scanned Output",  "Scan result will appear here"),
        ]:
            frame = tk.LabelFrame(top, text=f"  {title}  ",
                                  font=("Segoe UI", 10, "bold"),
                                  bg=self.BG, fg=self.TEXT, bd=2, relief="ridge")
            frame.pack(side="left", fill="both", expand=True, padx=(0, 6) if attr == "_orig_panel" else 0)
            lbl = tk.Label(frame, text=placeholder,
                           bg=self.PANEL, fg="#555",
                           font=("Segoe UI", 10))
            lbl.pack(fill="both", expand=True, padx=3, pady=3)
            setattr(self, attr, lbl)

        # Bottom: OCR text area
        ocr_frame = tk.LabelFrame(parent, text="  Extracted Text (OCR — Selectable & Copyable)  ",
                                  font=("Segoe UI", 10, "bold"),
                                  bg=self.BG, fg=self.TEXT, bd=2, relief="ridge")
        ocr_frame.pack(fill="x", pady=(8, 0), ipady=4)

        scroll = tk.Scrollbar(ocr_frame, orient="vertical")
        self._ocr_box = tk.Text(ocr_frame,
                                height=7,
                                font=("Courier New", 10),
                                bg="#0d1117", fg="#c9d1d9",
                                insertbackground=self.TEXT,
                                selectbackground=self.ACCENT,
                                relief="flat", padx=10, pady=8,
                                wrap="word",
                                yscrollcommand=scroll.set)
        scroll.config(command=self._ocr_box.yview)
        scroll.pack(side="right", fill="y")
        self._ocr_box.pack(fill="both", expand=True, padx=(3, 0), pady=3)
        self._ocr_box.insert("1.0", "[ OCR text will appear here after you click 'Extract Text (OCR)' ]")
        self._ocr_box.config(state="disabled")

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
        self._orig_image    = image
        self._scanned_image = None
        self._ocr_text      = ""
        self._file_lbl.config(text=os.path.basename(path))
        self._corners_lbl.config(text="")
        self._set_status(f"Loaded: {os.path.basename(path)}")

        self._orig_tk = cv2_to_tk(image)
        self._orig_panel.config(image=self._orig_tk, text="")
        self._scan_panel.config(image="", text="Press ▶ Scan Document")
        self._clear_ocr_box("[ OCR text will appear here after scanning and clicking Extract Text ]")

    # ── Scan ────────────────────────────────────────────────────────────

    def _scan_threaded(self):
        if self._orig_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        self._set_status("Scanning … please wait")
        self._scan_panel.config(image="", text="Scanning …")
        threading.Thread(target=self._do_scan, daemon=True).start()

    def _do_scan(self):
        try:
            processor = DocumentProcessor()
            enhancer  = ImageEnhancer()

            corners = processor.detect_corners(self._orig_image)
            if corners is not None:
                scanned  = processor.four_point_transform(self._orig_image, corners)
                detected = True
            else:
                scanned  = self._orig_image.copy()
                detected = False

            enhanced = enhancer.full_enhance(
                scanned,
                mode=self._mode_var.get(),
                sharpen=self._sharpen_var.get(),
                denoise=self._denoise_var.get(),
                remove_shadow=self._shadow_var.get(),
            )
            self._scanned_image = enhanced
            self.after(0, lambda d=detected: self._show_scan(d))
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Scan Error", str(exc)))
            self.after(0, lambda: self._set_status(f"Error: {exc}"))

    def _show_scan(self, corners_detected: bool):
        self._scan_tk = cv2_to_tk(self._scanned_image)
        self._scan_panel.config(image=self._scan_tk, text="")
        icon = "✅" if corners_detected else "⚠️"
        msg  = "Corners detected" if corners_detected else "No boundary found — full image used"
        self._corners_lbl.config(text=f"{icon} {msg}",
                                 fg=self.SUCCESS if corners_detected else self.WARNING)
        self._set_status("Scan complete! Extract text or save using the sidebar buttons.")

    # ── OCR ─────────────────────────────────────────────────────────────

    def _ocr_threaded(self):
        if self._scanned_image is None:
            messagebox.showwarning("No Scan", "Please scan a document first.")
            return
        self._set_status("Running OCR … please wait")
        self._clear_ocr_box("Running OCR …")
        threading.Thread(target=self._do_ocr, daemon=True).start()

    def _do_ocr(self):
        try:
            from utils.ocr import OCRProcessor
            ocr  = OCRProcessor()
            text = ocr.extract_text(self._scanned_image)
            self._ocr_text = text
            self.after(0, lambda: self._show_ocr(text))
        except (ImportError, RuntimeError) as exc:
            msg = str(exc)
            self.after(0, lambda: self._clear_ocr_box(f"⚠️  OCR unavailable:\n\n{msg}"))
            self.after(0, lambda: self._set_status("OCR unavailable — see text panel for details"))
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("OCR Error", str(exc)))
            self.after(0, lambda: self._set_status(f"OCR error: {exc}"))

    def _show_ocr(self, text: str):
        self._ocr_box.config(state="normal")
        self._ocr_box.delete("1.0", "end")
        if text:
            self._ocr_box.insert("1.0", text)
            self._set_status(f"OCR complete — {len(text.split())} words extracted. Text is now selectable and copyable.")
        else:
            self._ocr_box.insert("1.0", "[ No text detected. Try a different enhancement mode (e.g. 'otsu') before scanning. ]")
            self._set_status("OCR found no text. Try mode 'otsu' for better results.")
        self._ocr_box.config(state="normal")   # keep editable so user can select/copy

    def _copy_text(self):
        text = self._ocr_box.get("1.0", "end").strip()
        if not text or text.startswith("["):
            messagebox.showinfo("Nothing to Copy", "Run OCR first to extract text.")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self._set_status("OCR text copied to clipboard.")

    def _clear_ocr_box(self, placeholder: str = ""):
        self._ocr_box.config(state="normal")
        self._ocr_box.delete("1.0", "end")
        if placeholder:
            self._ocr_box.insert("1.0", placeholder)
        self._ocr_box.config(state="normal")

    # ── Save ────────────────────────────────────────────────────────────

    def _save(self, fmt: str):
        if self._scanned_image is None:
            messagebox.showwarning("Nothing to Save", "Please scan a document first.")
            return

        if fmt == "jpg":
            path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
                title="Save Scanned Image",
            )
            if path:
                cv2.imwrite(path, self._scanned_image)
                self._set_status(f"Saved: {os.path.basename(path)}")
                messagebox.showinfo("Saved", f"Image saved:\n{path}")

        elif fmt == "pdf":
            path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF", "*.pdf")],
                title="Save as PDF",
            )
            if path:
                PDFConverter.image_to_pdf(self._scanned_image, path)
                self._set_status(f"PDF saved: {os.path.basename(path)}")
                messagebox.showinfo("Saved", f"PDF saved:\n{path}")

        elif fmt == "word":
            path = filedialog.asksaveasfilename(
                defaultextension=".docx",
                filetypes=[("Word Document", "*.docx")],
                title="Save as Word Document",
            )
            if path:
                # Get current OCR text (may be empty if OCR not run yet)
                ocr_text = self._ocr_text
                try:
                    wc = WordConverter()
                    wc.save(
                        image=self._scanned_image,
                        output_path=path,
                        ocr_text=ocr_text,
                        title=Path(path).stem.replace("_", " ").title(),
                    )
                    self._set_status(f"Word saved: {os.path.basename(path)}")
                    tip = "\n\nTip: Run OCR first to include selectable text in the Word file." if not ocr_text else ""
                    messagebox.showinfo("Saved", f"Word document saved:\n{path}{tip}")
                except ImportError as exc:
                    messagebox.showerror(
                        "Missing package",
                        f"{exc}\n\nInstall it with:\n  pip install python-docx"
                    )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _set_status(self, msg: str):
        self._status_var.set(f"  {msg}")

    def _center(self):
        self.update_idletasks()
        w, h  = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")


# ===========================================================================
#  Entry point
# ===========================================================================

def main():
    app = DocumentScannerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
