# Document Scanner AI

> **KKR Gen AI Innovations** — *Empowering Tomorrow with AI-Powered Solutions*

Transform any photo of a document — taken at an angle, in poor lighting, or with shadows — into a **clean, flat, high-contrast scan** automatically using Python and OpenCV.

---

```
╔══════════════════════════════════════════════════════════════════╗
║         KKR Gen AI Innovations — Document Scanner AI            ║
║              Empowering Tomorrow with AI Innovation             ║
║  Web : https://kkrgenaiinnovations.com/  |  WA: +1 470-861-6312 ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Output Examples](#output-examples)
3. [How It Works (Step by Step)](#how-it-works-step-by-step)
4. [Project Structure](#project-structure)
5. [Setup — Install Everything](#setup--install-everything)
6. [How to Run — Step by Step](#how-to-run--step-by-step)
   - [Option A: Command Line](#option-a-command-line-simplest)
   - [Option B: Desktop GUI (Tkinter)](#option-b-desktop-gui-tkinter)
   - [Option C: Web App (Flask)](#option-c-web-app-flask)
   - [Option D: Batch Processing](#option-d-batch-processing)
7. [Enhancement Modes Explained](#enhancement-modes-explained)
8. [Common Errors and Fixes](#common-errors-and-fixes)
9. [About KKR Gen AI Innovations](#about-kkr-gen-ai-innovations)

---

## What This Project Does

When you photograph a document with your phone or camera:

- The document is **tilted** or taken from an angle
- Lighting is **uneven** (shadows on one side)
- The background is **cluttered**
- The image is **blurry** or low-contrast

This scanner fixes all of that automatically:

| Problem | Fix Applied |
|---------|------------|
| Tilted / angled shot | Perspective transformation (bird's-eye warp) |
| Uneven lighting / shadows | Shadow removal via background normalisation |
| Blurry text | Unsharp masking / sharpening filter |
| Random noise (camera grain) | Non-local means denoising |
| Low contrast | Adaptive thresholding or CLAHE |

---

## Output Examples

### Example 1 — Receipt Photo (Angled)

```
INPUT                         OUTPUT
┌─────────────────────┐       ┌──────────────────┐
│   ╱─────────────╲   │       │ RECEIPT           │
│  ╱  RECEIPT      ╲  │  ──►  │ Item 1 ..... $5   │
│ ╱   Item 1 $5     ╲ │       │ Item 2 ..... $12  │
│╱    Item 2 $12     ╲│       │ TOTAL ...... $17  │
└─────────────────────┘       └──────────────────┘
  Tilted, shadowed               Clean flat scan
```

**Before:** Photo taken at ~40° angle, one corner in shadow.  
**After:** Perfectly flat, black text on white background, all text readable.

---

### Example 2 — Printed Letter (Low Contrast)

```
INPUT                         OUTPUT
┌─────────────────────┐       ┌──────────────────┐
│ [grey background]   │       │ Dear John,        │
│ D̤e̤a̤r̤ ̤J̤o̤h̤n̤,̤           │  ──►  │                   │
│ [faded text]        │       │ [crisp black text]│
└─────────────────────┘       └──────────────────┘
  Faded, poor contrast          High-contrast scan
```

**Before:** Old printed letter with yellowed paper and faded ink.  
**After:** Clear black-and-white scan, every word legible.

---

### Example 3 — Invoice (Multiple Shadows)

```
INPUT                         OUTPUT
┌─────────────────────┐       ┌──────────────────┐
│ ████ INVOICE ████   │       │   INVOICE         │
│ ████ dark area ████ │  ──►  │   Item   Amount   │
│ [unreadable shadow] │       │   Book   $25.00   │
└─────────────────────┘       └──────────────────┘
  Heavy shadow across page      Shadow removed
```

---

### Enhancement Mode Comparison

| Mode | Best For | Output |
|------|----------|--------|
| `adaptive` | Camera photos with shadows | Black & white, very crisp |
| `otsu` | Clean flat scans | Black & white, fast |
| `color` | Coloured documents / forms | Full colour, contrast boosted |
| `grayscale` | Simple conversion | Grey tones |

---

## How It Works (Step by Step)

Here is exactly what happens inside the code when you scan a document:

```
Your Photo
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  STEP 1: RESIZE                                       │
│  Scale to 800px wide for faster processing.           │
└───────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  STEP 2: GRAYSCALE                                    │
│  Convert BGR (colour) → single channel grey.          │
│  Grey images are simpler to do edge detection on.     │
└───────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  STEP 3: GAUSSIAN BLUR                                │
│  Smooth out tiny texture bumps (paper grain, noise).  │
│  This prevents false edges from triggering later.     │
└───────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  STEP 4: CANNY EDGE DETECTION                        │
│  Find the strong edges in the image.                  │
│  The document boundary creates the strongest edges.   │
│                                                       │
│  threshold1=75  → ignore weak/noise edges             │
│  threshold2=200 → always keep strong edges            │
└───────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  STEP 5: FIND CONTOURS & PICK LARGEST RECTANGLE      │
│  findContours() traces all closed edge loops.         │
│  We sort by area, keep the 5 biggest.                 │
│  approxPolyDP() simplifies each to a polygon.         │
│  The first one with exactly 4 sides = our document!   │
└───────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  STEP 6: ORDER THE 4 CORNERS                         │
│  Sort into: top-left, top-right, bottom-right,        │
│  bottom-left (needed for the warp to work correctly). │
│                                                       │
│  Trick: top-left has smallest (x+y) sum.              │
└───────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  STEP 7: PERSPECTIVE WARP                             │
│  getPerspectiveTransform() calculates matrix M.       │
│  warpPerspective() maps the skewed document onto a    │
│  perfect rectangle — bird's-eye (flat) view.          │
└───────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  STEP 8: SHADOW REMOVAL (optional)                   │
│  Dilate each channel to estimate the background.      │
│  Divide original by background → flatten illumination.│
└───────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  STEP 9: DENOISE (optional)                          │
│  Non-local means: averages similar patches globally.  │
│  Removes camera grain without blurring text edges.    │
└───────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  STEP 10: SHARPEN                                     │
│  Convolution with a sharpening kernel:                │
│  [ 0 -1  0 ]                                          │
│  [-1  5 -1 ] → boosts edges, makes text pop.          │
│  [ 0 -1  0 ]                                          │
└───────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  STEP 11: ADAPTIVE THRESHOLD                         │
│  Divide the image into 11×11 regions.                 │
│  Compute a threshold per region (handles shadows).    │
│  Pixel > threshold → WHITE.  Pixel ≤ threshold → BLACK│
└───────────────────────────────────────────────────────┘
    │
    ▼
  Clean Scanned Image  ✅
```

---

## Project Structure

```
document-scanner-ai/
│
├── document_scanner.py     ← Main script  (start here!)
├── batch_scanner.py        ← Scan a whole folder at once
├── scanner_gui.py          ← Desktop GUI (Tkinter)
├── app.py                  ← Web app (Flask)
│
├── utils/                  ← Core processing modules
│   ├── __init__.py
│   ├── image_processor.py  ← Edge detection + perspective warp
│   ├── enhancer.py         ← Thresholding, sharpening, denoising
│   └── pdf_converter.py    ← Save as PDF
│
├── templates/              ← Flask HTML pages
│   ├── index.html          ← Upload form
│   └── result.html         ← Result/download page
│
├── static/
│   └── style.css           ← Web app stylesheet
│
├── samples/                ← Put your test images here
├── output/                 ← Scanned results are saved here
└── requirements.txt        ← All Python packages needed
```

---

## Setup — Install Everything

### Prerequisites

You need Python 3.9 or newer. Check your version:

```bash
python --version
```

If you see Python 3.9+, you are ready. Otherwise download Python from https://python.org.

---

### Step 1 — Download the project

```bash
git clone https://github.com/KKRGENAI/document-scanner-ai.git
cd document-scanner-ai
```

---

### Step 2 — Create a virtual environment (recommended)

A virtual environment keeps this project's packages separate from your system Python.

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should now see `(venv)` at the start of your terminal line.

---

### Step 3 — Install required packages

```bash
pip install -r requirements.txt
```

This installs:

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | ≥4.8 | All computer vision operations |
| `numpy` | ≥1.24 | Array/matrix maths |
| `Pillow` | ≥10.0 | Image I/O and PDF export |
| `Flask` | ≥3.0 | Web app server |

Installation takes about 1–2 minutes. You will see progress bars.

---

### Step 4 — Verify installation

```bash
python -c "import cv2; import numpy; import PIL; import flask; print('All packages installed correctly!')"
```

Expected output:
```
All packages installed correctly!
```

---

## How to Run — Step by Step

### Option A: Command Line (Simplest)

This is the easiest way to start. You type a command and get a scanned image.

#### Basic usage

```bash
python document_scanner.py --input samples/your_document.jpg
```

The scanned image is saved to `output/your_document_scanned.jpg`.

#### See all options

```bash
python document_scanner.py --help
```

#### Common examples

```bash
# Scan with default settings (adaptive threshold)
python document_scanner.py --input samples/receipt.jpg

# Also save a PDF version
python document_scanner.py --input samples/letter.jpg --pdf

# Keep colours (don't convert to black-and-white)
python document_scanner.py --input samples/form.jpg --mode color

# See what is happening at each step (opens windows)
python document_scanner.py --input samples/doc.jpg --debug

# Choose where to save the output
python document_scanner.py --input samples/doc.jpg --output output/my_clean_scan.jpg
```

#### What the output looks like

```
╔══════════════════════════════════════════════════════════════════╗
║         KKR Gen AI Innovations — Document Scanner AI            ║
╚══════════════════════════════════════════════════════════════════╝

  [1/5]  Loading image  →  samples/receipt.jpg
         Image size: 3024×4032 px
  [2/5]  Detecting document boundaries …
         Corners detected: [[45, 89], [2989, 102], [3001, 3958], [38, 3940]]
  [3/5]  Applying perspective transformation …
  [4/5]  Enhancing image  (mode=adaptive) …
  [5/5]  Saving result  →  output/receipt_scanned.jpg

  ✅  Scan complete!  Saved: output/receipt_scanned.jpg

  Powered by KKR Gen AI Innovations — https://kkrgenaiinnovations.com/
```

---

### Option B: Desktop GUI (Tkinter)

A graphical window — no typing required.

```bash
python scanner_gui.py
```

**What you see:**

```
┌─────────────────────────────────────────────────────────┐
│  📄 KKR Gen AI Innovations — Document Scanner AI        │
├──────────────┬──────────────────────────────────────────┤
│              │  Original Image    │  Scanned Output     │
│  1. LOAD     │                    │                     │
│  [Browse…]   │  [your image       │  [clean scan        │
│              │   displayed here]  │   appears here]     │
│  2. SETTINGS │                    │                     │
│  Mode: ▼     │                    │                     │
│  ☑ Denoise   │                    │                     │
│  ☑ Shadows   │                    │                     │
│  ☑ Sharpen   │                    │                     │
│              │                    │                     │
│  3. SCAN     │                    │                     │
│  [▶ Scan]    │                    │                     │
│              │                    │                     │
│  4. SAVE     │                    │                     │
│  [💾 JPG]    │                    │                     │
│  [📄 PDF]    │                    │                     │
└──────────────┴────────────────────┴─────────────────────┘
```

**Steps:**
1. Click **Browse** and pick your image
2. (Optional) Choose an enhancement mode from the dropdown
3. Click **Scan Document** — result appears on the right
4. Click **Save as JPG** or **Save as PDF**

---

### Option C: Web App (Flask)

Open the scanner in your web browser — works on any device on your network.

#### Start the server

```bash
python app.py
```

You will see:

```
  KKR Gen AI Innovations — Document Scanner AI (Web)
  Open your browser: http://localhost:5000
```

#### Open in browser

Go to: **http://localhost:5000**

You will see a page with:
- A drag-and-drop upload area
- Mode selector
- Scan button

After scanning, the result page shows a **before/after comparison** and download buttons for JPG and PDF.

#### Stop the server

Press `Ctrl + C` in the terminal.

---

### Option D: Batch Processing

Scan all images in a folder at once.

```bash
# Scan every image in the samples/ folder
python batch_scanner.py --input samples/

# Scan and combine everything into one PDF
python batch_scanner.py --input samples/ --pdf

# Use 4 parallel workers (faster on multi-core CPUs)
python batch_scanner.py --input samples/ --workers 4
```

**Progress output:**

```
  Found 8 image(s) to process.
  Output folder : output/batch
  Enhancement   : adaptive
  Workers       : 2
  ────────────────────────────────────────────────────────────
  ✅  [  1/8]  receipt.jpg          → receipt_scanned.jpg
  ✅  [  2/8]  letter.jpg           → letter_scanned.jpg
  ✅  [  3/8]  invoice.jpg          → invoice_scanned.jpg
  ...
  ────────────────────────────────────────────────────────────
  Batch complete in 4.2s
  ✅  Success : 8   ❌  Failed : 0
  📄  Combined PDF : output/batch/batch_scan.pdf
```

---

## Enhancement Modes Explained

### adaptive (default) — Best for most photos

Divides the image into small regions (11×11 pixels) and calculates a separate threshold for each region. Works brilliantly when parts of the document are in shadow.

**Use when:** Photo taken with a camera or phone, uneven lighting, shadows.

### otsu — For clean scans

Automatically finds the single best global threshold value using Otsu's algorithm. Fast and simple.

**Use when:** Already-scanned image, flat even lighting.

### color — Keep the colours

Applies CLAHE (Contrast Limited Adaptive Histogram Equalisation) only to the brightness channel. Colours are preserved while contrast improves.

**Use when:** Coloured forms, diagrams, certificates, coloured text.

### grayscale — Simple greyscale

Just converts to greyscale. No thresholding. Useful when you want a greyscale photo but not a hard black-and-white binary image.

**Use when:** You want to keep grey tones (e.g. a photo of a page with graphics).

---

## Common Errors and Fixes

### Error: `ModuleNotFoundError: No module named 'cv2'`

OpenCV is not installed.

```bash
pip install opencv-python
```

---

### Error: `ModuleNotFoundError: No module named 'PIL'`

Pillow is not installed.

```bash
pip install Pillow
```

---

### Error: `No document boundary found`

The scanner could not detect the document's edges. This happens when:

| Cause | Fix |
|-------|-----|
| Document same colour as background | Use a darker/contrasting surface |
| Very dark photo | Increase room lighting |
| Blurry image | Hold camera steady, use focus |
| No clear rectangular boundary | The `--debug` flag shows detected edges |

The scanner will still output an enhanced version using the full image.

---

### Error: `FileNotFoundError: Cannot open image`

The file path is wrong or the file doesn't exist.

```bash
# Check the file exists
ls samples/

# Use the correct path
python document_scanner.py --input samples/correct_name.jpg
```

---

### Error: `ImportError: Pillow is required for PDF export`

```bash
pip install Pillow
```

---

### The GUI window doesn't open

Make sure you have a display/screen attached. On headless servers, Tkinter won't work — use the CLI or Flask web app instead.

---

### Flask says `Address already in use`

Another process is using port 5000.

```bash
# Use a different port
python app.py --port 5001
```

Or find and stop the other process using port 5000.

---

## About KKR Gen AI Innovations

**KKR Gen AI Innovations** is a leading global IT solutions provider specialising in AI-driven technologies and digital transformation. We are committed to revolutionising the way businesses operate through innovative design, cutting-edge technology, and intelligent, data-driven solutions.

**Our Mission:** To future-proof businesses by delivering AI-powered solutions, cutting-edge digital technologies, and training programs that prepare professionals and young learners for the opportunities of tomorrow.

**Our Vision:** To help organisations innovate, grow, and excel in today's fast-paced digital environment through tailored solutions that drive sustainable growth and long-term success.

### Connect With Us

| Channel | Link |
|---------|------|
| Website | [kkrgenaiinnovations.com](https://kkrgenaiinnovations.com/) |
| Email | info@kkrgenaiinnovations.com |
| WhatsApp | [+1 470-861-6312](https://wa.me/14708616312) |
| Twitter / X | [@kkr_genai_](https://x.com/kkr_genai_) |
| Facebook | [kkrgenaiinnovations](https://www.facebook.com/kkrgenaiinnovations) |
| Instagram | [@kkrgenaiinnovations](https://www.instagram.com/kkrgenaiinnovations/) |
| LinkedIn | [KKR Gen AI Innovations](https://www.linkedin.com/company/kkr-genai-innovations/) |

---

*© 2025 KKR Gen AI Innovations. All rights reserved. Empowering Tomorrow.*
