# Sample Images for Testing

Place your test document images in this folder.

## What to use as test images

Any of the following work great:

| Image Type | Tips |
|---|---|
| Receipt / bill | Place on a dark table, photograph at a slight angle |
| Printed letter | Works best with white paper on dark surface |
| Invoice / form | Any A4 / Letter-size printed page |
| Book page | Photographed at angle to show perspective correction |
| ID card | Good for small document detection |

## Recommended test scenarios

1. **Straight-on photo** — document parallel to camera  
   → Tests enhancement without perspective correction

2. **Angled photo** — document at ~30–45° angle  
   → Tests perspective warp

3. **Low-light / shadowed** — one corner darker  
   → Tests shadow removal

4. **Crumpled page** — slightly creased document  
   → Tests noise reduction + sharpening

## Quick command to scan a sample

```bash
# After placing your image here:
python document_scanner.py --input samples/my_document.jpg
```

---

*KKR Gen AI Innovations — https://kkrgenaiinnovations.com/*
