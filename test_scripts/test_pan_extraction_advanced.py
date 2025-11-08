#!/usr/bin/env python3
"""
Test – PAN extraction via foduucom/pan-card-detection YOLO + EasyOCR with graceful fallbacks.

Run:
  python3 scripts/test_pan_extraction_advanced.py

What it does:
- Loads PAN image from scripts/pancard_image/PAN.jpeg
- Uses foduucom/pan-card-detection pre-trained YOLO model from Hugging Face to detect PAN fields
- Crops specific fields (pan_number, name, father_name, dob, photo), then runs EasyOCR on crops
- If YOLO unavailable, falls back to EasyOCR on full image, then pytesseract if needed
- Also extracts and saves photo crop via either YOLO 'photo' box or Haar cascade
"""
import os
from pan_advanced_extraction import extract_pan_fields, DEFAULT_IMAGE, OUT_DIR

if __name__ == '__main__':
    print("\n🔬 PAN Advanced Extraction Test (foduucom YOLO)")
    # Prefer explicit image via env or try common samples
    img_candidates = [
        os.environ.get('PAN_IMAGE') or '',
        DEFAULT_IMAGE,
        os.path.join(os.path.dirname(DEFAULT_IMAGE), 'PAN.jpeg'),
    ]
    img = next((p for p in img_candidates if p and os.path.exists(p)), '')
    if not img:
        print("❌ No PAN image found. Place one at scripts/pancard_image/PAN.jpeg or pancard_try.jpeg")
        raise SystemExit(1)

    res = extract_pan_fields(img)

    print("\n📊 Results")
    print(f"OCR engine: {res.get('ocr_engine')}")
    print(f"PAN number: {res.get('pan_number')}")
    print(f"DOB:        {res.get('dob')}")
    print(f"Name:       {res.get('name')}")
    print(f"Father:     {res.get('father_name')}")
    print(f"Crops saved to: {OUT_DIR}")

