#!/usr/bin/env python3
"""
Advanced PAN extraction using foduucom/pan-card-detection YOLOv8 model + EasyOCR.

Uses a pre-trained YOLO model from Hugging Face that detects PAN card fields:
- pan_number, name, father_name, dob, photo

Then applies EasyOCR to each cropped region for accurate text extraction.

Fallback: If YOLO fails, uses EasyOCR on full image with heuristics.

Usage (standalone):
  python3 scripts/pan_advanced_extraction.py
"""
import os
import re
import cv2
import numpy as np
from typing import Dict, Optional, Tuple

# Optional imports
try:
    from ultralyticsplus import YOLO
    HAS_YOLO = True
except Exception:
    HAS_YOLO = False

try:
    import easyocr
    HAS_EASYOCR = True
except Exception:
    HAS_EASYOCR = False

try:
    import pytesseract
    HAS_TESS = True
except Exception:
    HAS_TESS = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGE = os.path.join(PROJECT_ROOT, 'scripts', 'pancard_image', 'PAN.jpeg')
OUT_DIR = os.path.join(PROJECT_ROOT, 'scripts', 'extracted_pan_regions')
FACE_OUT_DIR = os.path.join(PROJECT_ROOT, 'scripts', 'extracted_face_image')
os.makedirs(OUT_DIR, exist_ok=True)

CASCADE_PATH = os.path.join(PROJECT_ROOT, 'scripts', 'haarcascade_frontalface_default.xml')

# ---------- Utilities ----------

def _clip_box(xyxy, w, h):
    """Clip bounding box to image bounds."""
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, min(w-1, x1))
    x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(0, min(h-1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def _extract_roi(image, box, pad=4):
    """Extract region of interest with padding."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return image[y1:y2, x1:x2]

# ---------- Regex helpers ----------

def _regex_pan(text: str) -> Optional[str]:
    """Extract PAN number (AAAAA9999A format)."""
    m = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', text)
    return m.group(0) if m else None

def _regex_dob(text: str) -> Optional[str]:
    """Extract date of birth."""
    for pat in (r'\b\d{2}/\d{2}/\d{4}\b', r'\b\d{2}-\d{2}-\d{4}\b'):
        m = re.search(pat, text)
        if m:
            return m.group(0)
    return None

# ---------- YOLO + EasyOCR Pipeline ----------

def detect_fields_with_foduucom_yolo(image_bgr) -> Dict[str, Tuple[np.ndarray, Tuple[int,int,int,int]]]:
    """
    Use foduucom/pan-card-detection YOLO model to detect PAN card fields.
    Returns dict mapping field_name -> (crop_bgr, (x1,y1,x2,y2)).
    """
    out: Dict[str, Tuple[np.ndarray, Tuple[int,int,int,int]]] = {}
    
    if not HAS_YOLO:
        return out
    
    try:
        # Download the model from Hugging Face manually
        import torch
        from huggingface_hub import hf_hub_download
        
        # Download model weights  
        model_path = hf_hub_download(
            repo_id="foduucom/pan-card-detection",
            filename="best.pt"
        )
        
        # Monkey-patch ultralytics torch_safe_load to use weights_only=False
        # This is needed for PyTorch 2.9+ which changed security defaults
        from ultralytics.nn import tasks
        original_torch_safe_load = tasks.torch_safe_load
        
        def patched_torch_safe_load(file):
            """Load torch file with weights_only=False for compatibility"""
            return torch.load(file, map_location='cpu', weights_only=False), file
        
        tasks.torch_safe_load = patched_torch_safe_load
        
        try:
            # Load with ultralyticsplus
            model = YOLO(model_path)
            
            # Override model parameters
            model.overrides['conf'] = 0.25
            model.overrides['iou'] = 0.45
            model.overrides['agnostic_nms'] = False
            model.overrides['max_det'] = 1000
            
            # Run detection
            results = model.predict(image_bgr, imgsz=640, conf=0.25)
            
        finally:
            # Restore original function
            tasks.torch_safe_load = original_torch_safe_load
        
        if not results or len(results) == 0:
            return out
        
        detected = results[0]
        h, w = image_bgr.shape[:2]
        
        # Extract bounding boxes and class IDs
        if not hasattr(detected, 'boxes') or detected.boxes is None or len(detected.boxes) == 0:
            return out
        
        boxes = detected.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        classes = detected.boxes.cls.cpu().numpy().astype(int)
        
        # Get class names - try multiple attributes
        names = {}
        if hasattr(model, 'names'):
            names = model.names
        elif hasattr(model, 'model') and hasattr(model.model, 'names'):
            names = model.model.names
        
        print(f"  🔍 Detected {len(boxes)} regions")
        
        for box, cls_id in zip(boxes, classes):
            label = names.get(cls_id, str(cls_id))
            
            print(f"     Class {cls_id}: {label}")
            
            # Clip to image bounds
            clipped = _clip_box(box, w, h)
            if clipped is None:
                continue
            
            # Extract ROI with padding
            roi = _extract_roi(image_bgr, clipped, pad=5)
            
            # Store the crop and coordinates
            out[label] = (roi, clipped)
        
        return out
        
    except Exception as e:
        print(f"YOLO detection error: {e}")
        import traceback
        traceback.print_exc()
        return out

def extract_text_from_crop(crop_bgr, reader) -> str:
    """Extract text from a cropped region using EasyOCR."""
    try:
        result = reader.readtext(crop_bgr, detail=0, paragraph=False)
        # Join all detected text
        text = ' '.join(result).strip()
        return text
    except Exception as e:
        return ''

def clean_name_field(text: str) -> str:
    """
    Clean name/father name fields by removing common labels and noise.
    PAN cards often have labels like "Name", "Father's Name", etc.
    Actual names are typically in ALL CAPS and are proper nouns.
    """
    if not text:
        return text
    
    # Uppercase for processing
    text_upper = text.upper()
    
    # Remove common labels and noise patterns
    noise_patterns = [
        r'PERMANENT\s+ACCOUNT\s+NUMBER',
        r'INCOME\s+TAX\s+DEPARTMENT',
        r'GOVT?\s+OF\s+INDIA',
        r"FATHER['\s]*S?\s+NAME",
        r'\bNAME\b',
        r'\bSURNAME\b',
        r'\bSIGNATURE\b',
        r'\bDATE\s+OF\s+BIRTH\b',
        r'\bDOB\b',
        r'\bCARD\b',
        r'[#@$%&*]+',  # Special characters
        r'\b[A-Z]{1}\b',  # Single letters
        r'\d+',  # Numbers
    ]
    
    cleaned = text_upper
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, ' ', cleaned)
    
    # Remove non-letter characters except spaces
    cleaned = re.sub(r'[^A-Z\s]', ' ', cleaned)
    
    # Extract words that are 2+ letters
    words = [w for w in cleaned.split() if len(w) >= 2]
    
    # Filter out common non-name words that might appear
    stopwords = {'THE', 'AND', 'OR', 'FOR', 'WITH', 'FAT', 'ENAME', 'TT'}
    words = [w for w in words if w not in stopwords]
    
    return ' '.join(words).strip()

# ---------- Face detection fallback ----------

def detect_face_roi(image_bgr) -> Optional[np.ndarray]:
    """Detect and extract face using Haar cascade."""
    if not os.path.exists(CASCADE_PATH):
        return None
    
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(CASCADE_PATH)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        
        if len(faces) == 0:
            return None
        
        # Pick largest face
        x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
        return image_bgr[y:y+h, x:x+w]
    except Exception:
        return None

# ---------- Full image EasyOCR fallback ----------

def ocr_easyocr_full(image_bgr, reader) -> str:
    """Run EasyOCR on full image."""
    try:
        result = reader.readtext(image_bgr, detail=0, paragraph=False)
        return '\n'.join(result)
    except Exception:
        return ''

# ---------- Main extraction function ----------

def extract_pan_fields(image_path: str) -> Dict[str, Optional[str]]:
    """
    Extract PAN card fields using foduucom YOLO + EasyOCR.
    
    Returns dict with keys: pan_number, name, father_name, dob, ocr_engine
    Also saves cropped regions to OUT_DIR.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    result = {
        'pan_number': None,
        'name': None,
        'father_name': None,
        'dob': None,
        'ocr_engine': None
    }

    # 1) Try YOLO + EasyOCR pipeline
    if HAS_YOLO and HAS_EASYOCR:
        print("🔍 Using YOLO (foduucom/pan-card-detection) + EasyOCR...")
        fields = detect_fields_with_foduucom_yolo(image)
        
        if fields:
            result['ocr_engine'] = 'yolo+easyocr'
            reader = easyocr.Reader(['en'], gpu=False)
            
            for field_name, (crop, box) in fields.items():
                # Save crop
                crop_path = os.path.join(OUT_DIR, f"{field_name}.jpg")
                cv2.imwrite(crop_path, crop)
                print(f"  ✓ Saved {field_name} crop")
                
                # Skip photo field (no OCR needed)
                if field_name == 'photo':
                    continue
                
                # Extract text using EasyOCR
                text = extract_text_from_crop(crop, reader)
                text_upper = text.upper()
                
                print(f"  📝 {field_name}: {text}")
                
                # Map to result fields
                if field_name == 'pan_number' or 'pan' in field_name.lower():
                    # Try regex first, fallback to raw text
                    pan = _regex_pan(text_upper)
                    result['pan_number'] = pan if pan else text_upper.replace(' ', '')
                    
                elif field_name == 'dob' or 'date' in field_name.lower():
                    dob = _regex_dob(text)
                    result['dob'] = dob if dob else text
                    
                elif field_name == 'name':
                    result['name'] = clean_name_field(text)
                    
                elif 'father' in field_name.lower():
                    result['father_name'] = clean_name_field(text)
            
            # If photo wasn't detected, try Haar cascade
            if 'photo' not in fields:
                face = detect_face_roi(image)
                if face is not None:
                    cv2.imwrite(os.path.join(OUT_DIR, 'photo.jpg'), face)
                    cv2.imwrite(os.path.join(FACE_OUT_DIR, 'pan_photo.jpg'), face)
                    print("  ✓ Saved photo (Haar cascade fallback)")
            
            return result

    # 2) Fallback: EasyOCR on full image
    if HAS_EASYOCR:
        print("⚠️  YOLO unavailable, using EasyOCR on full image...")
        result['ocr_engine'] = 'easyocr'
        reader = easyocr.Reader(['en'], gpu=False)
        text = ocr_easyocr_full(image, reader).upper()
        
        # Extract fields with heuristics
        result['pan_number'] = _regex_pan(text)
        result['dob'] = _regex_dob(text)
        
        # Names: take first two alphabetic lines
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        alpha_lines = [l for l in lines if sum(ch.isalpha() or ch.isspace() for ch in l)/max(1,len(l)) > 0.7]
        if alpha_lines:
            result['name'] = alpha_lines[0]
        if len(alpha_lines) > 1:
            result['father_name'] = alpha_lines[1]
        
        # Face detection
        face = detect_face_roi(image)
        if face is not None:
            cv2.imwrite(os.path.join(OUT_DIR, 'photo.jpg'), face)
            cv2.imwrite(os.path.join(FACE_OUT_DIR, 'pan_photo.jpg'), face)
        return result

    # 3) Final fallback: pytesseract
    if HAS_TESS:
        print("⚠️  EasyOCR unavailable, using pytesseract...")
        result['ocr_engine'] = 'pytesseract'
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thr = cv2.resize(thr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(thr, lang='eng', config='--psm 6').upper()
        
        result['pan_number'] = _regex_pan(text)
        result['dob'] = _regex_dob(text)
        
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        alpha_lines = [l for l in lines if sum(ch.isalpha() or ch.isspace() for ch in l)/max(1,len(l)) > 0.7]
        if alpha_lines:
            result['name'] = alpha_lines[0]
        if len(alpha_lines) > 1:
            result['father_name'] = alpha_lines[1]
        
        face = detect_face_roi(image)
        if face is not None:
            cv2.imwrite(os.path.join(OUT_DIR, 'photo.jpg'), face)
            cv2.imwrite(os.path.join(FACE_OUT_DIR, 'pan_photo.jpg'), face)
        return result

    # No OCR available
    result['ocr_engine'] = 'none'
    return result

if __name__ == '__main__':
    print("\n🔍 PAN ADVANCED EXTRACTION (foduucom YOLO + EasyOCR)")
    img = DEFAULT_IMAGE
    if not os.path.exists(img):
        print(f"❌ Image missing: {img}")
    else:
        res = extract_pan_fields(img)
        print("\n📊 Results:")
        print(f"  OCR engine: {res.get('ocr_engine')}")
        print(f"  PAN number: {res.get('pan_number')}")
        print(f"  DOB:        {res.get('dob')}")
        print(f"  Name:       {res.get('name')}")
        print(f"  Father:     {res.get('father_name')}")
        print(f"  Crops saved to: {OUT_DIR}")

