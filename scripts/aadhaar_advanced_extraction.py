#!/usr/bin/env python3
"""
Advanced Aadhaar extraction using YOLOv8 OpenVINO (thewalnutaisg) + EasyOCR.

Uses a pre-trained YOLO model optimized with OpenVINO that detects Aadhaar card fields:
- AADHAR_NUMBER, DATE_OF_BIRTH, GENDER, NAME, ADDRESS

Then applies EasyOCR to each cropped region for accurate text extraction.

Usage (standalone):
  python3 scripts/aadhaar_advanced_extraction.py
"""
import os
import re
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

# Optional imports
try:
    from openvino.runtime import Core
    HAS_OPENVINO = True
except Exception:
    HAS_OPENVINO = False

try:
    import easyocr
    HAS_EASYOCR = True
except Exception:
    HAS_EASYOCR = False

try:
    from huggingface_hub import snapshot_download
    HAS_HF_HUB = True
except Exception:
    HAS_HF_HUB = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGE = os.path.join(PROJECT_ROOT, 'scripts', 'aadhar_image', 'Aadhar both sides.png')
OUT_DIR = os.path.join(PROJECT_ROOT, 'scripts', 'extracted_aadhaar_regions')
FACE_OUT_DIR = os.path.join(PROJECT_ROOT, 'scripts', 'extracted_face_image')
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_REPO = "thewalnutaisg/YOLOv8-aadhar-card-int8-openvino"
MODEL_LOCAL_DIR = Path(PROJECT_ROOT) / "models" / "aadhaar_yolov8_openvino"

CASCADE_PATH = os.path.join(PROJECT_ROOT, 'scripts', 'haarcascade_frontalface_default.xml')

# Global EasyOCR reader instance (initialized once)
_EASYOCR_READER = None

# ---------- Utilities ----------

def _clip_box(xyxy, w, h):
    """Clip bounding box to image bounds."""
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    return (int(x1), int(y1), int(x2), int(y2))

def _get_easyocr_reader():
    """Get or initialize EasyOCR reader (singleton pattern)."""
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        if not HAS_EASYOCR:
            raise ImportError("EasyOCR not installed. Install with: pip install easyocr")
        _EASYOCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _EASYOCR_READER

# ---------- YOLO OpenVINO Detection ----------

def load_yolov8_openvino_model():
    """
    Download and load YOLOv8 OpenVINO model from HuggingFace.
    Returns (compiled_model, class_names, input_layer, output_layer) or (None, None, None, None).
    """
    if not HAS_OPENVINO:
        return None, None, None, None
    
    if not HAS_HF_HUB:
        return None, None, None, None
    
    try:
        # Download model if not cached
        if not MODEL_LOCAL_DIR.exists() or not (MODEL_LOCAL_DIR / "model.xml").exists():
            snapshot_download(
                repo_id=MODEL_REPO,
                local_dir=str(MODEL_LOCAL_DIR),
                local_dir_use_symlinks=False
            )
        
        model_xml = MODEL_LOCAL_DIR / "model.xml"
        
        if not model_xml.exists():
            return None, None, None, None
        
        # Load with OpenVINO
        ie = Core()
        model = ie.read_model(model=str(model_xml))
        compiled_model = ie.compile_model(model=model, device_name="CPU")
        
        # Get input/output layers
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        
        # Class names
        class_names = {
            0: "AADHAR_NUMBER",
            1: "DATE_OF_BIRTH", 
            2: "GENDER",
            3: "NAME",
            4: "ADDRESS"
        }
        
        return compiled_model, class_names, input_layer, output_layer
        
    except Exception:
        return None, None, None, None

def detect_aadhaar_fields_yolo(image_bgr, compiled_model, class_names, input_layer, output_layer, conf_threshold=0.25):
    """
    Run YOLO inference using OpenVINO to detect Aadhaar fields.
    Returns dict: {field_label: (crop_img, (x1,y1,x2,y2))}
    """
    try:
        img_height, img_width = image_bgr.shape[:2]
        input_height, input_width = input_layer.shape[2], input_layer.shape[3]
        
        # Preprocess image
        resized = cv2.resize(image_bgr, (input_width, input_height))
        input_data = np.transpose(resized, (2, 0, 1))  # HWC to CHW
        input_data = input_data.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        # Run inference
        result = compiled_model([input_data])[output_layer]
        
        # Post-process results (YOLOv8 format)
        detections = result[0]
        
        detected_fields = {}
        
        for detection in detections.T:
            # YOLOv8 format: [x, y, w, h, conf, class_scores...]
            x, y, w, h = detection[:4]
            confidences = detection[4:]
            
            class_id = np.argmax(confidences)
            confidence = confidences[class_id]
            
            if confidence > conf_threshold:
                # Convert from normalized coordinates
                x1 = int((x - w/2) * img_width / input_width)
                y1 = int((y - h/2) * img_height / input_height)
                x2 = int((x + w/2) * img_width / input_width)
                y2 = int((y + h/2) * img_height / input_height)
                
                x1, y1, x2, y2 = _clip_box((x1, y1, x2, y2), img_width, img_height)
                
                label = class_names.get(int(class_id), f"class_{int(class_id)}")
                crop = image_bgr[y1:y2, x1:x2]
                
                # Keep highest confidence if duplicate
                if label not in detected_fields or confidence > detected_fields[label][2]:
                    detected_fields[label] = (crop, (x1, y1, x2, y2), float(confidence))
        
        # Return format: {label: (crop_img, bbox)}
        return {k: (v[0], v[1]) for k, v in detected_fields.items()}
        
    except Exception:
        return {}

# ---------- OCR Extraction ----------

def extract_text_from_crop(crop_img, field_name: str) -> str:
    """
    Extract text from a cropped region using EasyOCR.
    Apply field-specific cleaning rules.
    """
    if crop_img is None or crop_img.size == 0:
        return ""
    
    try:
        reader = _get_easyocr_reader()
        results = reader.readtext(crop_img, detail=0, paragraph=True)
        
        if not results:
            return ""
        
        text = " ".join(results).strip()
        
        # Field-specific cleaning
        if field_name == "AADHAR_NUMBER":
            # Extract 12-digit Aadhaar number
            # Remove all non-digits
            digits = re.sub(r'\D', '', text)
            if len(digits) >= 12:
                aadhaar = ''.join(digits[:12])
                return f"{aadhaar[:4]} {aadhaar[4:8]} {aadhaar[8:12]}"
            return ''.join(digits)
        
        elif field_name == "DATE_OF_BIRTH":
            # Look for DD/MM/YYYY or DD-MM-YYYY
            # First, clean up common OCR mistakes
            text = re.sub(r'[O0]([O0]\d)', r'0\1', text)  # Replace O with 0
            text = re.sub(r'([^/\-])([O0])', r'\g<1>0', text)  # Replace O with 0 in dates
            
            dob_match = re.search(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b', text)
            if dob_match:
                day = dob_match.group(1).zfill(2)
                month = dob_match.group(2).zfill(2)
                year = dob_match.group(3)
                return f"{day}/{month}/{year}"
            return text
        
        elif field_name == "GENDER":
            # Look for Male/Female/Other
            text_lower = text.lower()
            if 'male' in text_lower and 'female' not in text_lower:
                return "Male"
            elif 'female' in text_lower:
                return "Female"
            return text.title()
        
        elif field_name == "NAME":
            # Clean up name - remove common prefixes and OCR noise
            text = text.strip()
            
            # Remove leading noise and common prefixes
            text = re.sub(r'^[^A-Z]*(?=[A-Z])', '', text)  # Remove junk before first capital letter
            text = re.sub(r'^\s*(?:Name|NAME|Holder|HOLDER)[\s:]*', '', text, flags=re.IGNORECASE)
            
            # Clean up multiple spaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove lines that are clearly OCR artifacts (mostly numbers/symbols)
            if text and sum(c.isalpha() for c in text) > len(text) * 0.5:  # At least 50% letters
                return text
            return text.title() if text else text
        
        elif field_name == "ADDRESS":
            # Clean up address - remove common OCR artifacts and prefixes
            # Remove leading noise/artifacts (OCR junk before actual address)
            # Common patterns: numbers at start, OCR garbage like "56uus7", "CIO:", etc.
            text = text.strip()
            
            # Remove leading digits, special characters, and common OCR artifacts
            # Pattern: starts with noise until we find actual address text
            text = re.sub(r'^[0-9]+\s*', '', text)  # Remove leading digits
            text = re.sub(r'^[^A-Z]*(?=[A-Z])', '', text)  # Remove junk before first capital letter
            text = re.sub(r'^\s*(?:Address|ADDRESS|Addr)[\s:]*', '', text, flags=re.IGNORECASE)  # Remove "Address:" prefix
            text = re.sub(r'^\s*(?:CIO|C/O|C\.O\.|Care\s+Of)[\s:]*', '', text, flags=re.IGNORECASE)  # Remove CIO/C/O prefix
            
            # Clean up multiple spaces and normalize
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove trailing noise (incomplete words, stray characters)
            text = re.sub(r'\s+[A-Za-z]{1}(?:\s|$)', '', text).strip()  # Remove single letter words at end
            
            return text
        
        return text
        
    except Exception:
        return ""

# ---------- Face Detection ----------

def detect_face_from_aadhaar(image_bgr) -> Optional[np.ndarray]:
    """
    Detect and extract face photo from Aadhaar card using Haar Cascade.
    Returns cropped face image or None.
    """
    if not os.path.exists(CASCADE_PATH):
        return None
    
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Load cascade
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        if len(faces) == 0:
            return None
        
        # Get largest face (Aadhaar photo)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Crop face with small padding
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image_bgr.shape[1], x + w + padding)
        y2 = min(image_bgr.shape[0], y + h + padding)
        
        face_crop = image_bgr[y1:y2, x1:x2]
        
        return face_crop if face_crop.size > 0 else None
        
    except Exception:
        return None

# ---------- Main Extraction Function ----------

def extract_aadhaar_fields(image_path: str) -> Dict[str, str]:
    """
    Extract Aadhaar card fields using YOLOv8 OpenVINO + EasyOCR.
    
    Args:
        image_path: Path to Aadhaar card image
    
    Returns:
        Dictionary with keys:
        - aadhaar_number: 12-digit Aadhaar number
        - name: Person's name
        - dob: Date of birth (DD/MM/YYYY)
        - gender: Gender
        - address: Full address
        - photo: Path to extracted face photo (if found)
        - ocr_engine: Extraction method used
    """
    result = {
        'aadhaar_number': '',
        'name': '',
        'dob': '',
        'gender': '',
        'address': '',
        'photo': '',
        'ocr_engine': 'none'
    }
    
    # Validate input
    if not os.path.exists(image_path):
        return result
    
    # Load image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return result
    
    # Load YOLO model
    compiled_model, class_names, input_layer, output_layer = load_yolov8_openvino_model()
    
    if compiled_model is None:
        return result
    
    # Detect fields
    detected_fields = detect_aadhaar_fields_yolo(
        image_bgr, compiled_model, class_names, input_layer, output_layer
    )
    
    if not detected_fields:
        return result
    
    result['ocr_engine'] = 'yolov8_openvino_easyocr'
    
    # Extract face photo from Aadhaar card
    face_crop = detect_face_from_aadhaar(image_bgr)
    if face_crop is not None:
        photo_path = os.path.join(OUT_DIR, 'photo.jpg')
        cv2.imwrite(photo_path, face_crop)
        photo_path = os.path.join(FACE_OUT_DIR, 'aadhar_photo.jpg')
        cv2.imwrite(photo_path, face_crop)
        result['photo'] = photo_path
    
    # Extract text from each field
    for field_name, (crop, bbox) in detected_fields.items():
        if crop.size == 0:
            continue
        
        # Save crop
        crop_filename = f"{field_name.lower()}.jpg"
        crop_path = os.path.join(OUT_DIR, crop_filename)
        cv2.imwrite(crop_path, crop)
        
        # Extract text
        text = extract_text_from_crop(crop, field_name)
        
        # Map to result keys
        if field_name == "AADHAR_NUMBER":
            result['aadhaar_number'] = text
        elif field_name == "NAME":
            result['name'] = text
        elif field_name == "DATE_OF_BIRTH":
            result['dob'] = text
        elif field_name == "GENDER":
            result['gender'] = text
        elif field_name == "ADDRESS":
            result['address'] = text
    
    return result

# ---------- Standalone Usage ----------

if __name__ == '__main__':
    print("\n🔍 AADHAAR ADVANCED EXTRACTION (YOLOv8 OpenVINO + EasyOCR)")
    print("=" * 70)
    
    # Use environment variable or default image
    img_path = os.environ.get('AADHAAR_IMAGE', DEFAULT_IMAGE)
    
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        print(f"   Set AADHAAR_IMAGE environment variable or place image at:")
        print(f"   {DEFAULT_IMAGE}")
        exit(1)
    
    print(f"📄 Processing: {img_path}")
    
    # Extract fields
    result = extract_aadhaar_fields(img_path)
    
    # Display results
    print("\n📊 Extraction Results:")
    print("-" * 70)
    print(f"OCR Engine:     {result['ocr_engine']}")
    print(f"Name:           {result['name']}")
    print(f"Aadhaar Number: {result['aadhaar_number']}")
    print(f"Date of Birth:  {result['dob']}")
    print(f"Gender:         {result['gender']}")
    print(f"Address:        {result['address']}")
    print(f"Photo:          {result['photo'] if result['photo'] else 'Not extracted'}")
    print("-" * 70)
    
    if result['name'] or result['aadhaar_number']:
        print(f"✅ Extraction successful")
        print(f"📁 Cropped regions saved to: {OUT_DIR}")
        if result['photo']:
            print(f"📷 Face photo extracted: {result['photo']}")
    else:
        print("❌ Extraction failed")
