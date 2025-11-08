#!/usr/bin/env python3
"""
Aadhaar Field Extraction using YOLOv8 OpenVINO + EasyOCR
Model: thewalnutaisg/YOLOv8-aadhar-card-int8-openvino

Extracts ALL Aadhaar card fields:
- Aadhaar Number
- Name  
- Date of Birth (DOB)
- Gender
- Address

Run:
    python test_scripts/test_aadhaar_yolov8_openvino.py
"""

import os
import sys
import cv2
import numpy as np
import re
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def section(title: str):
    """Print section header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def download_and_load_model():
    """Download YOLOv8 OpenVINO model from HuggingFace"""
    section("STEP 1: Loading YOLOv8 OpenVINO Model")
    
    try:
        from huggingface_hub import snapshot_download
        from openvino.runtime import Core
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install huggingface_hub openvino-dev")
        sys.exit(1)
    
    MODEL_REPO_ID = "thewalnutaisg/YOLOv8-aadhar-card-int8-openvino"
    MODEL_LOCAL_DIR = PROJECT_ROOT / "models" / "aadhaar_yolov8_openvino"
    
    print(f"📥 Downloading model from {MODEL_REPO_ID}...")
    print(f"   This may take a moment on first run...")
    
    try:
        snapshot_download(
            repo_id=MODEL_REPO_ID,
            local_dir=str(MODEL_LOCAL_DIR),
        )
        print(f"✅ Model downloaded to: {MODEL_LOCAL_DIR}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)
    
    print("🔄 Loading OpenVINO model...")
    try:
        # Load with OpenVINO runtime directly
        model_xml = MODEL_LOCAL_DIR / "model.xml"
        ie = Core()
        model = ie.read_model(model=str(model_xml))
        compiled_model = ie.compile_model(model=model, device_name="CPU")
        
        print("✅ Model loaded successfully with OpenVINO")
        
        # Class names
        class_names = {0: "AADHAR_NUMBER", 1: "DATE_OF_BIRTH", 2: "GENDER", 3: "NAME", 4: "ADDRESS"}
        print(f"   Supported classes: {list(class_names.values())}")
        
        # Get input/output layers
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        
        return compiled_model, class_names, input_layer, output_layer
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def load_image(image_path: str):
    """Load image without resizing to preserve quality"""
    section("STEP 2: Loading Aadhaar Image")
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to read image")
        sys.exit(1)
    
    h, w = img.shape[:2]
    print(f"✅ Image loaded: {w}x{h} pixels")
    print(f"   Path: {image_path}")
    return img

def detect_fields(model, image, class_names: dict, input_layer, output_layer, conf_threshold: float = 0.25):
    """Run YOLO inference using OpenVINO to detect Aadhaar fields"""
    section("STEP 3: Detecting Fields with YOLOv8 OpenVINO")
    
    print(f"🔍 Running inference (confidence >= {conf_threshold})...")
    
    try:
        # Preprocess image for YOLO
        img_height, img_width = image.shape[:2]
        input_height, input_width = input_layer.shape[2], input_layer.shape[3]
        
        # Resize and normalize
        resized = cv2.resize(image, (input_width, input_height))
        input_data = np.transpose(resized, (2, 0, 1))  # HWC to CHW
        input_data = input_data.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        # Run inference
        result = model([input_data])[output_layer]
        
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
                
                label = class_names.get(int(class_id), f"class_{int(class_id)}")
                
                print(f"   • {label}: bbox=({x1},{y1},{x2},{y2}), confidence={confidence:.2f}")
                
                # Keep highest confidence if duplicate
                if label not in detected_fields or confidence > detected_fields[label]['confidence']:
                    detected_fields[label] = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(confidence)
                    }
        
        if not detected_fields:
            print("⚠️  No fields detected")
            return {}
        
        print(f"✅ Detected {len(detected_fields)} unique field(s)")
        return detected_fields
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def crop_and_save_fields(img, detected_fields: dict, output_dir: str):
    """Crop detected regions and save them"""
    section("STEP 4: Cropping Regions")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output: {output_dir}")
    
    field_crops = {}
    
    for label, data in detected_fields.items():
        x1, y1, x2, y2 = data['bbox']
        
        # Ensure within bounds
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Crop region
        crop = img[y1:y2, x1:x2]
        
        if crop.size == 0:
            print(f"⚠️  Skipping {label}: empty crop")
            continue
        
        field_crops[label] = crop
        
        # Normalize filename
        filename = label.lower().replace(' ', '_')
        crop_path = os.path.join(output_dir, f"{filename}.jpg")
        cv2.imwrite(crop_path, crop)
        print(f"   ✓ {label}: {crop.shape[1]}x{crop.shape[0]} -> {crop_path}")
    
    return field_crops

def clean_aadhaar_number(text: str) -> str:
    """Extract and format Aadhaar number"""
    # Remove all non-digits
    digits = re.sub(r'\D', '', text)
    
    if len(digits) >= 12:
        # Take first 12 digits and format as XXXX XXXX XXXX
        aadhaar = digits[:12]
        return f"{aadhaar[0:4]} {aadhaar[4:8]} {aadhaar[8:12]}"
    elif len(digits) > 0:
        return digits
    return ""

def clean_dob(text: str) -> str:
    """Clean date of birth"""
    # Replace common OCR errors
    text = text.replace('|', '/').replace('I', '1').replace('O', '0').replace('o', '0')
    # Keep only digits and slashes/dashes
    text = re.sub(r'[^0-9/\-]', '', text)
    return text.strip()

def clean_text(text: str) -> str:
    """General text cleaning"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()

def extract_text_with_easyocr(field_crops: dict):
    """Extract text from cropped regions using EasyOCR"""
    section("STEP 5: Extracting Text with EasyOCR")
    
    try:
        import easyocr
    except ImportError:
        print("❌ EasyOCR not installed")
        print("Install with: pip install easyocr")
        sys.exit(1)
    
    print("🔄 Initializing EasyOCR...")
    try:
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("✅ EasyOCR ready")
    except Exception as e:
        print(f"❌ Failed to initialize EasyOCR: {e}")
        sys.exit(1)
    
    extracted_data = {}
    
    for label, crop_img in field_crops.items():
        print(f"   🔍 Processing {label}...", end=" ")
        
        try:
            # Convert BGR to RGB for EasyOCR
            crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            
            # Run OCR
            result = reader.readtext(crop_rgb, detail=0, paragraph=True)
            
            if result and len(result) > 0:
                text = result[0].strip()
                
                # Clean based on field type
                if 'AADHAR' in label or 'NUMBER' in label:
                    text = clean_aadhaar_number(text)
                elif 'BIRTH' in label or 'DOB' in label:
                    text = clean_dob(text)
                else:
                    text = clean_text(text)
                
                extracted_data[label] = text
                print(f"✓ '{text}'")
            else:
                extracted_data[label] = ""
                print("⚠️  No text detected")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            extracted_data[label] = ""
    
    return extracted_data

def format_results(extracted_data: dict):
    """Format and display final results"""
    section("STEP 6: Extraction Results")
    
    print("\n📋 Extracted Aadhaar Information:")
    print("-" * 70)
    
    # Display order
    field_order = ['NAME', 'AADHAR_NUMBER', 'DATE_OF_BIRTH', 'GENDER', 'ADDRESS']
    
    for field in field_order:
        if field in extracted_data:
            value = extracted_data[field]
            display_name = field.replace('_', ' ').title()
            print(f"  {display_name:20s}: {value}")
    
    # Print any other fields
    for field, value in extracted_data.items():
        if field not in field_order:
            display_name = field.replace('_', ' ').title()
            print(f"  {display_name:20s}: {value}")
    
    print("-" * 70)
    
    return extracted_data

def main():
    """Main execution flow"""
    print("\n" + "=" * 70)
    print("🎯 AADHAAR EXTRACTION - YOLOv8 OpenVINO + EasyOCR")
    print("=" * 70)
    
    # Configuration
    IMAGE_PATH = PROJECT_ROOT / "scripts" / "aadhar_image" / "Aadhar both sides.png"
    OUTPUT_DIR = PROJECT_ROOT / "scripts" / "extracted_aadhaar_regions"
    CONFIDENCE_THRESHOLD = 0.25
    
    print(f"\n⚙️  Configuration:")
    print(f"   Model: YOLOv8 OpenVINO (int8)")
    print(f"   Image: {IMAGE_PATH}")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"   Confidence: {CONFIDENCE_THRESHOLD}")
    
    # Execute pipeline
    try:
        # 1. Load model
        compiled_model, class_names, input_layer, output_layer = download_and_load_model()
        
        # 2. Load image
        img = load_image(str(IMAGE_PATH))
        
        # 3. Detect fields
        detected_fields = detect_fields(compiled_model, img, class_names, input_layer, output_layer, CONFIDENCE_THRESHOLD)
        
        if not detected_fields:
            print("\n❌ FAILED: No fields detected")
            print("\nTroubleshooting:")
            print("  1. Ensure image shows a clear Aadhaar card")
            print("  2. Try with a different image")
            print("  3. Lower confidence threshold")
            sys.exit(1)
        
        # 4. Crop regions
        field_crops = crop_and_save_fields(img, detected_fields, str(OUTPUT_DIR))
        
        # 5. Extract text
        extracted_data = extract_text_with_easyocr(field_crops)
        
        # 6. Display results
        final_data = format_results(extracted_data)
        
        # Success summary
        section("✅ TEST COMPLETED SUCCESSFULLY")
        print(f"✓ Detected {len(detected_fields)} field(s)")
        print(f"✓ Saved {len(field_crops)} crop(s) to {OUTPUT_DIR}")
        print(f"✓ Extracted {len([v for v in extracted_data.values() if v])} text field(s)")
        
        print("\n💡 Results saved to:")
        print(f"   Cropped regions: {OUTPUT_DIR}")
        
        return final_data
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    result = main()
