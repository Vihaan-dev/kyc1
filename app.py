from flask import Flask, request, jsonify
from flask_cors import CORS
import shutil
import os
import sys

# Add scripts and ocr_scripts directories to Python path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.insert(0, scripts_path)

ocr_scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ocr_scripts'))
sys.path.insert(0, ocr_scripts_path)

# Import extraction modules
from pan_advanced_extraction import extract_pan_fields
from aadhaar_advanced_extraction import extract_aadhaar_fields
from face_extraction_export import extract_adhaar_face
from face_matching_export import extract_and_store_embedding, compare_faces

app = Flask(__name__)
CORS(app)

ADHAAR_IMAGE = "scripts/aadhar_image"
EXTRACTED_FACE_IMAGE = "scripts/extracted_face_image"
COMPARISON_IMAGE = "scripts/comparison_image"
PANCARD_IMAGE = "scripts/pancard_image"

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/aadhar-upload', methods=['POST'])
def aadhar_upload():
    """
    Upload and process Aadhaar card image using YOLO + OCR extraction.
    Returns: aadhaar_number, name, dob, gender, address, photo
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        file_path = os.path.join(ADHAAR_IMAGE, file.filename)
        os.makedirs(ADHAAR_IMAGE, exist_ok=True)
        file.save(file_path)
        
        print(f"\n{'='*70}")
        print(f"📄 Processing Aadhaar: {file.filename}")
        print(f"📁 File saved to: {file_path}")
        print(f"📊 File exists: {os.path.exists(file_path)}")
        print(f"📏 File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'} bytes")
        print(f"{'='*70}\n")

        # Extract Aadhaar fields using YOLO + OCR
        print("🔍 Calling extract_aadhaar_fields()...")
        aadhaar_data = extract_aadhaar_fields(file_path)
        
        print(f"\n📊 Extraction Results:")
        print(f"   - OCR Engine: {aadhaar_data.get('ocr_engine', 'N/A')}")
        print(f"   - Name: {aadhaar_data.get('name', '')}")
        print(f"   - Aadhaar: {aadhaar_data.get('aadhaar_number', '')}")
        print(f"   - DOB: {aadhaar_data.get('dob', '')}")
        print(f"   - Gender: {aadhaar_data.get('gender', '')}")
        print(f"   - Address: {aadhaar_data.get('address', '')[:50]}..." if aadhaar_data.get('address') else "   - Address: ")
        print(f"   - Photo: {aadhaar_data.get('photo', '')}")
        print()
        
        # Check if extraction was successful
        extraction_success = bool(
            aadhaar_data.get('name') or 
            aadhaar_data.get('aadhaar_number')
        )
        
        # Build clean response - only extracted data
        response = {
            'message': 'Aadhaar uploaded and processed successfully',
            'success': extraction_success,
            'aadhaar_number': aadhaar_data.get('aadhaar_number', ''),
            'name': aadhaar_data.get('name', ''),
            'dob': aadhaar_data.get('dob', ''),
            'gender': aadhaar_data.get('gender', ''),
            'address': aadhaar_data.get('address', ''),
            'photo': aadhaar_data.get('photo', ''),
        }
        
        print(f"✅ Returning response with success={extraction_success}\n")

        return jsonify(response), 200
        
    except Exception as e:
        print(f"\n❌ ERROR in aadhar_upload: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    

@app.route('/pan-upload', methods=['POST'])
def pan_upload():
    """
    Upload and process PAN card image using YOLO + OCR extraction.
    Returns: pan_number, name, father_name, dob, photo
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save file
        file_path = os.path.join(PANCARD_IMAGE, file.filename)
        os.makedirs(PANCARD_IMAGE, exist_ok=True)
        file.save(file_path)
        
        print(f"\n{'='*70}")
        print(f"📄 Processing PAN Card: {file.filename}")
        print(f"📁 File saved to: {file_path}")
        print(f"📊 File exists: {os.path.exists(file_path)}")
        print(f"📏 File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'} bytes")
        print(f"{'='*70}\n")
        
        # Extract PAN details using YOLO + EasyOCR
        print("🔍 Calling extract_pan_fields()...")
        pan_data = extract_pan_fields(file_path)
        
        print(f"\n📊 Extraction Results:")
        print(f"   - OCR Engine: {pan_data.get('ocr_engine', 'N/A')}")
        print(f"   - PAN Number: {pan_data.get('pan_number', '')}")
        print(f"   - Name: {pan_data.get('name', '')}")
        print(f"   - Father Name: {pan_data.get('father_name', '')}")
        print(f"   - DOB: {pan_data.get('dob', '')}")
        print()
        
        # Check if extraction was successful
        extraction_success = bool(pan_data.get('pan_number'))
        
        # Build clean response
        response = {
            'message': 'PAN Card uploaded and processed successfully',
            'success': extraction_success,
            'pan_number': pan_data.get('pan_number', ''),
            'name': pan_data.get('name', ''),
            'father_name': pan_data.get('father_name', ''),
            'dob': pan_data.get('dob', ''),
            'photo': pan_data.get('photo', ''),
        }
        
        print(f"✅ Returning response with success={extraction_success}\n")

        return jsonify(response), 200
        
    except Exception as e:
        print(f"\n❌ ERROR in pan_upload: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/livephoto-upload', methods=['POST'])
def livephoto_upload():
    """Upload and compare live photo with extracted Aadhaar face."""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file part in request'
            }), 400
        
        file = request.files['file']
        
        if not file.filename or file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        # Save file
        file_path = os.path.join(COMPARISON_IMAGE, file.filename)
        os.makedirs(COMPARISON_IMAGE, exist_ok=True)
        file.save(file_path)

        # Extract and compare face
        extracted_face_path = os.path.join(EXTRACTED_FACE_IMAGE, "extracted_face.jpg")
        
        if not os.path.exists(extracted_face_path):
            return jsonify({
                'success': False,
                'error': 'No Aadhaar face found. Please upload Aadhaar first.'
            }), 400

        # Lazy import face matching (avoid NumPy/TensorFlow conflicts)
        from face_matching_export import extract_and_store_embedding, compare_faces
        
        extract_and_store_embedding(file_path)
        face_match = compare_faces(extracted_face_path, file_path)

        return jsonify({
            'success': True,
            'message': 'Live photo uploaded and compared successfully',
            'face_matching': face_match
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
