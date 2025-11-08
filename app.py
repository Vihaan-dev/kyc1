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

# Import new PAN extraction module
from pan_advanced_extraction import extract_pan_fields
from face_extraction_export import extract_adhaar_face
# Lazy import for face_matching to avoid NumPy/TensorFlow conflicts at startup
# from face_matching_export import extract_and_store_embedding, compare_faces
from qr_uid_matching_export import decode_qr_opencv, check_uid_last_4_digits

app = Flask(__name__)
CORS(app)

ADHAAR_IMAGE = "scripts/aadhar_image"
EXTRACTED_FACE_IMAGE = "scripts/extracted_face_image"
COMPARISON_IMAGE = "scripts/comparison_image"
PANCARD_IMAGE = "scripts/pancard_image"
SIGNATURE_IMAGE = "scripts/signature_image"
PASSPORT_SIZE_IMAGE = "scripts/passport_size_image"

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/aadhar-upload', methods=['POST'])
def aadhar_upload():
    """
    Upload and process Aadhaar card image.
    - Extracts face from Aadhaar
    - Decodes QR code with robust detection
    - Validates UID (if provided)
    """
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

        # Save uploaded file
        file_path = os.path.join(ADHAAR_IMAGE, file.filename)
        os.makedirs(ADHAAR_IMAGE, exist_ok=True)
        file.save(file_path)

        # Process the Aadhaar image
        results = {
            'success': True,
            'message': 'Aadhaar uploaded successfully',
            'face_extraction': False,
            'qr_detected': False,
            'qr_data': None,
            'uid_match': None
        }

        # Step 1: Extract QR code data
        qr_data = decode_qr_opencv(file_path)
        results['qr_detected'] = qr_data is not None
        
        if qr_data:
            results['qr_data'] = qr_data
            
            # Try to parse UID from QR data
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(qr_data)
                results['uid'] = root.get('uid', 'N/A')
                results['name'] = root.get('name', 'N/A')
                results['dob'] = root.get('dob', root.get('yob', 'N/A'))
                results['gender'] = root.get('gender', 'N/A')
            except Exception as e:
                results['qr_parse_error'] = str(e)

        # Step 2: Extract face from Aadhaar
        face_extracted = extract_adhaar_face(file_path, EXTRACTED_FACE_IMAGE)
        results['face_extraction'] = face_extracted

        # Step 3: Check UID match if OCR UID is provided in request
        ocr_uid = request.form.get('ocr_uid')
        if qr_data and ocr_uid:
            uid_match = check_uid_last_4_digits(qr_data, ocr_uid)
            results['uid_match'] = uid_match

        # Determine overall success
        if not qr_data and not face_extracted:
            results['warning'] = 'Could not extract QR code or face from image. Please ensure image is clear and contains Aadhaar card.'
        elif not qr_data:
            results['warning'] = 'QR code not detected. Face extracted successfully.'
        elif not face_extracted:
            results['warning'] = 'Face not detected. QR code extracted successfully.'
        else:
            results['message'] = 'Aadhaar processed successfully - QR code and face extracted'

        return jsonify(results), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500
    

@app.route('/pan-upload', methods=['POST'])
def pan_upload():
    """Upload and process PAN card image with advanced YOLO + EasyOCR extraction."""
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
        file_path = os.path.join(PANCARD_IMAGE, file.filename)
        os.makedirs(PANCARD_IMAGE, exist_ok=True)
        file.save(file_path)
        
        # Extract PAN details using advanced YOLO + EasyOCR
        result = extract_pan_fields(file_path)
        
        # Check if extraction was successful
        if not result.get('pan_number'):
            return jsonify({
                'success': False,
                'error': 'Failed to extract PAN number from image',
                'ocr_engine': result.get('ocr_engine')
            }), 400

        return jsonify({
            'success': True,
            'message': 'PAN card uploaded and processed successfully',
            'pan_number': result.get('pan_number'),
            'dob': result.get('dob'),
            'name': result.get('name'),
            'father_name': result.get('father_name'),
            'ocr_engine': result.get('ocr_engine')
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/signature-upload', methods=['POST'])
def signature_upload():
    """Upload signature image."""
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
        file_path = os.path.join(SIGNATURE_IMAGE, file.filename)
        os.makedirs(SIGNATURE_IMAGE, exist_ok=True)
        file.save(file_path)
        
        return jsonify({
            'success': True,
            'message': 'Signature uploaded successfully'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

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
    

@app.route('/passport-photo-upload', methods=['POST'])
def passport_photo_upload():
    """Upload and compare passport photo with live photo."""
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
        file_path = os.path.join(PASSPORT_SIZE_IMAGE, file.filename)
        os.makedirs(PASSPORT_SIZE_IMAGE, exist_ok=True)
        file.save(file_path)

        # Compare with live photo
        live_photo_path = os.path.join(COMPARISON_IMAGE, "live_photo.jpg")
        
        if not os.path.exists(live_photo_path):
            return jsonify({
                'success': False,
                'error': 'No live photo found. Please upload live photo first.'
            }), 400

        # Lazy import face matching (avoid NumPy/TensorFlow conflicts)
        from face_matching_export import extract_and_store_embedding, compare_faces
        
        extract_and_store_embedding(file_path)
        face_match = compare_faces(file_path, live_photo_path)

        return jsonify({
            'success': True,
            'message': 'Passport photo uploaded and compared successfully',
            'face_matching': face_match
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
