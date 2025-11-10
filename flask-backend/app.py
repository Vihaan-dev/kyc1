from flask import Flask, request, jsonify
from flask_cors import CORS
import shutil
import os
import sys

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

scripts_path = os.path.join(PROJECT_ROOT, 'scripts')
sys.path.append(scripts_path)

ocr_scripts_path = os.path.join(PROJECT_ROOT, 'ocr_scripts')
sys.path.append(ocr_scripts_path)

from pan_advanced_extraction import extract_pan_fields
from aadhaar_advanced_extraction import extract_aadhaar_fields
from face_extraction_export import extract_adhaar_face
from face_matching_export import compare_faces
from qr_uid_matching_export import decode_qr_opencv, check_uid_last_4_digits

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"], "allow_headers": ["Content-Type"]}})

# Set max file size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Use absolute paths from project root
ADHAAR_IMAGE = os.path.join(PROJECT_ROOT, 'scripts', 'aadhar_image')
EXTRACTED_FACE_IMAGE = os.path.join(PROJECT_ROOT, 'scripts', 'extracted_face_image')
COMPARISON_IMAGE = os.path.join(PROJECT_ROOT, 'scripts', 'comparison_image')
PANCARD_IMAGE = os.path.join(PROJECT_ROOT, 'scripts', 'pancard_image')
SIGNATURE_IMAGE = os.path.join(PROJECT_ROOT, 'scripts', 'signature_image')
PASSPORT_SIZE_IMAGE = os.path.join(PROJECT_ROOT, 'scripts', 'passport_size_image')

# Ensure directories exist
for directory in [ADHAAR_IMAGE, EXTRACTED_FACE_IMAGE, COMPARISON_IMAGE, PANCARD_IMAGE, SIGNATURE_IMAGE, PASSPORT_SIZE_IMAGE]:
    os.makedirs(directory, exist_ok=True)

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/aadhar-upload', methods=['POST'])
def aadhar_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        file_path = os.path.join(ADHAAR_IMAGE, file.filename)
        file.save(file_path)

        # Extract Aadhaar fields using YOLO + OCR
        aadhaar_data = extract_aadhaar_fields(file_path)
        
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

        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error in aadhar_upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/pan-upload', methods=['POST'])
def pan_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        file_path = os.path.join(PANCARD_IMAGE, file.filename)
        file.save(file_path)
        
        # Extract PAN fields using advanced YOLO + OCR
        pan_data = extract_pan_fields(file_path)
        
        # Check if extraction was successful
        extraction_success = bool(pan_data.get('pan_number'))

        return jsonify({
            'message': 'PAN Card uploaded and processed successfully',
            'success': extraction_success,
            'pan_number': pan_data.get('pan_number', ''),
            'name': pan_data.get('name', ''),
            'father_name': pan_data.get('father_name', ''),
            'dob': pan_data.get('dob', ''),
            'photo': pan_data.get('photo', ''),
        }), 200
    except Exception as e:
        print(f"Error in pan_upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/signature-upload', methods=['POST'])
def signature_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    # Do something with the uploaded signature file
    #return jsonify({'message': 'Signature uploaded successfully'})
    if file.filename != '':  # Check if filename is not empty
        file_path = os.path.join(SIGNATURE_IMAGE, file.filename)
        file.save(file_path)
        return jsonify({'message': 'Signature uploaded and stored successfully'})
    else:
        return jsonify({'error': 'Invalid file'})

@app.route('/livephoto-upload', methods=['POST'])
def livephoto_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        file_path = os.path.join(COMPARISON_IMAGE, file.filename)
        file.save(file_path)

        aadhar_extracted_face_path = os.path.join(EXTRACTED_FACE_IMAGE, "aadhar_photo.jpg")
        pan_extracted_face_path = os.path.join(EXTRACTED_FACE_IMAGE, 'pan_photo.jpg')
        aadhar_check_face_result = compare_faces(aadhar_extracted_face_path, file_path)
        pan_check_face_result = compare_faces(pan_extracted_face_path, file_path)

        return jsonify({
            'message': 'Live photo uploaded and stored successfully',
            'AADHAR_face_matching': aadhar_check_face_result,
            'PAN_face_matching': pan_check_face_result
        }), 200
    except Exception as e:
        print(f"Error in livephoto_upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    

@app.route('/passport-photo-upload', methods=['POST'])
def passport_photo_upload():
    global COMPARISON_IMAGE
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    # Do something with the uploaded passport photo file
    # return jsonify({'message': 'Passport photo uploaded successfully'})
    if file.filename != '':  # Check if filename is not empty
        file_path = os.path.join(PASSPORT_SIZE_IMAGE, file.filename)
        file.save(file_path)

        extract_and_store_embedding(file_path)
        check_face_matching = compare_faces(file_path, COMPARISON_IMAGE + "/live_photo.jpg")

        return jsonify({
            'message': 'Passport photo uploaded and stored successfully',
            'face_matching': check_face_matching
            })
    else:
        return jsonify({'error': 'Invalid file'})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
