from face_matching_export import extract_and_store_embedding, compare_faces
from face_extraction_export import extract_adhaar_face
from qr_uid_matching_export import decode_qr_opencv, check_uid_last_4_digits
import os

def run_pipeline(aadhar_image_path, extracted_face_folder_path, extracted_face_path, comparison_image_path, qr_image_path, uid=None):
    """
    Run the complete KYC pipeline for Aadhaar verification.
    
    Args:
        aadhar_image_path: Path to Aadhaar image
        extracted_face_folder_path: Folder to save extracted face
        extracted_face_path: Path where extracted face will be saved
        comparison_image_path: Path to comparison image for face matching
        qr_image_path: Path to image containing QR code (usually same as aadhar_image_path)
        uid: UID from OCR for verification (optional)
    
    Returns:
        dict: Results containing face_extracted, qr_detected, uid_match, face_match
    """
    results = {
        'face_extracted': False,
        'qr_detected': False,
        'uid_match': False,
        'face_match': False,
        'qr_data': None,
        'error': None
    }
    
    try:
        # Step 1: Extract face from Aadhaar
        results['face_extracted'] = extract_adhaar_face(aadhar_image_path, extracted_face_folder_path)
        
        # Step 2: Decode QR code
        qr_data = decode_qr_opencv(qr_image_path)
        results['qr_detected'] = qr_data is not None
        results['qr_data'] = qr_data
        
        # Step 3: Check UID match if both QR and UID provided
        if qr_data and uid:
            results['uid_match'] = check_uid_last_4_digits(qr_data, uid)
        
        # Step 4: Face matching (only if face was extracted)
        if results['face_extracted'] and os.path.exists(extracted_face_path):
            try:
                extract_and_store_embedding(extracted_face_path)
                results['face_match'] = compare_faces(extracted_face_path, comparison_image_path)
            except Exception as e:
                results['error'] = f"Face matching error: {str(e)}"
        
        return results
        
    except Exception as e:
        results['error'] = str(e)
        return results

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    aadhar_image_path = os.path.join(PROJECT_ROOT, "scripts/aadhar_image/aadhar.png")
    extracted_face_folder_path = os.path.join(PROJECT_ROOT, "scripts/extracted_face_image")
    extracted_face_path = os.path.join(PROJECT_ROOT, "scripts/extracted_face_image/extracted_face.jpg")
    comparison_image_path = os.path.join(PROJECT_ROOT, "scripts/comparison_image/comparison_Img.JPG")
    qr_image_path = os.path.join(PROJECT_ROOT, "scripts/aadhar_image/aadhar.png")
    
    uid = 'XXXXXXXX7743'  # Later to be replaced with OCR UID function
    results = run_pipeline(aadhar_image_path, extracted_face_folder_path, extracted_face_path, 
                          comparison_image_path, qr_image_path, uid)
    
    print("\n=== Pipeline Results ===")
    print(f"Face extracted: {results['face_extracted']}")
    print(f"QR detected: {results['qr_detected']}")
    print(f"UID match: {results['uid_match']}")
    print(f"Face match: {results['face_match']}")
    if results['error']:
        print(f"Error: {results['error']}")



#Reffrence for Backend Intergration
    
# from pan_ocr import ExtractDetails
# from face_extraction_export import extract_adhaar_face
# from face_matching_export import extract_and_store_embedding, compare_faces
# from qr_uid_matching_export import decode_qr_opencv, check_uid_last_4_digits


#Must Be added for due to a strange file structure
    

# scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'scripts'))
# sys.path.append(scripts_path)

# ocr_scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'ocr_scripts'))
# sys.path.append(ocr_scripts_path)