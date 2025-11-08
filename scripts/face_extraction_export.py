# from PIL import Image
# from io import BytesIO
# import numpy as np
# import requests
import cv2
import os

def pick_largest_face(faces):
    """
    Pick the largest face from detected faces array.
    Returns None if no faces, otherwise returns (x, y, w, h) tuple.
    """
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])

def extract_adhaar_face(aadhar_image_path, extracted_face_path):
    """
    Extract face from Aadhaar image using robust detection methods.
    Tries multiple preprocessing techniques and upscaling if needed.
    Returns True if face extracted successfully, False otherwise.
    """
    # Uncomment to use URL
    # response = requests.get(aadhar_image_path)
    # image = Image.open(BytesIO(response.content))
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Read image
    image = cv2.imread(aadhar_image_path)
    if image is None:
        return False
    
    # Use absolute path to cascade classifier
    cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
    if not os.path.exists(cascade_path):
        return False
    
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Try multiple preprocessing variants
    variants = []
    variants.append(("original", image))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants.append(("gray", gray))
    
    # Histogram equalization for better contrast
    eq = cv2.equalizeHist(gray)
    variants.append(("hist-eq", eq))

    found_box = None
    
    # Try detection on each variant
    for name, candidate in variants:
        if len(candidate.shape) == 3:
            candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
        else:
            candidate_gray = candidate
        
        faces = face_cascade.detectMultiScale(
            candidate_gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            found_box = pick_largest_face(faces)
            break

    # If still not found, try upscaling
    if found_box is None:
        up = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        up_gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            up_gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        if len(faces) > 0:
            found_box = pick_largest_face(faces)
            # Scale down coordinates back to original image size
            if found_box is not None:
                x, y, w, h = found_box
                found_box = (int(x/1.5), int(y/1.5), int(w/1.5), int(h/1.5))

    # If no face detected
    if found_box is None:
        return False

    # Extract and save face
    x, y, w, h = found_box
    face_roi = image[y:y+h, x:x+w]
    
    # Ensure output directory exists
    os.makedirs(extracted_face_path, exist_ok=True)
    output_path = os.path.join(extracted_face_path, "extracted_face.jpg")
    cv2.imwrite(output_path, face_roi)
    
    return True

if __name__ == "__main__":
    import sys
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    aadhar_image_path = os.path.join(PROJECT_ROOT, "scripts/aadhar_image/aadhar.png")
    extracted_face_path = os.path.join(PROJECT_ROOT, "scripts/extracted_face_image")
    extract_adhaar_face(aadhar_image_path, extracted_face_path)