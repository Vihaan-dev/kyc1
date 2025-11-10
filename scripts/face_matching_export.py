import cv2
import os
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace


def crop_face_with_retinaface(image_path, resize_dim=(112, 112)):

    output_dir = os.path.dirname(image_path)

    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not load image: {image_path}")
        return None

    try:
        faces = RetinaFace.detect_faces(image_path)
    except Exception as e:
        print(f"RetinaFace error on {image_path}: {e}")
        return None

    if isinstance(faces, dict) and len(faces) > 0:
        best_face = None
        max_area = 0
        for _, face in faces.items():
            x1, y1, x2, y2 = face["facial_area"]
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_face = (x1, y1, x2, y2)
        x1, y1, x2, y2 = best_face
        cropped = img[y1:y2, x1:x2]
        cropped = cv2.resize(cropped, resize_dim)

        base = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"cropped_{base}")
        cv2.imwrite(output_path, cropped)
        return output_path
    else:
        print(f"No face detected in {image_path}")
        return None

def compare_faces(reference_image_path, live_image_path):
    try:
        # Verify both images exist
        if not os.path.exists(reference_image_path):
            raise FileNotFoundError(f"Reference image not found: {reference_image_path}")
        if not os.path.exists(live_image_path):
            raise FileNotFoundError(f"Live image not found: {live_image_path}")
        
        print(f"Comparing faces:")
        print(f"  Reference: {reference_image_path}")
        print(f"  Live: {live_image_path}")
        
        cropped_ref_img_path = crop_face_with_retinaface(reference_image_path)
        cropped_live_img_path = crop_face_with_retinaface(live_image_path)

        result = DeepFace.verify(cropped_ref_img_path, cropped_live_img_path
          , model_name = 'ArcFace', detector_backend = 'retinaface')

        print(f"Verification result: {result}")

        distance = result.get('distance', None)
        threshold = result.get('threshold', None)
        verified = result.get('verified', False)
        confidence = result.get('confidence', None)
        
        if result['verified']:
            print(f"✓ Faces match! (Distance: {distance:.4f}, Threshold: {threshold:.4f}, Confidence: {confidence:.4f})")
        else:
            print(f"✗ Faces do not match! (Distance: {distance:.4f}, Threshold: {threshold:.4f}, Confidence: {confidence:.4f})")
        
        return {
            'verified': verified,
            'distance': distance,
            'threshold': threshold,
            'confidence': confidence,
        }
            
    except Exception as e:
        print(f"Error comparing faces: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    extracted_face_path = os.path.join(PROJECT_ROOT, "scripts/extracted_face_image/extracted_face.jpg")
    live_image_path = os.path.join(PROJECT_ROOT, "scripts/comparison_image/comparison_Img.JPG")
    compare_faces(extracted_face_path, live_image_path)