import cv2
from deepface import DeepFace
import numpy as np
import os

def extract_and_store_embedding(reference_image_path):
    try:
        # Verify image exists and is valid
        if not os.path.exists(reference_image_path):
            raise FileNotFoundError(f"Reference image not found: {reference_image_path}")
        
        # Check if image can be loaded
        img = cv2.imread(reference_image_path)
        if img is None:
            raise ValueError(f"Could not load image: {reference_image_path}")
        
        print(f"Extracting embedding from: {reference_image_path}")
        
        # Extract embedding with error handling
        embedding_result = DeepFace.represent(
            img_path=reference_image_path, 
            model_name="VGG-Face",
            enforce_detection=False  # Allow processing even if face detection is uncertain
        )
        
        # DeepFace.represent returns a list of dictionaries
        if isinstance(embedding_result, list) and len(embedding_result) > 0:
            embedding = embedding_result[0]['embedding']
        else:
            raise ValueError("No face detected or embedding extraction failed")
        
        # Use absolute path for storing embedding
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        embedding_path = os.path.join(PROJECT_ROOT, "stored_embedding.npy")
        
        np.save(embedding_path, embedding)
        print(f"Embedding stored at: {embedding_path}")
        
        return embedding
        
    except Exception as e:
        print(f"Error extracting embedding: {str(e)}")
        raise

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
        
        # Perform face verification
        result = DeepFace.verify(
            img1_path=reference_image_path, 
            img2_path=live_image_path, 
            model_name="VGG-Face", 
            distance_metric="cosine",
            enforce_detection=False  # Allow processing even if face detection is uncertain
        )

        print(f"Verification result: {result}")
        
        if result['verified']:
            print(f"✓ Faces match! (Distance: {result['distance']:.4f}, Threshold: {result['threshold']:.4f})")
            return True
        else:
            print(f"✗ Faces do not match! (Distance: {result['distance']:.4f}, Threshold: {result['threshold']:.4f})")
            return False
            
    except Exception as e:
        print(f"Error comparing faces: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    extracted_face_path = os.path.join(PROJECT_ROOT, "scripts/extracted_face_image/extracted_face.jpg")
    live_image_path = os.path.join(PROJECT_ROOT, "scripts/comparison_image/comparison_Img.JPG")
    extract_and_store_embedding(extracted_face_path)
    compare_faces(extracted_face_path, live_image_path)