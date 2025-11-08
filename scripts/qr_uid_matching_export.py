import cv2
import xml.etree.ElementTree as ET
import os

def rotate_image(img, angle):
    """Rotate image by angle (0, 90, 180, 270 degrees)."""
    if angle % 360 == 0:
        return img
    if angle % 360 == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle % 360 == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle % 360 == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def try_qr_opencv(img):
    """Try QR detection with OpenCV using multiple preprocessing techniques."""
    detector = cv2.QRCodeDetector()

    def detect(candidate, tag=""):
        data, bbox, _ = detector.detectAndDecode(candidate)
        if data:
            return data
        return None

    # Try multiple variations: rotations, scales, preprocessing
    rotations = [0, 90, 180, 270]
    scales = [1.0, 1.5, 2.0]

    for r in rotations:
        img_r = rotate_image(img, r)

        # Try raw image
        got = detect(img_r)
        if got:
            return got

        # Try grayscale
        gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        got = detect(gray)
        if got:
            return got

        # Try adaptive threshold
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        got = detect(thr)
        if got:
            return got

        # Try histogram equalization
        eq = cv2.equalizeHist(gray)
        got = detect(eq)
        if got:
            return got

        # Try upscaled versions
        for s in scales:
            if s == 1.0:
                continue  # Skip 1.0 as we already tried it
            up = cv2.resize(img_r, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            got = detect(up)
            if got:
                return got

    return None

def try_qr_pyzbar(img):
    """Fallback: try pyzbar library with rotations."""
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
    except ImportError:
        return None

    rotations = [0, 90, 180, 270]
    for r in rotations:
        img_r = rotate_image(img, r)
        decoded = pyzbar_decode(img_r)
        if decoded:
            data = decoded[0].data.decode('utf-8', errors='ignore')
            return data
    return None

def decode_qr_opencv(image_path):
    """
    Robust QR code detection with multiple preprocessing techniques.
    Returns QR data string or None if not detected.
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Try OpenCV detector with preprocessing
    data = try_qr_opencv(image)
    
    # Fallback to pyzbar if available
    if not data:
        data = try_qr_pyzbar(image)

    return data

def check_uid_last_4_digits(qr_data, ocr_uid):
    """
    Compare last 4 digits of UID from QR data with OCR UID.
    Returns True if match, False otherwise.
    Handles None/empty values gracefully.
    """
    if not qr_data or not ocr_uid:
        return False
    
    try:
        root = ET.fromstring(qr_data)
        qr_uid = root.get('uid', '')
        
        if not qr_uid or len(qr_uid) < 4 or len(ocr_uid) < 4:
            return False
        
        is_match = qr_uid[-4:] == ocr_uid[-4:]
        return is_match
    except Exception as e:
        # Handle XML parsing errors gracefully
        return False

# Example usage
if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    image_path = os.path.join(PROJECT_ROOT, 'scripts/aadhar_image/aadhar.png')
    ocr_uid = 'XXXXXXXX7743'  # Replace with your actual UID
    qr_data = decode_qr_opencv(image_path)
    if qr_data:
        check_uid_last_4_digits(qr_data, ocr_uid)