import cv2
import xml.etree.ElementTree as ET

def decode_qr_opencv(image_path):
    image = cv2.imread(image_path)
    
    # Try detecting with original image first
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(image)

    if data:
        print("QR Code data:", data)
        return data
    
    # If not detected, try with preprocessing
    print("QR not detected with original image, trying preprocessing...")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    
    # Try detection with preprocessed image
    data, bbox, _ = detector.detectAndDecode(thresh)
    
    if data:
        print("QR Code data (after preprocessing):", data)
        return data
    
    # Try with increased contrast
    gray = cv2.equalizeHist(gray)
    data, bbox, _ = detector.detectAndDecode(gray)
    
    if data:
        print("QR Code data (after histogram equalization):", data)
        return data
    
    print("QR code not detected in the image even after preprocessing.")
    return None

def check_uid_last_4_digits(qr_data, ocr_uid):
    root = ET.fromstring(qr_data)
    qr_uid = root.get('uid', '')
    if qr_uid[-4:] == ocr_uid[-4:]:
        print("Last 4 digits of UID match.")
    else:
        print("Last 4 digits of UID do not match.")

# Example usage
image_path = 'scripts/qrcode.png'
qr_data = decode_qr_opencv(image_path)
ocr_uid = 'XXXXXXXX7743'
check_uid_last_4_digits(qr_data, ocr_uid)
