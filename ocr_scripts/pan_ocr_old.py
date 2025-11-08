import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
#Tesseract Library
import pytesseract

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

### Make prettier the prints ###
from colorama import Fore, Style
c_ = Fore.CYAN
m_ = Fore.MAGENTA
r_ = Fore.RED
b_ = Fore.BLUE
y_ = Fore.YELLOW
g_ = Fore.GREEN
w_ = Fore.WHITE

import warnings
warnings.filterwarnings(action='ignore') 

def ExtractDetails(image_path):
    # Read and preprocess image for better OCR
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to preprocess the image
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Resize image to improve OCR (tesseract works better with larger text)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Perform OCR
    text = pytesseract.image_to_string(gray, lang='eng', config='--psm 6')
    print(f"Extracted text: {text}")  # Debug output
    
    text = text.replace("\n", " ")
    text = text.replace("  ", " ")
    text = text.upper()  # Convert to uppercase for better matching
    
    regex_DOB = re.compile(r'\d{2}[-/]\d{2}[-/]\d{4}')
    regex_num = re.compile(r'[A-Z]{5}[0-9]{4}[A-Z]{1}')
    
    pan_numbers = regex_num.findall(text)
    dob_dates = regex_DOB.findall(text)
    
    if len(pan_numbers) == 0:
        print(f'{y_}Could not extract PAN number from image')
        print(f'Extracted text was: {text[:200]}')  # Show first 200 chars
        print(Style.RESET_ALL)
        pan_number = None
    else:
        pan_number = pan_numbers[0]
        print(f'{g_}Found PAN: {pan_number}{Style.RESET_ALL}')
        
    if len(dob_dates) == 0:
        print(f'{y_}Could not extract DOB from image')
        print(Style.RESET_ALL)
        dob = None
    else:
        dob = dob_dates[0]
        print(f'{g_}Found DOB: {dob}{Style.RESET_ALL}')
        
    # Return None if data couldn't be extracted
    if pan_number is None or dob is None:
        raise ValueError(f"Could not extract PAN details. PAN: {pan_number}, DOB: {dob}. Check image quality and ensure text is clearly visible.")
    
    result = [pan_number, dob]
    return result

if __name__ == '__main__':
    import os
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    test_image_path = os.path.join(PROJECT_ROOT, 'ocr_scripts/pancard_try.jpeg')
    ExtractDetails(test_image_path)