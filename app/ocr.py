import cv2              # OpenCV for image processing
import pytesseract      # Tesseract OCR library
from PIL import Image   # Pillow for image handling (if needed)
import numpy as np      # Numpy for numerical operations
import re               # Regular expressions for pattern matching

def preprocess_image(image_path):
    # Read image
    image = cv2.imread(image_path)

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return gray

