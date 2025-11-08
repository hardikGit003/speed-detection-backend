import cv2
import numpy as np
import pytesseract
from PIL import Image

class LicensePlateRecognizer:
    def __init__(self):
        # Cascade classifier for license plate detection
        self.plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
        
    def preprocess_image(self, image):
        """Preprocess the image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to remove noise while keeping edges sharp
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Edge detection
        edged = cv2.Canny(gray, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        return gray, contours
    
    def find_plate_contour(self, contours):
        """Find the contour that most likely represents a license plate"""
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
            
            # License plate should be rectangular
            if len(approx) == 4:
                return approx
        return None
    
    def extract_plate(self, image, plate_contour):
        """Extract and preprocess the license plate region"""
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(plate_contour)
        
        # Extract plate region
        plate = image[y:y+h, x:x+w]
        
        # Resize for better OCR
        plate = cv2.resize(plate, (w*2, h*2))
        
        # Convert to grayscale
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return plate_thresh, (x, y, w, h)
    
    def recognize_plate(self, image):
        """Main method to recognize license plate in an image"""
        # Preprocess image
        gray, contours = self.preprocess_image(image)
        
        # Find plate contour
        plate_contour = self.find_plate_contour(contours)
        
        if plate_contour is None:
            return None, None
        
        # Extract and preprocess plate
        plate_thresh, (x, y, w, h) = self.extract_plate(image, plate_contour)
        
        # Convert to PIL Image for Tesseract
        plate_pil = Image.fromarray(plate_thresh)
        
        # Perform OCR
        plate_text = pytesseract.image_to_string(plate_pil, config='--psm 7')
        
        # Clean up the text
        plate_text = ''.join(c for c in plate_text if c.isalnum())
        
        return plate_text, (x, y, w, h)
    
    def process_frame(self, frame):
        """Process a single frame and draw results"""
        # Detect plates using cascade classifier
        plates = self.plate_cascade.detectMultiScale(frame, 1.1, 4)
        
        for (x, y, w, h) in plates:
            # Extract plate region
            plate_region = frame[y:y+h, x:x+w]
            
            # Recognize plate text
            plate_text, _ = self.recognize_plate(plate_region)
            
            if plate_text:
                # Draw rectangle around plate
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add plate text
                cv2.putText(frame, plate_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame 