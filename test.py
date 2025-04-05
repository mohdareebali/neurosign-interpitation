import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize Camera
cap = cv2.VideoCapture(0)

# Initialize Hand Detector & Classifier
detector = HandDetector(maxHands=1)
classifier = Classifier(r"D:\tabrez KBN\Sign-Language-detection\Model\keras_model.h5", 
                        r"D:\tabrez KBN\Sign-Language-detection\Model\labels.txt")

offset = 20
imgSize = 300

# Labels for predictions
labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

while True:
    success, img = cap.read()
    
    if not success:
        print("Failed to capture image")
        continue  # Skip iteration if the frame is not captured properly

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region safely
        y1, y2 = max(0, y-offset), min(img.shape[0], y+h+offset)
        x1, x2 = max(0, x-offset), min(img.shape[1], x+w+offset)
        imgCrop = img[y1:y2, x1:x2]

        # Check if imgCrop is valid
        if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
            print("Invalid crop detected, skipping frame...")
            continue

        # Aspect ratio calculation
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        # Get Prediction
        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Ensure valid index
        label_text = labels[index] if 0 <= index < len(labels) else "Unknown"

        # Draw label box
        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), 
                      (x - offset + 400, y - offset + 60 - 50), 
                      (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, label_text, (x, y - 30), 
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), 
                      (x + w + offset, y + h + offset), 
                      (0, 255, 0), 4)

        # Show images
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    key = cv2.waitKey(1)

    if key == 27:  # Press 'Esc' to exit
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
