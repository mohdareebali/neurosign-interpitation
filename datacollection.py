import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Set up camera
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

# Corrected folder path (Windows requires double backslashes or raw string)
folder = r"D:\tabrez KBN\Sign-Language-detection\Data\Yes"

# Ensure the folder exists
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to read from camera!")
        continue

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure cropping does not go out of bounds
        hImg, wImg, _ = img.shape
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(wImg, x + w + offset), min(hImg, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            print("⚠️ Skipping empty crop frame!")
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Determine aspect ratio
        aspectRatio = h / w

        if aspectRatio > 1:  # Tall image
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:  # Wide image
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)

    if key == ord("s"):  # Save image when 's' is pressed
        file_path = os.path.join(folder, f"Image_{int(time.time())}.jpg")
        success = cv2.imwrite(file_path, imgWhite)

        if success:
            counter += 1
            print(f"✅ Image {counter} saved: {file_path}")
        else:
            print("❌ Failed to save image!")

    if key == ord("q"):  # Quit when 'q' is pressed
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
