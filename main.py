from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2 as cv
import numpy as np

app = FastAPI(title="Healthcare AI - Dark Circle Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load cascade files
eyes_area = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")
face_area = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_dark_circles(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Histogram equalization helps normalize lighting
    gray = cv.equalizeHist(gray)
    
    faces = face_area.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return {"status": "No face detected", "remedies": []}

    max_severity = 0

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eyes_area.detectMultiScale(roi_gray, 1.1, 10)

        for (ex, ey, ew, eh) in eyes:
            # Define the area immediately BELOW the eye
            # We look about 20% of the eye's height further down
            under_eye_y = ey + int(eh * 1.1)
            under_eye_h = int(eh * 0.4)
            
            # Ensure we don't go out of bounds of the face ROI
            if under_eye_y + under_eye_h > h:
                continue

            under_eye_region = roi_gray[under_eye_y : under_eye_y + under_eye_h, ex : ex + ew]
            
            # Reference region (Cheek) to compare skin tone
            cheek_y = under_eye_y + under_eye_h + 10
            cheek_h = int(eh * 0.4)
            if cheek_y + cheek_h > h:
                cheek_region = under_eye_region # Fallback
            else:
                cheek_region = roi_gray[cheek_y : cheek_y + cheek_h, ex : ex + ew]

            avg_under_eye = np.mean(under_eye_region)
            avg_cheek = np.mean(cheek_region)

            # Calculate darkness ratio (Lower means darker under-eye compared to cheek)
            # A difference of > 15-20 units in a grayscale 0-255 range is usually noticeable
            diff = avg_cheek - avg_under_eye
            if diff > max_severity:
                max_severity = diff

    # Determine results based on intensity difference
    if max_severity > 30:
        return {
            "status": "Severe Dark Circles Detected",
            "remedies": [
                "Prioritize 7-9 hours of sleep",
                "Apply cold compresses or ice packs for 15 mins",
                "Elevate your head while sleeping",
                "Stay hydrated and reduce salt intake",
                "Consult a dermatologist if persistent"
            ]
        }
    elif max_severity > 12:
        return {
            "status": "Mild Dark Circles Detected",
            "remedies": [
                "Avoid late-night screen time",
                "Use a moisturizing eye cream",
                "Apply chilled cucumber slices",
                "Ensure consistent sleep patterns"
            ]
        }
    else:
        return {"status": "No significant dark circles detected", "remedies": ["Maintain your current skincare routine!"]}

@app.post("/darkcircle")
async def dark_circle(image: UploadFile = File(...)):
    contents = await image.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv.imdecode(np_img, cv.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image format"}

    result = detect_dark_circles(img)
    return result
