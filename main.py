from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2 as cv
import numpy as np

app = FastAPI(title="Healthcare AI - Dark Circle Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development. In production, use ["https://your-site.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load cascade files
eyes_area = cv.CascadeClassifier("haarcascade_eye.xml")
face_area = cv.CascadeClassifier("haarcascade_frontalface_default.xml")



if eyes_area.empty() or face_area.empty():
    print("Error: Cascade files not found! Check your file paths.")


def detect_dark_circles(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_area.detectMultiScale(gray, 1.3, 5)

    dark_circle_detected = False

    for (x, y, w, h) in faces:
        gray_roi = gray[y:y+h, x:x+w]
        eyes = eyes_area.detectMultiScale(gray_roi)

        for (ex, ey, ew, eh) in eyes:
            eye_region = gray_roi[ey:ey+eh, ex:ex+ew]

            avg_intensity = eye_region.mean()

            if avg_intensity < 40:
                dark_circle_detected = True

    if dark_circle_detected:
        return "Dark circle detected"
    else:
        return "No dark circles detected"


@app.post("/darkcircle")
async def dark_circle(image: UploadFile = File(...)):
    
    contents = await image.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv.imdecode(np_img, cv.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    result = detect_dark_circles(img)

    return {"result": result}
