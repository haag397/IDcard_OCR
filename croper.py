from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import io
import base64

# Load YOLO model
model = YOLO("best.pt")

# Initialize FastAPI
app = FastAPI()

def read_imagefile(file) -> np.ndarray:
    """Convert uploaded file to OpenCV image format."""
    image = Image.open(io.BytesIO(file))
    return np.array(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = read_imagefile(image_bytes)

    # Run YOLO model on image
    results = model(image)

    # Extract bounding boxes and crop detected areas
    cropped_images = []
    detected_boxes = []
    
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)  # Convert to integers
            cropped = image[y1:y2, x1:x2]   # Crop region
            cropped_images.append(cropped)
            detected_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

            # Draw bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Convert processed image to base64 for easy return
    _, encoded_image = cv2.imencode(".jpg", image)
    processed_image_base64 = base64.b64encode(encoded_image).decode()

    # Convert cropped images to base64
    cropped_images_base64 = [
        base64.b64encode(cv2.imencode(".jpg", img)[1]).decode() for img in cropped_images
    ]

    return {
        "processed_image": processed_image_base64,
        "detected_boxes": detected_boxes,
        "cropped_images": cropped_images_base64
    }

