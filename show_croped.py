from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import io
import os

# Create output directory if not exists
os.makedirs("output", exist_ok=True)

# Load YOLO model
model = YOLO("best.pt")

# Initialize FastAPI
app = FastAPI()

def read_imagefile(file) -> np.ndarray:
    """Convert uploaded file to OpenCV image format."""
    image = Image.open(io.BytesIO(file))
    return np.array(image)

@app.post("/show_predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = read_imagefile(image_bytes)
    
    # Convert to OpenCV format (BGR)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run YOLO model on image
    results = model(image)

    # Extract bounding boxes and crop detected areas
    cropped_images = []
    detected_boxes = []
    
    for idx, result in enumerate(results):
        for i, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)  # Convert to integers
            cropped = image_bgr[y1:y2, x1:x2]   # Crop region
            cropped_filename = f"output/cropped_{i}.jpg"
            cv2.imwrite(cropped_filename, cropped)  # Save cropped image
            cropped_images.append(cropped_filename)
            detected_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

            # Draw bounding box on the image
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save processed image with bounding boxes
    output_filename = "output/output.jpg"
    cv2.imwrite(output_filename, image_bgr)

    return {
        "processed_image": output_filename,
        "detected_boxes": detected_boxes,
        "cropped_images": cropped_images
    }

@app.get("/download/{filename}")
async def download_image(filename: str):
    """Download processed image or cropped image."""
    file_path = f"output/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg", filename=filename)
    return {"error": "File not found"}
