from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import mediapipe as mp
from tensorflow.keras.applications.efficientnet import preprocess_input


BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [BASE_DIR / "best_model.keras", BASE_DIR / "stress_model.keras"]
CLASS_NAMES = ["nostress", "stress"]


def resolve_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No trained model found. Expected best_model2.keras or stress_model2.keras."
    )


MODEL_PATH = resolve_model_path()
MODEL = tf.keras.models.load_model(MODEL_PATH)

INPUT_SHAPE = MODEL.input_shape
IMAGE_SIZE = int(INPUT_SHAPE[1]) if INPUT_SHAPE and INPUT_SHAPE[1] else 224
FACE_DETECTOR = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5,
)


app = FastAPI(
    title="Stress Detection Inference API",
    description="FastAPI wrapper for the trained face stress detector.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def prepare_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(image_bytes))
    except Exception as exc:  # pragma: no cover - user input dependent
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    image = ImageOps.exif_transpose(image)

    rgb_image = image.convert("RGB")
    image_array = np.asarray(rgb_image)
    detection_result = FACE_DETECTOR.process(image_array)

    if not detection_result.detections:
        raise HTTPException(
            status_code=400,
            detail="No face detected. Please upload a close-up face image.",
        )

    height, width = image_array.shape[:2]
    detection = max(
        detection_result.detections,
        key=lambda item: item.location_data.relative_bounding_box.width
        * item.location_data.relative_bounding_box.height,
    )
    box = detection.location_data.relative_bounding_box

    center_x = (box.xmin + box.width / 2.0) * width
    center_y = (box.ymin + box.height / 2.0) * height
    box_size = max(box.width * width, box.height * height) * 1.0
    half_size = box_size / 2.0

    left = max(0, int(center_x - half_size))
    top = max(0, int(center_y - half_size))
    right = min(width, int(center_x + half_size))
    bottom = min(height, int(center_y + half_size))

    if right <= left or bottom <= top:
        raise HTTPException(
            status_code=400,
            detail="Unable to crop a face region from the uploaded image.",
        )

    face_image = rgb_image.crop((left, top, right, bottom))
    face_image_bw = ImageOps.grayscale(face_image)
    face_image_bw = face_image_bw.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Convert to RGB for model input (keep as grayscale data)
    face_image_rgb = face_image_bw.convert("RGB")

    array = np.asarray(face_image_rgb, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    return array


def predict_image(image_bytes: bytes) -> dict:
    batch = prepare_image(image_bytes)
    probabilities = MODEL.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_index]

    return {
        "label": predicted_label,
        "class_index": predicted_index,
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            CLASS_NAMES[index]: float(probability)
            for index, probability in enumerate(probabilities)
        },
        "image_size": IMAGE_SIZE,
        "model_path": MODEL_PATH.name,
    }


@app.get("/")
def root() -> dict:
    return {
        "message": "Stress Detection API is running.",
        "model": MODEL_PATH.name,
        "image_size": IMAGE_SIZE,
        "classes": CLASS_NAMES,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    return predict_image(image_bytes)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)