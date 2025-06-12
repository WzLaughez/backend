from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np
import io
import uvicorn
import os
import gdown
app = FastAPI()

# CORS biar bisa akses dari React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti ini di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
# Load model
file_id = "1FWvJJdqOZ-BzFK6Yjed9SojibACqJaYz"
model_path = "model/final_model_3.h5"

if not os.path.exists(model_path):
    os.makedirs("model", exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    print("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

model = load_model(model_path)
class_names = ["HDPE", "PET", "PP", "PS"]

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((512, 512))  # Ukuran sesuai input model
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # <- Preprocessing dari EfficientNetB7
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)

    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions[0])]

    return {"prediction": predicted_class}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
