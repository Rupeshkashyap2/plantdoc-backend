from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load your trained model (same as your test.ipynb does)
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'plant_model.h5'))

# Your 38 class names — copy these from your notebook output
CLASS_NAMES = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
'Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

@app.get("/")
def home():
    return {"message": "PlantDoc API is running!"}

@app.post("/chat")
async def chat(request: ChatRequest):
    import httpx
    headers = {
        "x-api-key": "YOUR_NEW_API_KEY_HERE",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    body = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 500,
        "system": "You are a plant disease expert. Only answer plant related questions. Keep answers short and helpful.",
        "messages": [{"role": "user", "content": request.message}]
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=body
            )
            data = response.json()
            if "content" in data:
                return {"reply": data["content"][0]["text"]}
            else:
                return {"reply": "Error: " + str(data)}
    except Exception as e:
        return {"reply": "Error: " + str(e)}

from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    import httpx
    headers = {
        "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    body = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 500,
        "system": "You are a plant disease expert. Only answer plant related questions. Keep answers short and helpful.",
        "messages": [{"role": "user", "content": request.message}]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=body)
        data = response.json()
        return {"reply": data["content"][0]["text"]}