from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'plant_model.h5'))

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

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "PlantDoc API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).resize((128, 128))
    input_arr = np.array([np.array(image)])
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)
    confidence = float(np.max(predictions))
    return {
        "disease": CLASS_NAMES[result_index],
        "confidence": round(confidence * 100, 2)
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    import httpx
    headers = {
        "Authorization": f"Bearer {os.environ.get('HUGGINGFACE_API_KEY', '')}",
        "Content-Type": "application/json"
    }
    body = {
        "inputs": f"You are a plant disease expert. Answer only plant related questions briefly.\n\nQuestion: {request.message}\n\nAnswer:",
        "parameters": {"max_new_tokens": 200, "temperature": 0.7}
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
                headers=headers,
                json=body
            )
            data = response.json()
            if isinstance(data, list):
                reply = data[0]["generated_text"].split("Answer:")[-1].strip()
                return {"reply": reply}
            else:
                return {"reply": "Error: " + str(data)}
    except Exception as e:
        return {"reply": "Error: " + str(e)}