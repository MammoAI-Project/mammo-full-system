from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
import tensorflow as tf
from tensorflow.keras.models import load_model
from .schema import AnalysisResponse
from .process import process_image

app = FastAPI(
    title="Breast Cancer Analysis API",
    description="API for breast cancer detection and risk assessment from mammogram images",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BIRADS_DESCRIPTIONS = {
    'CL1': 'Normal mammogram, no significant abnormalities',
    'CL3': 'Probably benign finding, short-term follow-up suggested',
    'CL4': 'Suspicious abnormality, biopsy should be considered',
    'CL5': 'Highly suspicious of malignancy, appropriate action should be taken'
}

# Global model variable
model = None

def load_ai_model():
    global model
    try:
        model = load_model("model/best_breast_cancer_model.h5")
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        model = "mock_model"
        return False

@app.on_event("startup")
async def startup_event():
    load_ai_model()


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...), threshold: float = 0.7):
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a JPG, JPEG, or PNG image.")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")
    
    try:
        contents = await file.read()
        return process_image(model, contents, threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/birads-descriptions")
async def get_birads_descriptions():
    return BIRADS_DESCRIPTIONS

@app.get("/")
async def root():
    return {"message": "Welcome to the Breast Cancer Analysis API", "status": "active"}

# used in development to run the server locally
# Uncomment the following lines to run the server locally
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)



