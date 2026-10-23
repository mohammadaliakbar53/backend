"""
Sindh Smart Farmer AI - Python Deep Learning REST API
Serves real-time multilingual inferences using the trained PyTorch SindhAgriNet model.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from inference import SindhAgriInferenceEngine

app = FastAPI(
    title="Sindh Smart Farmer AI - PyTorch Backend",
    description="Agronomic intelligence and multilingual inference powered by PyTorch deep learning",
    version="1.0.0"
)

# Enable CORS for local integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SindhAgriInferenceEngine()

class PredictRequest(BaseModel):
    cropName: Optional[str] = "Cotton"
    location: Optional[str] = "Sindh"
    symptoms: str
    language: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = None
    conversationHistory: Optional[List[Dict[str, Any]]] = None

@app.get("/")
def root():
    return {
        "service": "Sindh Smart Farmer AI - PyTorch Backend",
        "status": "online",
        "model_loaded": engine.is_loaded,
        "device": str(engine.device)
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": engine.is_loaded,
        "framework": "PyTorch",
        "device": str(engine.device),
        "parameters": engine.meta.get("parameters", 0)
    }

@app.get("/api/model/metrics")
def get_metrics():
    if not engine.meta:
        return {
            "model_name": "SindhAgriNet-PyTorch",
            "status": "initialized",
            "parameters": 428000
        }
    return engine.meta

@app.post("/api/model/predict")
def predict_crop(req: PredictRequest):
    try:
        result = engine.generate_crop_analysis(
            crop_name=req.cropName,
            location=req.location,
            symptoms=req.symptoms,
            target_lang=req.language
        )
        return {
            "success": True,
            "analysis": result,
            "engine": "PyTorch-SindhAgriNet"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/chat")
def chat(req: ChatRequest):
    try:
        reply = engine.generate_chat_reply(
            message=req.message,
            language_pref=req.language
        )
        detected = engine.detect_language(req.message, req.language)
        return {
            "success": True,
            "reply": reply,
            "language": detected,
            "engine": "PyTorch-SindhAgriNet"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5001, reload=False)
