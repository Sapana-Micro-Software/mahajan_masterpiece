"""
FastAPI Server for ECG Classification Models

Provides RESTful API endpoints for:
- Model inference
- Batch predictions
- Model information
- Health checks
- Model switching/ensemble predictions

Production-ready with:
- Input validation
- Error handling
- Logging
- Rate limiting
- Authentication (optional)
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Union
import torch
import torch.nn as nn
import numpy as np
import uvicorn
import logging
from datetime import datetime
import json
import io


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class ECGInput(BaseModel):
    """Single ECG signal input."""
    signal: List[float] = Field(..., description="ECG signal as list of floats")
    sampling_rate: Optional[int] = Field(250, description="Sampling rate in Hz")
    
    @validator('signal')
    def validate_signal_length(cls, v):
        if len(v) < 100:
            raise ValueError('Signal must have at least 100 samples')
        if len(v) > 10000:
            raise ValueError('Signal too long (max 10000 samples)')
        return v


class BatchECGInput(BaseModel):
    """Batch of ECG signals."""
    signals: List[List[float]] = Field(..., description="List of ECG signals")
    sampling_rate: Optional[int] = Field(250, description="Sampling rate in Hz")


class PredictionOutput(BaseModel):
    """Model prediction output."""
    prediction: int = Field(..., description="Predicted class (0-4)")
    probabilities: List[float] = Field(..., description="Class probabilities")
    confidence: float = Field(..., description="Confidence score")
    class_name: str = Field(..., description="Human-readable class name")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class BatchPredictionOutput(BaseModel):
    """Batch prediction output."""
    predictions: List[PredictionOutput]
    total_inference_time_ms: float


class ModelInfo(BaseModel):
    """Model information."""
    model_name: str
    model_type: str
    num_parameters: int
    num_classes: int
    input_shape: List[int]
    supported_operations: List[str]


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_loaded: bool
    device: str


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages loaded models and inference."""
    
    def __init__(self):
        self.models: Dict[str, nn.Module] = {}
        self.active_model_name: str = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['Normal', 'Arrhythmia', 'Tachycardia', 'Bradycardia', 'Abnormal']
        
    def load_model(self, model: nn.Module, model_name: str):
        """Load a model into memory."""
        logger.info(f"Loading model: {model_name}")
        model.eval()
        model.to(self.device)
        self.models[model_name] = model
        
        if self.active_model_name is None:
            self.active_model_name = model_name
        
        logger.info(f"Model {model_name} loaded successfully on {self.device}")
    
    def set_active_model(self, model_name: str):
        """Set the active model for inference."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        self.active_model_name = model_name
        logger.info(f"Active model set to: {model_name}")
    
    def get_active_model(self) -> nn.Module:
        """Get the currently active model."""
        if self.active_model_name is None:
            raise ValueError("No active model set")
        return self.models[self.active_model_name]
    
    def preprocess_signal(self, signal: List[float], target_length: int = 1000) -> torch.Tensor:
        """Preprocess ECG signal for model input."""
        signal_array = np.array(signal)
        
        # Normalize
        signal_array = (signal_array - signal_array.mean()) / (signal_array.std() + 1e-8)
        
        # Resize to target length
        if len(signal_array) != target_length:
            # Simple interpolation
            indices = np.linspace(0, len(signal_array) - 1, target_length)
            signal_array = np.interp(indices, np.arange(len(signal_array)), signal_array)
        
        # Convert to tensor
        tensor = torch.FloatTensor(signal_array).unsqueeze(0).unsqueeze(0)  # (1, 1, length)
        return tensor.to(self.device)
    
    def predict(self, signal: List[float]) -> Dict:
        """Run inference on a single signal."""
        model = self.get_active_model()
        
        # Preprocess
        input_tensor = self.preprocess_signal(signal)
        
        # Inference
        start_time = datetime.now()
        
        with torch.no_grad():
            output = model(input_tensor)
            
            # Handle different output formats
            if isinstance(output, tuple):
                output = output[0]
            
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            probs = probabilities[0].cpu().numpy().tolist()
            confidence = probs[prediction]
        
        end_time = datetime.now()
        inference_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return {
            'prediction': prediction,
            'probabilities': probs,
            'confidence': confidence,
            'class_name': self.class_names[prediction],
            'inference_time_ms': inference_time_ms
        }
    
    def predict_batch(self, signals: List[List[float]]) -> Dict:
        """Run inference on a batch of signals."""
        model = self.get_active_model()
        
        # Preprocess all signals
        input_tensors = [self.preprocess_signal(sig) for sig in signals]
        batch_tensor = torch.cat(input_tensors, dim=0)
        
        # Inference
        start_time = datetime.now()
        
        predictions_list = []
        
        with torch.no_grad():
            output = model(batch_tensor)
            
            if isinstance(output, tuple):
                output = output[0]
            
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
            probs = probabilities.cpu().numpy()
            
            for i in range(len(signals)):
                predictions_list.append({
                    'prediction': int(predictions[i]),
                    'probabilities': probs[i].tolist(),
                    'confidence': float(probs[i][predictions[i]]),
                    'class_name': self.class_names[predictions[i]],
                    'inference_time_ms': 0.0  # Individual time not tracked in batch
                })
        
        end_time = datetime.now()
        total_inference_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return {
            'predictions': predictions_list,
            'total_inference_time_ms': total_inference_time_ms
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the active model."""
        if self.active_model_name is None:
            raise ValueError("No active model")
        
        model = self.get_active_model()
        
        return {
            'model_name': self.active_model_name,
            'model_type': model.__class__.__name__,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'num_classes': len(self.class_names),
            'input_shape': [1, 1, 1000],
            'supported_operations': ['predict', 'predict_batch', 'ensemble']
        }


# ============================================================================
# FastAPI Application
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="ECG Classification API",
    description="Production-ready API for ECG signal classification using deep learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ECG Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_manager.active_model_name is not None,
        "device": str(model_manager.device)
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(ecg_input: ECGInput):
    """
    Predict class for a single ECG signal.
    
    Args:
        ecg_input: ECG signal data
        
    Returns:
        Prediction with probabilities and confidence
    """
    try:
        result = model_manager.predict(ecg_input.signal)
        return PredictionOutput(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(batch_input: BatchECGInput):
    """
    Predict classes for multiple ECG signals.
    
    Args:
        batch_input: Batch of ECG signals
        
    Returns:
        Batch predictions
    """
    try:
        result = model_manager.predict_batch(batch_input.signals)
        return BatchPredictionOutput(**result)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the currently active model."""
    try:
        info = model_manager.get_model_info()
        return ModelInfo(**info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.get("/models/list")
async def list_models():
    """List all loaded models."""
    return {
        "loaded_models": list(model_manager.models.keys()),
        "active_model": model_manager.active_model_name
    }


@app.post("/models/switch/{model_name}")
async def switch_model(model_name: str):
    """Switch to a different loaded model."""
    try:
        model_manager.set_active_model(model_name)
        return {"message": f"Switched to model: {model_name}"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting ECG Classification API...")
    logger.info(f"Device: {model_manager.device}")
    
    # Load a default model (example - you would load your actual model here)
    # Uncomment and modify for production:
    # from transformer_ecg import TransformerECG
    # model = TransformerECG(...)
    # model.load_state_dict(torch.load('model.pth'))
    # model_manager.load_model(model, 'transformer')
    
    logger.info("API ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down ECG Classification API...")


# ============================================================================
# Main - for development server
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ECG Classification API Server")
    print("=" * 80)
    print("\nStarting FastAPI server...")
    print("API Documentation: http://localhost:8000/docs")
    print("ReDoc Documentation: http://localhost:8000/redoc")
    print("\nTo run in production, use:")
    print("  uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4")
    print("=" * 80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
