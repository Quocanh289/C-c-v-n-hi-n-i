from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys
import os

# Add ai_nlp to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ai_nlp'))

from analyzer import analyze_text

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Text Analyzer API",
    description="API for analyzing mental health text to detect emotions, symptoms, and related conditions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class AnalyzeRequest(BaseModel):
    """Request model for text analysis"""
    text: str = Field(..., min_length=10, max_length=5000, description="Text to analyze")

class RiskSignal(BaseModel):
    """Risk signal model"""
    signal: str

class Condition(BaseModel):
    """Mental health condition model"""
    name: str
    name_en: str
    confidence: float
    reason: str
    description: str

class AnalyzeResponse(BaseModel):
    """Response model for text analysis"""
    sentiment: str = Field(description="Overall sentiment: positive, neutral, negative, unknown")
    emotions: list[str] = Field(description="List of detected emotions")
    severity: str = Field(description="Risk severity level: low, medium, high, critical")
    risk_signals: list[str] = Field(description="Detected risk signals")
    possible_related_conditions: list[Condition] = Field(description="Possible mental health conditions")
    recommendation: str = Field(description="Recommendation based on analysis")

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Mental Health Analyzer API"}

# Main analysis endpoint
@app.post(
    "/api/analyze",
    response_model=AnalyzeResponse,
    tags=["Analysis"],
    summary="Analyze text for mental health indicators"
)
async def analyze_text_endpoint(request: AnalyzeRequest):
    """
    Analyze input text for mental health indicators including emotions,
    risk signals, and possible related conditions.
    
    - **text**: The input text to analyze (minimum 10 characters)
    """
    try:
        result = analyze_text(request.text)
        
        # Format conditions
        formatted_conditions = []
        for cond in result['possible_related_conditions']:
            formatted_conditions.append(Condition(
                name=cond['name'],
                name_en=cond['name_en'],
                confidence=cond['confidence'],
                reason=cond['reason'],
                description=cond['description']
            ))
        
        return AnalyzeResponse(
            sentiment=result['sentiment'],
            emotions=result['emotions'],
            severity=result['severity'],
            risk_signals=result['risk_signals'],
            possible_related_conditions=formatted_conditions,
            recommendation=result['recommendation']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

# Sentiment analysis endpoint
@app.post("/api/sentiment", tags=["Analysis"])
async def get_sentiment(request: AnalyzeRequest):
    """Analyze text sentiment only"""
    try:
        result = analyze_text(request.text)
        return {
            "sentiment": result['sentiment'],
            "score": result['debug_info']['sentiment_score'] if 'debug_info' in result else 0.5
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {str(e)}")

# Emotions endpoint
@app.post("/api/emotions", tags=["Analysis"])
async def get_emotions(request: AnalyzeRequest):
    """Extract emotions from text"""
    try:
        result = analyze_text(request.text)
        return {
            "emotions": result['emotions']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emotion detection error: {str(e)}")

# Symptoms endpoint
@app.post("/api/symptoms", tags=["Analysis"])
async def get_symptoms(request: AnalyzeRequest):
    """Extract mental health symptoms from text"""
    try:
        result = analyze_text(request.text)
        return {
            "risk_signals": result['risk_signals'],
            "severity": result['severity']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Symptom extraction error: {str(e)}")

# Error handler
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))

# Root endpoint
@app.get("/", tags=["Info"])
async def root():
    """API information"""
    return {
        "name": "Mental Health Text Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/api/analyze",
            "sentiment": "/api/sentiment",
            "emotions": "/api/emotions",
            "symptoms": "/api/symptoms",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)