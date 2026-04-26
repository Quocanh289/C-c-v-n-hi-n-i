from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ai_nlp'))
from analyzer import analyze_text

app = FastAPI(title="Mental Health Text Analyzer API")

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    sentiment: str
    emotions: list[str]
    severity: str
    risk_signals: list[str]
    possible_related_conditions: list[dict]
    recommendation: str

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_text_endpoint(request: AnalyzeRequest):
    result = analyze_text(request.text)
    return result