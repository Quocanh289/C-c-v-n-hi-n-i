# ====================================================
# Slang Detection & Monitoring Routes
# Automatically detects emerging slang terms and 
# monitors frequency spikes for continuous adaptation
# ====================================================

import os
import re
import json
import hashlib
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import List, Optional, Dict, Set
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/slang", tags=["slang"])


# ====================================================
# Pydantic Models
# ====================================================

class SlangTerm(BaseModel):
    """A detected slang term with metadata."""
    term: str
    language: str = "mixed"  # "vi" | "en" | "mixed"
    possible_emotions: List[str] = []
    frequency: int = 0
    first_seen: str = ""
    last_seen: str = ""
    examples: List[str] = []
    is_emerging: bool = False
    is_verified: bool = False
    verified_emotion: Optional[str] = None


class SlangReport(BaseModel):
    """Report for slang monitoring dashboard."""
    total_terms: int
    emerging_terms: List[SlangTerm]
    verified_terms: List[SlangTerm]
    top_unusual_words: List[dict]
    monitoring_window_hours: int
    generated_at: str


class SlangDetectRequest(BaseModel):
    """Request to detect slang in text."""
    text: str = Field(..., description="Text to check for slang")
    known_slang: List[str] = Field(default_factory=list, description="Known slang terms to skip")


class SlangDetectResponse(BaseModel):
    """Response with detected slang."""
    detected_slang: List[SlangTerm]
    unknown_terms: List[dict]
    has_emerging_slang: bool


# ====================================================
# Slang Detection Service
# ====================================================

# Known Vietnamese internet slang
VIETNAMESE_SLANG: Dict[str, list] = {
    "xỉu": ["surprise", "joy"],
    "đỉnh nóc": ["joy", "surprise"],
    "kịch trần": ["joy", "surprise"],
    "gắt": ["anger", "sarcastic"],
    "cà khịa": ["sarcastic"],
    "thả thính": ["joy"],
    "ngáo": ["surprise", "sarcastic"],
    "bất lực": ["sadness", "anxiety"],
    "quạo": ["anger"],
    "hãm": ["toxic", "anger"],
    "toxic": ["toxic", "anger"],
    "sao cũng được": ["neutral"],
    "chán": ["sadness", "anxiety"],
    "mệt mỏi": ["sadness"],
    "áp lực": ["anxiety"],
    "cringe": ["sarcastic"],
    "cook": ["sarcastic"],
    "cooked": ["sarcastic"],
    "vibe": ["joy", "neutral"],
    "slay": ["joy"],
    "npc": ["sarcastic", "neutral"],
    "delulu": ["sarcastic"],
    "sus": ["anxiety", "sarcastic"],
    "based": ["joy"],
    "no cap": ["joy"],
    "cap": ["sarcastic"],
    "ghosted": ["sadness"],
    "salty": ["anger"],
    "mid": ["neutral"],
    "fire": ["joy"],
    "bet": ["joy"],
    "simp": ["sarcastic"],
    "flex": ["joy"],
    "main character": ["joy", "sarcastic"],
    "red flag": ["sarcastic"],
    "green flag": ["joy"],
    "ick": ["sarcastic"],
}

# Known English slang
ENGLISH_SLANG: Dict[str, list] = {
    "cooked": ["sarcastic"],
    "delulu": ["sarcastic"],
    "slay": ["joy"],
    "based": ["joy"],
    "cringe": ["sarcastic"],
    "ghosted": ["sadness"],
    "salty": ["anger"],
    "savage": ["sarcastic"],
    "lit": ["joy"],
    "fire": ["joy"],
    "simp": ["sarcastic"],
    "flex": ["joy"],
    "no cap": ["joy"],
    "cap": ["sarcastic"],
    "sus": ["anxiety"],
    "bet": ["joy"],
    "bussin": ["joy"],
    "mid": ["neutral"],
    "npc": ["neutral"],
    "main character": ["joy"],
    "side quest": ["sarcastic"],
    "ick": ["sarcastic"],
    "red flag": ["sarcastic"],
    "green flag": ["joy"],
}

# Storage for monitoring
MONITORING_DIR = os.getenv("MONITORING_DIR", "data/monitoring")
os.makedirs(MONITORING_DIR, exist_ok=True)

# In-memory frequency tracking
term_frequencies: Dict[str, Dict] = {}
unknown_term_buffer: List[dict] = []


def _normalize_text(text: str) -> str:
    """Normalize text for slang detection."""
    text = text.lower().strip()
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags symbols but keep text
    text = text.replace('#', '')
    return text


def _tokenize(text: str) -> List[str]:
    """Tokenize text into words and phrases."""
    text = _normalize_text(text)
    words = text.split()
    
    # Also consider bi-grams for multi-word slang
    tokens = list(words)
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]} {words[i+1]}")
    
    return tokens


def _generate_term_id(term: str) -> str:
    """Generate unique ID for a term."""
    return hashlib.md5(term.encode()).hexdigest()[:12]


def _save_monitoring_data():
    """Persist monitoring data to disk."""
    try:
        data = {
            "term_frequencies": term_frequencies,
            "unknown_terms": unknown_term_buffer[-1000:],  # Keep last 1000
            "updated_at": datetime.utcnow().isoformat(),
        }
        filepath = os.path.join(MONITORING_DIR, "slang_monitoring.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save monitoring data: {e}")


def _load_monitoring_data():
    """Load monitoring data from disk."""
    global term_frequencies, unknown_term_buffer
    
    filepath = os.path.join(MONITORING_DIR, "slang_monitoring.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            term_frequencies = data.get("term_frequencies", {})
            unknown_term_buffer = data.get("unknown_terms", [])
        except Exception as e:
            logger.error(f"Failed to load monitoring data: {e}")


# Load existing data
_load_monitoring_data()


def detect_slang_in_text(
    text: str,
    known_slang: Optional[List[str]] = None,
) -> SlangDetectResponse:
    """
    Detect slang terms in text and identify potentially emerging slang.
    """
    tokens = _tokenize(text)
    known_set = set(w.lower() for w in (known_slang or []))
    
    detected = []
    unknown_terms = []
    has_emerging = False
    
    now = datetime.utcnow().isoformat()
    
    # Check for known slang
    combined_slang = {**VIETNAMESE_SLANG, **ENGLISH_SLANG}
    
    for token in tokens:
        if token in combined_slang:
            term = SlangTerm(
                term=token,
                language="vi" if token in VIETNAMESE_SLANG else "en",
                possible_emotions=combined_slang[token],
                first_seen=now,
                last_seen=now,
                is_verified=True,
                verified_emotion=combined_slang[token][0] if combined_slang[token] else None,
            )
            detected.append(term)
            
            # Track frequency
            if token not in term_frequencies:
                term_frequencies[token] = {
                    "count": 0,
                    "first_seen": now,
                    "emotions": combined_slang[token],
                }
            term_frequencies[token]["count"] += 1
            term_frequencies[token]["last_seen"] = now
        
        elif token not in known_set and len(token) > 2 and re.match(r'^[a-zA-Zà-ỹ]+$', token):
            # Check for unknown terms (potential emerging slang)
            if token not in combined_slang and token not in known_set:
                unknown_terms.append({
                    "term": token,
                    "context": text[:100],
                    "timestamp": now,
                })
                unknown_term_buffer.append({
                    "term": token,
                    "context": text[:100],
                    "timestamp": now,
                })
                
                # Track unknown term frequency
                if token not in term_frequencies:
                    term_frequencies[token] = {
                        "count": 0,
                        "first_seen": now,
                        "emotions": [],
                    }
                term_frequencies[token]["count"] += 1
                term_frequencies[token]["last_seen"] = now
    
    # Detect emerging slang (new terms appearing frequently)
    emerging = []
    for term, info in term_frequencies.items():
        if (info["count"] >= 3 and  # Appeared at least 3 times
            term not in combined_slang):  # Not already known
            
            first = datetime.fromisoformat(info["first_seen"])
            if datetime.utcnow() - first < timedelta(hours=24):  # Within last 24h
                emerging.append(SlangTerm(
                    term=term,
                    language="mixed",
                    possible_emotions=[],
                    frequency=info["count"],
                    first_seen=info["first_seen"],
                    last_seen=info["last_seen"],
                    is_emerging=True,
                    is_verified=False,
                ))
                has_emerging = True
    
    # Persist periodically
    _save_monitoring_data()
    
    return SlangDetectResponse(
        detected_slang=detected,
        unknown_terms=unknown_terms[:20],
        has_emerging_slang=has_emerging,
    )


# ====================================================
# Routes
# ====================================================

@router.post("/detect", response_model=SlangDetectResponse)
async def detect_slang(request: SlangDetectRequest):
    """
    Detect slang terms in text.
    
    Analyzes text for:
    - Known Vietnamese/English internet slang
    - Potentially emerging unknown terms
    - Frequency tracking for new terms
    
    This is the primary API used by the extension's
    continuous learning pipeline.
    """
    try:
        result = detect_slang_in_text(
            text=request.text,
            known_slang=request.known_slang,
        )
        return result
    except Exception as e:
        logger.error(f"Slang detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report", response_model=SlangReport)
async def get_slang_report(
    hours: int = 24,
    min_frequency: int = 3,
):
    """
    Generate slang monitoring report.
    
    Reports on:
    - All tracked terms with frequencies
    - Emerging slang (new, trending terms)
    - Verified slang terms
    - Top unknown/unusual words
    """
    try:
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=hours)
        
        emerging_terms = []
        verified_terms = []
        top_unusual = []
        
        combined_slang = {**VIETNAMESE_SLANG, **ENGLISH_SLANG}
        
        for term, info in term_frequencies.items():
            first_seen = datetime.fromisoformat(info["first_seen"])
            
            term_obj = SlangTerm(
                term=term,
                language="vi" if term in VIETNAMESE_SLANG else "en",
                possible_emotions=info.get("emotions", []),
                frequency=info["count"],
                first_seen=info["first_seen"],
                last_seen=info["last_seen"],
                is_verified=term in combined_slang,
            )
            
            if term in combined_slang:
                term_obj.verified_emotion = combined_slang[term][0] if combined_slang[term] else None
                verified_terms.append(term_obj)
            elif first_seen > cutoff and info["count"] >= min_frequency:
                term_obj.is_emerging = True
                emerging_terms.append(term_obj)
            
            # Track for top unusual
            if info["count"] >= min_frequency and term not in combined_slang:
                top_unusual.append({
                    "term": term,
                    "frequency": info["count"],
                    "times_seen": info["count"],
                })
        
        # Sort by frequency descending
        emerging_terms.sort(key=lambda x: x.frequency, reverse=True)
        verified_terms.sort(key=lambda x: x.frequency, reverse=True)
        top_unusual.sort(key=lambda x: x["frequency"], reverse=True)
        
        return SlangReport(
            total_terms=len(term_frequencies),
            emerging_terms=emerging_terms[:50],
            verified_terms=verified_terms[:100],
            top_unusual_words=top_unusual[:20],
            monitoring_window_hours=hours,
            generated_at=now.isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify")
async def verify_slang_term(
    term: str,
    emotion: str,
    language: str = "mixed",
):
    """
    Manually verify a potential slang term.
    
    Once verified, the term will be:
    1. Added to the known slang dictionary
    2. Used in future emotion predictions
    3. Tracked in the monitoring system
    """
    try:
        term_lower = term.lower().strip()
        
        # Update our knowledge base
        if language == "vi":
            VIETNAMESE_SLANG[term_lower] = [emotion]
        else:
            ENGLISH_SLANG[term_lower] = [emotion]
        
        # Update tracking
        if term_lower in term_frequencies:
            term_frequencies[term_lower]["verified"] = True
            term_frequencies[term_lower]["verified_emotion"] = emotion
            term_frequencies[term_lower]["emotions"] = [emotion]
        
        _save_monitoring_data()
        
        # Save to permanent slang database
        slang_db_path = os.path.join(MONITORING_DIR, "verified_slang.json")
        verified = {}
        if os.path.exists(slang_db_path):
            with open(slang_db_path, "r", encoding="utf-8") as f:
                verified = json.load(f)
        
        verified[term_lower] = {
            "emotion": emotion,
            "language": language,
            "verified_at": datetime.utcnow().isoformat(),
            "frequency": term_frequencies.get(term_lower, {}).get("count", 0),
        }
        
        with open(slang_db_path, "w", encoding="utf-8") as f:
            json.dump(verified, f, ensure_ascii=False, indent=2)
        
        return {
            "status": "verified",
            "term": term_lower,
            "emotion": emotion,
            "language": language,
            "message": f"'{term}' has been verified as slang expressing {emotion}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))