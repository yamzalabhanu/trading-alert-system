# models.py
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime

class Alert(BaseModel):
    side: str
    symbol: str
    underlying_price_from_alert: float
    strike: float
    expiry: str
    expiry_source: str

class Decision(BaseModel):
    final: str
    path: str
    score: Optional[float]
    rating: Optional[str]

class LLMResponse(BaseModel):
    ran: bool
    reason: str
    decision: str
    confidence: float
    checklist: Dict[str, Any]
    ev_estimate: Dict[str, Any]

class WebhookResponse(BaseModel):
    ok: bool
    parsed_alert: Alert
    option_ticker: str
    features: Dict[str, Any]
    prescore: Optional[float]
    recommendation: Dict[str, Any]
    decision: Decision
    llm: LLMResponse
    cooldown: Dict[str, Any]
    quota: Dict[str, Any]
    telegram: Dict[str, Any]
    notes: str