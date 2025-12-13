# app/schemas.py
from typing import List
from pydantic import BaseModel, Field


class CreditScoreRequest(BaseModel):
    borrower_id: str = Field(..., description="External borrower ID")

    monthly_income: float
    monthly_expense: float
    age: int
    has_smartphone: bool
    has_wallet: bool
    avg_wallet_balance: float
    on_time_payment_ratio: float
    num_loans_taken: int


class Factor(BaseModel):
    feature: str
    impact: float  # SHAP value
    direction: str  # "increased_risk" or "decreased_risk"
    detail: str     # human readable text


class CreditScoreResponse(BaseModel):
    borrower_id: str
    risk_score: float  # 0-1 probability of default
    decision: str
    explanation: str
    top_factors: List[Factor]
