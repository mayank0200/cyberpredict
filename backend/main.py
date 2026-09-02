from datetime import datetime, timezone
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from mangum import Mangum

app = FastAPI(title="CyberPredict API", version="0.1.0", description="Decision-support API using anonymized synthetic demo data.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Vercel serverless handler
handler = Mangum(app)

class CopilotRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    case_id: str = "CP-2026-1042"

class ReportRequest(BaseModel):
    case_id: str = Field(pattern=r"^CP-[0-9]{4}-[0-9]+$")
    format: str = Field(default="brief", pattern="^(brief|pdf)$")

CASES = [
    {"id": "CP-2026-1042", "status": "ACTIVE", "risk_score": 91, "risk_level": "CRITICAL", "city": "Jaipur", "fraud_type": "Digital Payment Fraud", "amount": 120000, "predicted_atm": "ATM-124"},
    {"id": "CP-2026-1039", "status": "ACTIVE", "risk_score": 82, "risk_level": "HIGH", "city": "Lucknow", "fraud_type": "UPI Impersonation", "amount": 84500, "predicted_atm": "ATM-087"},
    {"id": "CP-2026-1031", "status": "ACTIVE", "risk_score": 68, "risk_level": "HIGH", "city": "Delhi", "fraud_type": "Remote Access Fraud", "amount": 210000, "predicted_atm": "ATM-211"},
]

PREDICTIONS = [
    {"id": "PR-2026-124", "case_id": "CP-2026-1042", "atm_id": "ATM-124", "city": "Jaipur South", "probability": 0.91, "risk_score": 91, "window": {"start": "10:55", "end": "11:40"}, "factors": [{"name": "Historical similarity", "weight": 0.31}, {"name": "Linked account activity", "weight": 0.24}, {"name": "Transaction velocity", "weight": 0.18}, {"name": "Geographic proximity", "weight": 0.15}, {"name": "Time-of-day pattern", "weight": 0.12}]}
]

ATMS = [{"id": "ATM-124", "city": "Jaipur South", "lat": 26.9124, "lng": 75.7873, "risk_score": 91, "risk_level": "CRITICAL"}, {"id": "ATM-087", "city": "Lucknow Central", "lat": 26.8467, "lng": 80.9462, "risk_score": 82, "risk_level": "HIGH"}, {"id": "ATM-211", "city": "Delhi East", "lat": 28.6139, "lng": 77.209, "risk_score": 68, "risk_level": "HIGH"}]

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "operational", "data_mode": "synthetic_demo"}

@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    return {"updated_at": datetime.now(timezone.utc).isoformat(), "data_mode": "synthetic_demo", "metrics": {"active_cases": 142, "high_risk_atms": 12, "predicted_withdrawals": 17, "critical_alerts": 6, "suspicious_accounts": 386, "connected_networks": 64}, "pipeline": ["collection", "preprocessing", "feature_engineering", "anomaly_detection", "network_analysis", "location_prediction", "prioritization", "investigator_action"]}

@app.get("/api/cases")
def cases() -> list[dict[str, Any]]:
    return CASES

@app.get("/api/complaints")
def complaints() -> list[dict[str, Any]]:
    return [{"id": case["id"], "received_at": "2026-09-02T10:17:00Z", "fraud_type": case["fraud_type"], "city": case["city"], "data_mode": "synthetic_demo"} for case in CASES]

@app.get("/api/accounts")
def accounts() -> list[dict[str, Any]]:
    return [{"id": "ACCOUNT-A91", "type": "fraud_account", "linked_cases": 17, "risk_score": 89}, {"id": "ACCOUNT-M22", "type": "mule_account", "linked_cases": 8, "risk_score": 84}]

@app.get("/api/transactions")
def transactions() -> list[dict[str, Any]]:
    return [{"id": "TXN-8821", "account_id": "ACCOUNT-M22", "amount": 120000, "city": "Jaipur", "anomaly_score": 0.84}]

@app.get("/api/risk-scores")
def risk_scores() -> list[dict[str, Any]]:
    return [{"entity_id": atm["id"], "score": atm["risk_score"], "level": atm["risk_level"], "requires_verification": True} for atm in ATMS]

@app.get("/api/analytics")
def analytics() -> dict[str, Any]:
    return {"cases_this_month": 1842, "predicted_cashouts": 216, "priority_locations": 38, "networks_surfaced": 64, "accuracy": 0.874, "data_mode": "synthetic_demo"}

@app.get("/api/cases/{case_id}")
def case(case_id: str) -> dict[str, Any]:
    item = next((entry for entry in CASES if entry["id"] == case_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Case not found")
    return {**item, "data_mode": "synthetic_demo", "connected_accounts": 8, "suspicious_transactions": 23, "recommended_action": "Requires investigator verification"}

@app.get("/api/predictions")
def predictions() -> list[dict[str, Any]]:
    return PREDICTIONS

@app.get("/api/predictions/{prediction_id}")
def prediction(prediction_id: str) -> dict[str, Any]:
    item = next((entry for entry in PREDICTIONS if entry["id"] == prediction_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return item

@app.get("/api/atms")
def atms() -> list[dict[str, Any]]:
    return ATMS

@app.get("/api/atms/{atm_id}")
def atm(atm_id: str) -> dict[str, Any]:
    item = next((entry for entry in ATMS if entry["id"] == atm_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="ATM not found")
    return {**item, "confidence": 0.87, "similar_cases": 17, "linked_accounts": 8, "data_mode": "synthetic_demo"}

@app.get("/api/alerts")
def alerts() -> list[dict[str, Any]]:
    return [{"id": "AL-124", "level": "CRITICAL", "reason": "High probability withdrawal predicted", "atm_id": "ATM-124", "case_id": "CP-2026-1042", "confidence": 0.91, "recommended_action": "Immediate field verification"}]

@app.get("/api/ip-intelligence/{case_id}")
def ip_intelligence(case_id: str) -> dict[str, Any]:
    if not any(entry["id"] == case_id for entry in CASES):
        raise HTTPException(status_code=404, detail="Case not found")
    return {"case_id": case_id, "trace_status": "COMPLETED", "source_ip": "203.0.113.42", "approximate_region": "Jaipur, Rajasthan", "coordinates": {"lat": 26.9124, "lng": 75.7873}, "confidence": 0.84, "data_mode": "synthetic_demo", "requires_authorized_verification": True}

@app.get("/api/networks/{case_id}")
def network(case_id: str) -> dict[str, Any]:
    if not any(entry["id"] == case_id for entry in CASES):
        raise HTTPException(status_code=404, detail="Network not found")
    return {"case_id": case_id, "nodes": [{"id": "ACC-A91", "type": "fraud_account"}, {"id": "ACC-M22", "type": "mule_account"}, {"id": "ATM-124", "type": "atm"}], "edges": [["ACC-A91", "ACC-M22"], ["ACC-M22", "ATM-124"]]}

@app.post("/api/copilot")
def copilot(request: CopilotRequest) -> dict[str, Any]:
    return {"answer": "ATM-124 is ranked high because multiple indicators match previously observed fraud cases.", "confidence": 0.91, "evidence": ["17 similar historical cases", "8 linked accounts", "High transaction velocity", "Geographic proximity"], "case_id": request.case_id, "data_mode": "synthetic_demo"}

@app.post("/api/reports")
def report(request: ReportRequest) -> dict[str, Any]:
    if not any(entry["id"] == request.case_id for entry in CASES):
        raise HTTPException(status_code=404, detail="Case not found")
    return {"status": "generated", "case_id": request.case_id, "format": request.format, "created_at": datetime.now(timezone.utc).isoformat(), "classification": "AUTHORIZED"}
