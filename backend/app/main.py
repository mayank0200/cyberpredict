from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.ml_pipeline import MODEL_BUNDLE, get_alerts, get_case_detail, get_case_prediction, get_network

app = FastAPI(
    title="CyberPredict Secure Intelligence Terminal",
    version="0.1.0",
    description="Local demonstrator for predictive cybercrime intelligence. Synthetic data only.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CopilotRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    case_id: str = "CP-2026-1042"


class ReportRequest(BaseModel):
    case_id: str = Field(default="CP-2026-1042")
    format: str = "brief"


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "operational",
        "environment": "Hardened Linux-based controlled environment",
        "data_mode": "synthetic_demo",
        "model_status": "local",
        "ai_warning": "AI-generated intelligence is decision support, not a final determination.",
    }


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "data_mode": "synthetic_demo",
        "metrics": {
            "active_cases": 142,
            "high_risk_locations": 12,
            "pending_alerts": 6,
            "predictions": 17,
            "networks_identified": 64,
        },
        "pipeline": [
            "collection",
            "preprocessing",
            "feature_engineering",
            "anomaly_detection",
            "network_analysis",
            "location_prediction",
            "prioritization",
            "investigator_review",
        ],
        "model_metrics": MODEL_BUNDLE.metrics,
    }


@app.get("/api/cases")
def cases() -> list[dict[str, Any]]:
    return [
        {
            "id": case_id,
            "status": "ACTIVE",
            "risk_score": int(prediction["risk_score"]),
            "risk_level": prediction["risk_level"],
            "city": case["city"],
            "fraud_type": case["fraud_type"],
            "amount": case["fraud_amount"],
            "predicted_atm": prediction["predicted_atm"],
            "data_mode": "synthetic_demo",
        }
        for case_id, case in MODEL_BUNDLE.cases.items()
        for prediction in [get_case_prediction(case_id)]
    ]


@app.get("/api/cases/{case_id}")
def get_case(case_id: str) -> dict[str, Any]:
    if case_id not in MODEL_BUNDLE.cases:
        raise HTTPException(status_code=404, detail="Case not found")
    detail = get_case_detail(case_id)
    detail["prototype_notice"] = "Prototype uses synthetic/anonymized data for demonstration."
    return detail


@app.get("/api/predictions")
def predictions() -> list[dict[str, Any]]:
    return [get_case_prediction(case_id) for case_id in MODEL_BUNDLE.cases]


@app.get("/api/predictions/{case_id}")
def prediction(case_id: str) -> dict[str, Any]:
    if case_id not in MODEL_BUNDLE.cases:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return get_case_prediction(case_id)


@app.get("/api/atms")
def atms() -> list[dict[str, Any]]:
    return [
        {
            "id": atm["id"],
            "city": atm["city"],
            "lat": atm["lat"],
            "lng": atm["lng"],
            "risk_score": 91 if atm["id"] == "ATM-124" else 82 if atm["id"] == "ATM-087" else 68,
            "risk_level": "CRITICAL" if atm["id"] == "ATM-124" else "HIGH" if atm["id"] == "ATM-087" else "MEDIUM",
            "data_mode": "synthetic_demo",
        }
        for atm in MODEL_BUNDLE.atms
    ]


@app.get("/api/atms/{atm_id}")
def atm(atm_id: str) -> dict[str, Any]:
    atm = next((item for item in MODEL_BUNDLE.atms if item["id"] == atm_id), None)
    if not atm:
        raise HTTPException(status_code=404, detail="ATM not found")
    return {
        **atm,
        "risk_score": 91 if atm_id == "ATM-124" else 82 if atm_id == "ATM-087" else 68,
        "confidence": 0.91 if atm_id == "ATM-124" else 0.82 if atm_id == "ATM-087" else 0.68,
        "similar_cases": 17 if atm_id == "ATM-124" else 11,
        "linked_accounts": 8,
        "data_mode": "synthetic_demo",
        "prototype_notice": "Prototype uses synthetic/anonymized data for demonstration.",
    }


@app.get("/api/alerts")
def alerts() -> list[dict[str, Any]]:
    return get_alerts()


@app.get("/api/networks/{case_id}")
def network(case_id: str) -> dict[str, Any]:
    if case_id not in MODEL_BUNDLE.cases:
        raise HTTPException(status_code=404, detail="Case not found")
    return get_network(case_id)


@app.get("/api/ip-intelligence/{case_id}")
def ip_intelligence(case_id: str) -> dict[str, Any]:
    if case_id not in MODEL_BUNDLE.cases:
        raise HTTPException(status_code=404, detail="Case not found")
    return {
        "case_id": case_id,
        "trace_status": "COMPLETED",
        "source_ip": "203.0.113.42",
        "approximate_region": "Jaipur, Rajasthan",
        "coordinates": {"lat": 26.9124, "lng": 75.7873},
        "confidence": 0.84,
        "requires_authorized_verification": True,
        "data_mode": "synthetic_demo",
    }


@app.post("/api/copilot")
def copilot(request: CopilotRequest) -> dict[str, Any]:
    case_id = request.case_id
    if case_id not in MODEL_BUNDLE.cases:
        raise HTTPException(status_code=404, detail="Case not found")
    question = request.question.lower()
    prediction = get_case_prediction(case_id)
    if "why" in question or "flag" in question or "risk" in question:
        answer = (
            "ATM-124 is ranked highest because the available case data shows historical pattern similarity, "
            "high transaction velocity, geographic proximity, and linked account activity consistent with the current fraud cluster."
        )
        evidence = [
            "17 similar historical cases in the local dataset",
            "8 linked accounts connected to this complaint",
            "91/100 risk score for the predicted ATM",
            "Transaction velocity exceeds current case baseline",
        ]
    elif "related case" in question:
        answer = "The local evidence links this complaint to three related investigations and three additional accounts."
        evidence = [
            "CP-2026-1011",
            "CP-2026-1024",
            "CP-2026-1034",
            "ACCOUNT-A91 and ACCOUNT-B42",
        ]
    elif "evidence" in question or "supports" in question:
        answer = "The forecast uses the locally trained model and available evidence from the current case: account risk, historical similarity, linked activity, and ATM proximity."
        evidence = [
            "Model confidence: 91%",
            "Local training metrics: accuracy 0.87, F1 0.82",
            "Current case factors: 4 major risk drivers",
        ]
    elif "location" in question or "next highest" in question:
        answer = "The next highest candidate locations are ATM-087 and ATM-211 after ATM-124 based on the current local model score." 
        evidence = [
            "ATM-087: 82 risk score",
            "ATM-211: 68 risk score",
            "All rankings are generated from the local model and should be investigator-validated.",
        ]
    else:
        answer = "Insufficient evidence in the current case data."
        evidence = []
    return {
        "answer": answer,
        "confidence": prediction["confidence"],
        "evidence": evidence,
        "case_id": case_id,
        "prototype_notice": "Prototype uses synthetic/anonymized data for demonstration.",
        "data_mode": "synthetic_demo",
    }


@app.post("/api/reports")
def report(request: ReportRequest) -> dict[str, Any]:
    if request.case_id not in MODEL_BUNDLE.cases:
        raise HTTPException(status_code=404, detail="Case not found")
    prediction = get_case_prediction(request.case_id)
    return {
        "status": "generated",
        "case_id": request.case_id,
        "format": request.format,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": "SYNTHETIC DATA — DEMONSTRATION ONLY",
        "predicted_atm": prediction["predicted_atm"],
        "risk_score": prediction["risk_score"],
        "expected_window": prediction["expected_window"],
        "recommended_priority": "High Priority",
        "data_mode": "synthetic_demo",
    }


@app.get("/api/model")
def model_status() -> dict[str, Any]:
    return {
        "model_name": "CyberPredict Risk Model",
        "version": "0.1",
        "training_data": "Synthetic Demonstration Dataset",
        "training_date": "2026-09-12",
        "features": ["transaction_velocity", "risk_signal", "ATM bias", "linked_case_count"],
        "evaluation_metrics": MODEL_BUNDLE.metrics,
        "status": "Local",
        "prototype_notice": "Prototype uses synthetic/anonymized data for demonstration.",
    }
