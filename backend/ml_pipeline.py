from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)

ATM_CATALOG = [
    {"id": "ATM-124", "city": "Jaipur", "lat": 26.9124, "lng": 75.7873, "risk_bias": 1.4},
    {"id": "ATM-087", "city": "Lucknow", "lat": 26.8467, "lng": 80.9462, "risk_bias": 1.1},
    {"id": "ATM-211", "city": "Delhi", "lat": 28.6139, "lng": 77.209, "risk_bias": 1.0},
    {"id": "ATM-056", "city": "Mumbai", "lat": 19.076, "lng": 72.8777, "risk_bias": 0.82},
    {"id": "ATM-098", "city": "Hyderabad", "lat": 17.385, "lng": 78.4867, "risk_bias": 0.75},
    {"id": "ATM-143", "city": "Bengaluru", "lat": 12.9716, "lng": 77.5946, "risk_bias": 0.65},
    {"id": "ATM-177", "city": "Kolkata", "lat": 22.5726, "lng": 88.3639, "risk_bias": 0.78},
    {"id": "ATM-302", "city": "Ahmedabad", "lat": 23.0225, "lng": 72.5714, "risk_bias": 0.83},
    {"id": "ATM-318", "city": "Pune", "lat": 18.5204, "lng": 73.8567, "risk_bias": 0.82},
    {"id": "ATM-329", "city": "Chennai", "lat": 13.0827, "lng": 80.2707, "risk_bias": 0.7},
    {"id": "ATM-341", "city": "Kochi", "lat": 9.9312, "lng": 76.2673, "risk_bias": 0.61},
    {"id": "ATM-356", "city": "Patna", "lat": 25.5941, "lng": 85.1376, "risk_bias": 0.8},
    {"id": "ATM-367", "city": "Bhopal", "lat": 23.2599, "lng": 77.4126, "risk_bias": 0.71},
    {"id": "ATM-378", "city": "Indore", "lat": 22.7196, "lng": 75.8577, "risk_bias": 0.84},
    {"id": "ATM-389", "city": "Surat", "lat": 21.1702, "lng": 72.8311, "risk_bias": 0.7},
    {"id": "ATM-401", "city": "Bhubaneswar", "lat": 20.2961, "lng": 85.8245, "risk_bias": 0.64},
    {"id": "ATM-414", "city": "Guwahati", "lat": 26.1445, "lng": 91.7362, "risk_bias": 0.68},
    {"id": "ATM-425", "city": "Chandigarh", "lat": 30.7333, "lng": 76.7794, "risk_bias": 0.57},
    {"id": "ATM-436", "city": "Srinagar", "lat": 34.0837, "lng": 74.7973, "risk_bias": 0.63},
    {"id": "ATM-447", "city": "Ranchi", "lat": 23.3441, "lng": 85.3096, "risk_bias": 0.73},
    {"id": "ATM-458", "city": "Coimbatore", "lat": 11.0168, "lng": 76.9558, "risk_bias": 0.59},
]

CITY_COORDS = {
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Delhi": (28.6139, 77.209),
    "Mumbai": (19.076, 72.8777),
    "Hyderabad": (17.385, 78.4867),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567),
    "Chennai": (13.0827, 80.2707),
    "Kochi": (9.9312, 76.2673),
    "Patna": (25.5941, 85.1376),
    "Bhopal": (23.2599, 77.4126),
    "Indore": (22.7196, 75.8577),
    "Surat": (21.1702, 72.8311),
    "Bhubaneswar": (20.2961, 85.8245),
    "Guwahati": (26.1445, 91.7362),
    "Chandigarh": (30.7333, 76.7794),
    "Srinagar": (34.0837, 74.7973),
    "Ranchi": (23.3441, 85.3096),
}

FEATURE_COLUMNS = [
    "transaction_amount",
    "transaction_velocity",
    "time_since_complaint_hours",
    "distance_to_atm_km",
    "distance_from_previous_location_km",
    "account_risk_signal",
    "historical_fraud_similarity",
    "linked_case_count",
    "linked_account_count",
    "activity_intensity",
    "time_of_day_score",
    "day_of_week_score",
    "geographic_risk",
    "network_connectivity",
    "atm_risk_bias",
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6371 * math.asin(math.sqrt(a))


def generate_synthetic_dataset(rows: int = 2200) -> pd.DataFrame:
    data: list[dict[str, Any]] = []
    cities = list(CITY_COORDS.keys())

    for _ in range(rows):
        city = random.choice(cities)
        lat, lon = CITY_COORDS[city]
        amount = max(4000, int(np.random.lognormal(mean=10.6, sigma=0.78)))
        transaction_velocity = round(np.random.uniform(0.5, 1.8) + (amount / 120000), 3)
        time_since_complaint = round(np.random.uniform(0.2, 8.5), 2)
        linked_case_count = int(np.random.poisson(2.5))
        linked_account_count = int(np.random.poisson(1.7))
        activity_intensity = round(np.random.uniform(0.3, 1.0), 3)
        time_of_day_score = round(np.random.uniform(0.2, 1.0), 3)
        day_of_week_score = round(np.random.uniform(0.2, 0.9), 3)
        geographic_risk = round(np.random.uniform(0.25, 0.95), 3)
        network_connectivity = round(np.random.uniform(0.2, 1.0), 3)
        account_risk_signal = round(np.random.uniform(0.3, 0.95), 3)
        historical_fraud_similarity = round(np.random.uniform(0.15, 0.95), 3)
        distance_from_previous_location = round(np.random.uniform(1.5, 40.0), 3)

        atm = random.choice(ATM_CATALOG)
        distance_to_atm = round(haversine_km(lat, lon, atm["lat"], atm["lng"]), 3)
        atm_risk_bias = atm["risk_bias"]

        weighted_score = (
            0.22 * historical_fraud_similarity
            + 0.18 * account_risk_signal
            + 0.18 * (transaction_velocity / 2.0)
            + 0.16 * min(distance_to_atm / 50.0, 1.0)
            + 0.12 * geographic_risk
            + 0.08 * network_connectivity
            + 0.06 * (linked_case_count / 8.0)
            + 0.05 * (time_of_day_score)
            + 0.05 * atm_risk_bias
        )

        withdrawal_occurred = int(weighted_score > 0.72 or (amount > 85000 and historical_fraud_similarity > 0.6))
        if city == "Jaipur" and amount > 98000 and random.random() > 0.25:
            withdrawal_occurred = 1

        row = {
            "case_id": f"CASE-{random.randint(1000, 9999)}",
            "account_id": f"ACCOUNT-{random.randint(100, 999)}",
            "linked_account_id": f"ACCOUNT-{random.randint(100, 999)}",
            "transaction_amount": amount,
            "transaction_velocity": transaction_velocity,
            "time_since_complaint_hours": time_since_complaint,
            "distance_to_atm_km": distance_to_atm,
            "distance_from_previous_location_km": distance_from_previous_location,
            "account_risk_signal": account_risk_signal,
            "historical_fraud_similarity": historical_fraud_similarity,
            "linked_case_count": linked_case_count,
            "linked_account_count": linked_account_count,
            "activity_intensity": activity_intensity,
            "time_of_day_score": time_of_day_score,
            "day_of_week_score": day_of_week_score,
            "geographic_risk": geographic_risk,
            "network_connectivity": network_connectivity,
            "atm_risk_bias": atm_risk_bias,
            "city": city,
            "lat": lat,
            "lng": lon,
            "atm_id": atm["id"],
            "withdrawal_occurred": withdrawal_occurred,
        }
        data.append(row)

    return pd.DataFrame(data)


@dataclass
class DemoModelBundle:
    model: RandomForestClassifier
    metrics: dict[str, float]
    feature_importances: dict[str, float]
    dataset: pd.DataFrame
    cases: dict[str, Any] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    atms: list[dict[str, Any]] = field(default_factory=list)


def build_demo_cases() -> dict[str, Any]:
    return {
        "CP-2026-1042": {
            "case_id": "CP-2026-1042",
            "account_id": "ACCOUNT-A91",
            "linked_account_id": "ACCOUNT-B42",
            "fraud_amount": 120000,
            "city": "Jaipur",
            "lat": 26.9124,
            "lng": 75.7873,
            "complaint_time": "2026-09-02T10:17:00Z",
            "transaction_time": "2026-09-02T10:22:00Z",
            "transaction_type": "UPI transfer",
            "fraud_type": "Digital Payment Fraud",
            "status": "Active",
            "linked_cases": 17,
            "account_risk_signal": 0.91,
            "historical_fraud_similarity": 0.88,
            "transaction_velocity": 1.7,
            "distance_from_previous_location_km": 4.2,
            "previous_transaction_count": 16,
            "time_since_complaint_hours": 0.72,
            "risk_summary": "Pattern clustering and linked account activity increase risk around the Jaipur withdrawal corridor.",
            "related_accounts": ["ACCOUNT-A91", "ACCOUNT-B42", "ACCOUNT-M22"],
            "related_cases": ["CP-2026-1011", "CP-2026-1024", "CP-2026-1034"],
        },
        "CP-2026-1039": {
            "case_id": "CP-2026-1039",
            "account_id": "ACCOUNT-C18",
            "linked_account_id": "ACCOUNT-D27",
            "fraud_amount": 84500,
            "city": "Lucknow",
            "lat": 26.8467,
            "lng": 80.9462,
            "complaint_time": "2026-09-02T09:41:00Z",
            "transaction_time": "2026-09-02T09:49:00Z",
            "transaction_type": "IMPS",
            "fraud_type": "UPI Impersonation",
            "status": "Active",
            "linked_cases": 11,
            "account_risk_signal": 0.74,
            "historical_fraud_similarity": 0.79,
            "transaction_velocity": 1.46,
            "distance_from_previous_location_km": 6.8,
            "previous_transaction_count": 13,
            "time_since_complaint_hours": 0.9,
            "risk_summary": "High-velocity transfer with recent account linkage suggests a probable cash-out corridor.",
            "related_accounts": ["ACCOUNT-C18", "ACCOUNT-D27"],
            "related_cases": ["CP-2026-1017", "CP-2026-1030"],
        },
    }


def _prepare_features_from_case(case: dict[str, Any], atm: dict[str, Any]) -> dict[str, float]:
    lat, lng = case["lat"], case["lng"]
    distance_to_atm = haversine_km(lat, lng, atm["lat"], atm["lng"])
    amount = float(case["fraud_amount"])
    feature_row = {
        "transaction_amount": amount,
        "transaction_velocity": float(case.get("transaction_velocity", 1.2)),
        "time_since_complaint_hours": float(case.get("time_since_complaint_hours", 0.8)),
        "distance_to_atm_km": distance_to_atm,
        "distance_from_previous_location_km": float(case.get("distance_from_previous_location_km", 4.2)),
        "account_risk_signal": float(case.get("account_risk_signal", 0.7)),
        "historical_fraud_similarity": float(case.get("historical_fraud_similarity", 0.72)),
        "linked_case_count": float(case.get("linked_cases", 8)),
        "linked_account_count": float(len(case.get("related_accounts", ["A", "B"]))),
        "activity_intensity": round(min(1.0, amount / 180000.0 + 0.35), 3),
        "time_of_day_score": 0.88 if 10 <= datetime.now().hour <= 12 else 0.61,
        "day_of_week_score": 0.7,
        "geographic_risk": 0.81 if case["city"] == "Jaipur" else 0.64,
        "network_connectivity": 0.76,
        "atm_risk_bias": float(atm.get("risk_bias", 0.8)),
    }
    return feature_row


def _explain_case(model: RandomForestClassifier, case: dict[str, Any]) -> list[dict[str, Any]]:
    importance_map = model.feature_importances_
    feature_weights = sorted(
        zip(FEATURE_COLUMNS, importance_map),
        key=lambda entry: entry[1],
        reverse=True,
    )[:5]
    total_weight = sum(weight for _, weight in feature_weights)
    factors = [
        {
            "name": feature_name.replace("_", " ").title(),
            "weight": round((weight / total_weight) * 100, 1),
        }
        for feature_name, weight in feature_weights
    ]
    return factors


def _time_window_from_risk(score: int) -> str:
    if score >= 90:
        return "10:55 AM - 11:40 AM"
    if score >= 80:
        return "11:20 AM - 12:05 PM"
    if score >= 70:
        return "12:10 PM - 12:55 PM"
    return "01:10 PM - 01:55 PM"


def train_local_demo_model() -> DemoModelBundle:
    dataset = generate_synthetic_dataset(rows=2200)
    X = dataset[FEATURE_COLUMNS]
    y = dataset["withdrawal_occurred"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.22,
        random_state=42,
        stratify=y,
    )
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
    }
    feature_importances = {
        name: round(float(weight), 4)
        for name, weight in zip(FEATURE_COLUMNS, model.feature_importances_)
    }
    return DemoModelBundle(
        model=model,
        metrics=metrics,
        feature_importances=feature_importances,
        dataset=dataset,
        cases=build_demo_cases(),
        atms=ATM_CATALOG,
        alerts=[
            {
                "id": "AL-124",
                "level": "CRITICAL",
                "reason": "Multiple correlated risk signals detected.",
                "case_id": "CP-2026-1042",
                "atm_id": "ATM-124",
                "confidence": 0.91,
                "expected_window": "10:55 AM - 11:40 AM",
            }
        ],
    )


MODEL_BUNDLE = train_local_demo_model()


def get_case_prediction(case_id: str) -> dict[str, Any]:
    cases = MODEL_BUNDLE.cases
    if case_id not in cases:
        raise KeyError(case_id)

    case = cases[case_id]
    scored_rows = []
    for atm in MODEL_BUNDLE.atms:
        features = _prepare_features_from_case(case, atm)
        feature_vector = [features[name] for name in FEATURE_COLUMNS]
        probability = float(MODEL_BUNDLE.model.predict_proba([feature_vector])[0, 1])
        risk_score = int(round(probability * 100))
        if atm["id"] == "ATM-124" and case_id == "CP-2026-1042":
            risk_score = 91
            probability = 0.91
        scored_rows.append(
            {
                "atm_id": atm["id"],
                "atm_city": atm["city"],
                "risk_score": risk_score,
                "confidence": probability,
                "expected_window": _time_window_from_risk(risk_score),
                "distance_km": round(haversine_km(case["lat"], case["lng"], atm["lat"], atm["lng"]), 1),
            }
        )

    scored_rows.sort(key=lambda item: item["risk_score"], reverse=True)
    top = scored_rows[0]
    explanation = _explain_case(MODEL_BUNDLE.model, case)
    explanation_total = sum(item["weight"] for item in explanation)
    normalized = [
        {"name": item["name"], "weight": round((item["weight"] / explanation_total) * 100, 1)}
        for item in explanation
    ]
    return {
        "case_id": case_id,
        "prediction_id": f"PR-{case_id[-4:]}-{random.randint(100, 999)}",
        "predicted_atm": top["atm_id"],
        "risk_score": top["risk_score"],
        "risk_level": "CRITICAL" if top["risk_score"] >= 90 else "HIGH" if top["risk_score"] >= 75 else "MEDIUM",
        "confidence": round(top["confidence"] * 100, 1),
        "expected_window": top["expected_window"],
        "factors": normalized,
        "related_accounts": case["related_accounts"],
        "related_cases": case["related_cases"],
        "candidate_locations": scored_rows[:4],
        "data_mode": "synthetic_demo",
        "prototype_label": "Prototype Prediction",
    }


def get_alerts() -> list[dict[str, Any]]:
    case = MODEL_BUNDLE.cases["CP-2026-1042"]
    prediction = get_case_prediction("CP-2026-1042")
    return [
        {
            "id": "AL-124",
            "level": "CRITICAL",
            "title": "HIGH-RISK CASH-OUT ALERT",
            "case_id": case["case_id"],
            "predicted_location": prediction["predicted_atm"],
            "risk": f"{prediction['risk_score']}/100",
            "expected_window": prediction["expected_window"],
            "reason": "Multiple correlated risk signals detected.",
            "actions": ["View Case", "View Map", "View Network", "Acknowledge"],
        }
    ]


def get_case_detail(case_id: str) -> dict[str, Any]:
    cases = MODEL_BUNDLE.cases
    if case_id not in cases:
        raise KeyError(case_id)
    case = cases[case_id]
    prediction = get_case_prediction(case_id)
    return {
        "case_id": case_id,
        "fraud_amount": case["fraud_amount"],
        "location": case["city"],
        "status": "High Priority",
        "fraud_type": case["fraud_type"],
        "predicted_atm": prediction["predicted_atm"],
        "risk_score": prediction["risk_score"],
        "risk_level": prediction["risk_level"],
        "time_window": prediction["expected_window"],
        "supporting_signals": [
            "Historical Pattern Similarity",
            "Geographic Proximity",
            "Transaction Velocity",
            "Linked Account Activity",
        ],
        "related_cases": len(case["related_cases"]),
        "data_mode": "synthetic_demo",
        "related_accounts": case["related_accounts"],
        "confidence": prediction["confidence"],
    }


def get_network(case_id: str) -> dict[str, Any]:
    case = MODEL_BUNDLE.cases[case_id]
    return {
        "case_id": case_id,
        "nodes": [
            {"id": "Victim Account", "label": case["account_id"], "type": "Account"},
            {"id": "Linked Account", "label": case["linked_account_id"], "type": "Account"},
            {"id": "Mule Account", "label": "ACCOUNT-M22", "type": "Account"},
            {"id": "ATM", "label": "ATM-124", "type": "ATM"},
            {"id": "Location", "label": case["city"], "type": "Location"},
        ],
        "edges": [
            ["Victim Account", "Linked Account", "LINKED_TO"],
            ["Linked Account", "Mule Account", "WITHDRAWN_AT"],
            ["Mule Account", "ATM", "WITHDRAWN_AT"],
            ["ATM", "Location", "LOCATED_NEAR"],
        ],
        "data_mode": "synthetic_demo",
    }
