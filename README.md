# CyberPredict

AI-powered predictive intelligence prototype for proactive cybercrime intervention.

## Run the frontend

Open `index.html` in a browser. The frontend uses Leaflet and OpenStreetMap for the interactive map. It is usable in demo mode with anonymized/synthetic data.

## Run the API

```powershell
cd backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

API docs are available at `http://127.0.0.1:8000/docs`.

The API currently exposes typed synthetic demo endpoints for dashboard metrics, cases, predictions, ATMs, alerts, fraud networks, Copilot responses, and intelligence briefs. PostgreSQL/PostGIS/Neo4j are intended integration targets; no confidential banking or police data is included.

## Responsible use

Predictions are probabilistic intelligence signals, not proof of criminal activity. All outputs require authorized investigator verification. The prototype uses anonymized/synthetic demonstration data.
