from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from agent.agent import run_agent
import requests
import uvicorn


# ============================================================
# Create FastAPI app FIRST
# ============================================================

app = FastAPI(
    title="Fraud Detection API",
    version="1.0"
)


# ============================================================
# Input Schema
# ============================================================

class TransactionRequest(BaseModel):
    data: Dict[str, Any]


# ============================================================
# Health Check
# ============================================================

@app.get("/")
def health():
    return {"status": "API running"}


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict(transaction: TransactionRequest):

    user_input = transaction.data

    result = run_agent(user_input)

    response = {
        "transaction_id": f"TXN-{user_input.get('card1', 'UNKNOWN')}",
        "risk_level": result.get("risk_level"),
        "score": result.get("score"),
        "decision": result.get("decision"),
        "confidence": result.get("confidence"),
        "agent_reasoning": result.get("agent_reasoning"),
    }

    # Send to n8n webhook
    try:
        requests.post(
            "https://omm-dutta.app.n8n.cloud/webhook-test/fraud-alert",
            json=response,
            timeout=5
        )
    except Exception as e:
        print("n8n webhook failed:", e)

    return response


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )