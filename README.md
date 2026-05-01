# 🔍 Fraud Detection Agent

> A multi-agent AI system for real-time transaction fraud detection — combining XGBoost ensemble scoring, LangGraph orchestration, SHAP explainability, and Groq LLM reasoning in a Streamlit dashboard.

---

## 📌 Overview

This project is a production-style fraud detection pipeline built on the **IEEE-CIS Fraud Detection** dataset. It goes beyond a simple ML model — every transaction is processed through a **6-node agent pipeline** that scores, explains, and reasons about risk before delivering a final decision with a human-readable report.

Built as a placement portfolio project demonstrating end-to-end ML engineering: feature engineering, ensemble modeling, agentic AI, and interactive UI.

---

## 🧠 How It Works

Each transaction passes through 6 sequential agent nodes:

```
Risk Scorer → Pattern Analyzer → Rule Engine → SHAP Explainer → Groq LLM → Report Writer
```

| Step | Node | What It Does |
|------|------|--------------|
| 1 | **Risk Scorer** | Runs the XGBoost + Isolation Forest ensemble; produces a fraud probability score |
| 2 | **Pattern Analyzer** | Detects behavioural signals (round amounts, off-hours activity, risky email domains, high-risk product codes) |
| 3 | **Rule Engine** | Applies hard business rules (velocity checks, block-listed domains, amount thresholds) |
| 4 | **SHAP Explainer** | Computes per-feature SHAP values so every decision is explainable |
| 5 | **Groq LLM** | Uses Llama 3.3 70B to synthesise all signals into a natural-language reasoning summary |
| 6 | **Report Writer** | Assembles a structured investigation report ready for download |

**Final decision:** `APPROVE` / `FLAG` / `BLOCK` — with confidence %, risk level, and threshold comparison.


## 📊 Model Details

| Property | Value |
|----------|-------|
| Dataset | IEEE-CIS Fraud Detection (Kaggle) |
| Base model | XGBoost |
| Ensemble | XGBoost + Isolation Forest |
| Explainability | SHAP (TreeExplainer) |
| LLM | Groq — Llama 3.3 70B |
| Orchestration | LangGraph |

### Key engineered features

- `card_email_fraud_rate` — historical fraud rate for the card × email domain combination
- `card2_amount_deviation` — how much this transaction deviates from the card's typical spend
- `C13`, `C14` — Vesta anonymised count features (transaction velocity signals)
- `hour` — extracted from `TransactionDT`; captures time-of-day risk patterns

---

## 🚩 Fraud Heuristics

### Pattern flags (soft signals)
| Flag | Rationale |
|------|-----------|
| Suspiciously round amount | Fraudsters testing cards or manually entering amounts tend to use round figures (e.g. $100, $500) |
| Off-hours transaction | Activity between midnight and 5 AM has elevated fraud rates |
| High-risk email domain | Disposable/anonymous domains (e.g. `guerrillamail.com`) correlate with fraud |
| High-risk product code | Product `W` (Digital Goods) and `C` (Cash) are instant and untraceable |

### Rule flags (hard rules)
| Rule | Threshold |
|------|-----------|
| Very high amount | > $5,000 |
| Blocked email domain | `anonymous.com`, `guerrillamail.com` |
| Late-night + high amount | Hour ∈ [0,5] AND amount > $1,000 |

---

## 🖥️ Dashboard Features

- **Risk gauge** — animated dial showing fraud probability vs. threshold
- **SHAP waterfall bar chart** — per-feature contribution to the score (red = increases risk, green = decreases risk)
- **Agent pipeline tracker** — live summary of what each node found
- **AI reasoning panel** — plain-English explanation from Groq LLM
- **Downloadable report** — full investigation report as `.txt`
- **Session history** — score trend line chart + decision distribution donut across all transactions in the session

---

## ⚙️ Setup & Running

### 1. Clone and install

```bash
git clone https://github.com/your-username/fraud-detection-agent.git
cd fraud-detection-agent
pip install -r requirements.txt
```

### 2. Set your Groq API key

```bash
export GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Train the model (first time only)

```bash
python train.py
```

This saves the XGBoost model, Isolation Forest, SHAP explainer, and `models/model_summary.json`.

### 4. Run the dashboard

```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
plotly
pandas
numpy
xgboost
shap
scikit-learn
langgraph
langchain-groq
```

---

## 🔬 Sample Decision Interpretation

```
Score: 0.36  |  Threshold: 0.89  |  Decision: FLAG
```

Even though the ML score is well below the block threshold, the transaction is **flagged** rather than auto-approved because:
- Product W (Digital Goods) + round amount ($500) is a known fraud pattern
- The SHAP top feature (`card_email_fraud_rate`) suggests the card/email pair has historical fraud association
- C13 and C14 velocity counters are elevated

This is intentional — the rule and pattern layers act as a **safety net** that the ML score alone can miss.

---

## 📈 Roadmap

- [ ] Add real-time card velocity tracking (transactions per hour per card)
- [ ] Integrate device fingerprint features
- [ ] Add feedback loop — analysts can mark decisions as correct/incorrect to retrain
- [ ] Deploy to Streamlit Cloud with environment secret management
- [ ] Add batch CSV upload mode for offline investigation

---

## 🙏 Acknowledgements

- [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)
- [Groq](https://groq.com) for ultra-fast LLM inference
- [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- [SHAP](https://github.com/slundberg/shap) for model explainability

---

<p align="center">
  Built for placement portfolio · XGBoost + LangGraph + Groq LLM · IEEE-CIS Dataset
</p>
