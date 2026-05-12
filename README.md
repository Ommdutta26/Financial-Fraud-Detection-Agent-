# 🔍 Fraud Detection Agent

> A multi-agent AI system for real-time transaction fraud detection — combining Calibrated XGBoost, LangGraph orchestration, SHAP explainability, Groq LLM reasoning, LangSmith observability, n8n alerting, and in-session velocity memory in a Streamlit dashboard.

## 🚀 Live Demo

👉 https://financial-fraud-detection-agent.onrender.com/

> ⚠️ The app may take a few seconds to start due to cold boot on Render.

---

## 📌 Overview

Production-style fraud detection pipeline built on the **IEEE-CIS Fraud Detection** dataset (590,540 transactions). Every transaction passes through a **7-node LangGraph agent pipeline** that scores, explains, and reasons about risk before delivering a decision with a downloadable investigation report.

Every node execution is **fully traced in LangSmith** — latency, token usage, decision metadata, and analyst feedback captured per transaction. Flagged and blocked transactions are automatically routed through an **n8n workflow** that sends Telegram alerts and stores decisions in MongoDB.

Built as a placement portfolio project demonstrating end-to-end ML engineering: data-driven feature engineering, calibrated XGBoost, agentic AI orchestration, SHAP explainability, LLM reasoning, production observability, and automated alerting.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9247** |
| PR-AUC | **0.6562** |
| F1 Score | **0.6552** |
| Dataset | 590,540 transactions |
| Fraud rate | 3.50% |
| Training samples | 472,432 |
| Features | 29 (inferable at inference) |
| Calibration | Isotonic regression |

> **Why PR-AUC matters more than ROC-AUC for fraud:** With only 3.5% fraud rate, ROC-AUC is inflated by the dominant legitimate class. PR-AUC measures how well the model finds the rare fraud cases — 0.66 on a 3.5% base rate represents ~19× better than random guessing.

---

## 🔄 Production Pipeline Architecture

```
Your App / API
     │
     ▼
[Streamlit Dashboard]
     │  POST transaction
     ▼
[Fraud Agent — LangGraph]
  ├── node: score       ──→ XGBoost + Isotonic Calibration
  ├── node: memory      ──→ In-session velocity tracking
  ├── node: patterns    ──→ Behavioral soft-signal flags
  ├── node: rules       ──→ 8 deterministic business rules
  ├── node: explain     ──→ SHAP + counterfactual analysis
  ├── node: decide      ──→ Groq LLM structured reasoning
  └── node: report      ──→ Full investigation report
     │
     ├──→ [LangSmith]        Trace every node, latency,
     │                       token usage, analyst feedback,
     │                       transaction metadata per run
     │
     └──→ [n8n Webhook]      Receive decision payload
               │
               ▼
          [Switch node]       Route by risk level
          ├── APPROVE  ──→   MongoDB (log only)
          ├── FLAG     ──→   Telegram alert + Edit Fields + MongoDB
          ├── BLOCK    ──→   Telegram (urgent) + Edit Fields + MongoDB
          └── ERROR    ──→   DevOps notification
```

---

## 🧠 How It Works

Each transaction passes through 7 sequential agent nodes:

```
Risk Scorer → Memory Check → Pattern Analyzer → Rule Engine →
SHAP Explainer → Groq LLM → Report Writer
```

| Step | Node | What It Does |
|------|------|--------------|
| 1 | **Risk Scorer** | Calibrated XGBoost produces honest fraud probability score |
| 2 | **Memory Check** | Checks in-session card velocity — escalating amounts, burst transactions |
| 3 | **Pattern Analyzer** | Detects soft behavioural signals — risky hours, email domains, amount patterns |
| 4 | **Rule Engine** | Applies 8 hard business rules — AML thresholds, product+email combos |
| 5 | **SHAP Explainer** | Per-feature contribution + counterfactual ("what would make this safer?") |
| 6 | **Groq LLM** | Llama 3.3 70B synthesises all signals into 4-step structured reasoning |
| 7 | **Report Writer** | Full investigation report with historical context, ready for download |

**Final decision:** `APPROVE` / `FLAG` / `BLOCK` — with calibrated confidence %, risk level, and threshold.

> **Smart routing:** Low-risk transactions (score < 0.001 AND zero rule flags) fast-approve at Node 1, skipping the LLM entirely. This reduces Groq API calls by ~40%.

---

## 📡 Observability — LangSmith

Every fraud decision is fully traced end-to-end in LangSmith:

| What's Traced | Details |
|---|---|
| Pipeline latency | Per-node timing (score → decide → report) |
| LLM calls | Groq prompt, token count, raw response, latency |
| Transaction metadata | Amount, email, card, hour, product |
| Decision output | APPROVE/FLAG/BLOCK + confidence + risk level + flag count |
| Analyst feedback | Correct flag / false positive logged back via `run_id` |

**Trace structure in LangSmith:**
```
fraud-agent  (root trace)
├── score          → xgb, calibrated score, risk level
├── memory         → velocity flags
├── patterns       → pattern flags detected
├── rules          → business rules triggered
├── explain        → top SHAP features, counterfactual
├── groq-fraud-decision  → prompt tokens, latency, raw LLM response
│   └── decide     → APPROVE/FLAG/BLOCK, confidence
└── generate_report
```

**Setup:**
```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=fraud-agent
```

Traces appear at [smith.langchain.com](https://smith.langchain.com) under the `fraud-agent` project.

---

## ⚡ Alerting — n8n Workflow

Flagged and blocked transactions are automatically POSTed to an n8n webhook and routed through a pipeline:

| Node | Purpose |
|---|---|
| **Webhook** | Receives decision payload from fraud agent |
| **Switch** | Routes by decision — APPROVE / FLAG / BLOCK / ERROR |
| **Telegram** | Sends structured alert for FLAG and BLOCK decisions |
| **Edit Fields** | Normalises and transforms payload for storage |
| **MongoDB** | Stores all decisions for full audit trail |

**Payload sent to n8n:**
```json
{
  "decision": "FLAG",
  "risk_level": "MEDIUM",
  "score": 0.0054,
  "confidence": 80,
  "rule_flags": ["R04", "R06", "R07"],
  "pattern_flags": 7,
  "reasoning": "Disposable email domain with round amount in fraud sweet spot...",
  "amount": 800.00,
  "email": "protonmail.com",
  "hour": 14,
  "langsmith_run_id": "abc-123-xyz"
}
```

**Sample Telegram alert:**
```
🚨 FRAUD ALERT

📊 Model Score:  0.0054  (LOW)
📋 Rules Fired:  R04, R06, R07
⚔️  Conflict: Rules overrode model score
✅ Final Decision: FLAG — human review required

Amount:  $800.00
Email:   protonmail.com
Hour:    14:00
Risk:    MEDIUM | Confidence: 80%

Reasoning: Disposable email domain (protonmail.com) combined
with round amount $800 in fraud sweet spot ($500-$1K range).
Card+email combination seen for first time.

🔗 LangSmith Trace: smith.langchain.com/...

This message was sent automatically with n8n
```

Import `workflows/n8n_fraud_alert.json` to replicate the full workflow.

---

## 🏗️ Project Structure

```
fraud_dashboard/
├── app.py                          # Entry point — orchestration only
├── styles/
│   └── custom_css.py               # CSS + decision color maps
├── components/
│   ├── sidebar.py                  # Input controls + model info
│   ├── charts.py                   # Plotly figure builders
│   ├── result_display.py           # Post-analysis render
│   └── history.py                  # Session state + history panel
├── agent/
│   ├── agent.py                    # Public entry point (@traceable root)
│   ├── config.py                   # All constants (thresholds, email scores)
│   ├── features.py                 # Transaction → 29-feature vector
│   ├── graph.py                    # LangGraph StateGraph
│   ├── nodes.py                    # 7 node functions + smart router
│   ├── scoring.py                  # XGBoost + calibration inference
│   ├── rules.py                    # Rule engine + velocity tracker
│   ├── llm_client.py               # Groq LLM wrapper (@traceable LLM node)
│   └── report.py                   # Report generator
├── models/
│   ├── xgb_model.pkl               # XGBoost model
│   ├── calibrated_model.pkl        # Isotonic-calibrated XGBoost
│   ├── shap_explainer.pkl          # SHAP TreeExplainer
│   ├── feature_cols.pkl            # 29 feature column list
│   ├── threshold.pkl               # Optimal threshold + operating points
│   ├── feature_store.pkl           # Historical aggregates (cards, emails)
│   ├── imputation.pkl              # Training-set medians
│   ├── counterfactuals.pkl         # Risk profiles by hour/amount
│   ├── fraud_examples.pkl          # Top fraud case profiles
│   └── agent_config.json           # Agent configuration
└── workflows/
    └── n8n_fraud_alert.json        # n8n workflow export (importable)
```

---

## ⚙️ Features Used (29 Total)

The model trains on **only features that can be populated at inference time** — eliminating training-serving skew.

| Group | Features | Source |
|-------|----------|--------|
| Direct inputs | `TransactionAmt`, `hour`, `card1` | UI |
| Time derived | `is_night`, `is_early_morning`, `is_risky_hour`, `is_business_hour`, `is_weekend` | Computed |
| Amount derived | `amount_log`, `amount_dec`, `is_round_amt`, `is_fraud_sweetspot`, `is_card_test_amt` | Computed |
| Card history | `card1_fraud_rate`, `card1_tx_count`, `card1_avg_amount`, `card1_amount_deviation`, `card1_max_amount` | Feature store |
| Email history | `email_fraud_rate`, `p_email_high_risk`, `card_email_fraud_rate` | Feature store |
| Graph/network | `card_email_count`, `card_device_count`, `card_addr_count`, `card1_unique_emails` | Feature store |
| Velocity | `card_amount_rank`, `card_tx_sequence`, `is_first_tx` | Computed |

### Data-driven insights built into features
- `is_risky_hour` — hours 5-9AM have **3.5× higher fraud score** than midday (from counterfactual analysis)
- `is_fraud_sweetspot` — $500-$1K range has **5.6% fraud rate** — highest of any amount band
- `email_fraud_rate` — `protonmail.com` has **40.8% historical fraud rate** in training data

---

## 🚩 Fraud Detection Layers

### Pattern flags (soft signals)
| Flag | Rationale |
|------|-----------|
| Early morning (5-9AM) | Peak fraud window — 3.5× higher risk than midday |
| Round amount on digital goods | Classic card-testing pattern (R07) |
| $500-$1K amount | Fraud sweet spot — highest fraud rate band in training data |
| High-risk email domain | `protonmail.com` (40.8%), `mail.com` (19%), `outlook.es` (13%) |
| High card+email fraud rate | Historical fraud rate for this exact card+email combo |
| Unknown card | First time this card appears — no history to compare |

### Rule flags (hard rules)
| Rule | Condition |
|------|-----------|
| R01 | Amount > $5,000 |
| R02 | Amount > $10,000 — AML reporting required |
| R03 | Night (0-5AM) + amount > $500 |
| R04 | High-risk email + amount > $200 |
| R05 | XGBoost score > 0.75 (CRITICAL band) |
| R06 | High-risk product + risky email + amount > $500 |
| R07 | Round amount + digital goods — card-testing pattern |
| R08 | Early morning (5-9AM) + amount > $500 |

### Velocity detection (in-session memory)
| Alert | Condition |
|-------|-----------|
| Burst transactions | 3+ transactions on same card within 1 hour |
| Escalating amounts | Each transaction larger than the last |
| Amount spike | This transaction > 3× card's session average |

> **Note:** When model score is low but rules fire, the system correctly escalates. This model/rules conflict is logged in LangSmith and flagged in the Telegram alert so analysts understand the signal disagreement.

---

## 🧪 Test Cases

### BLOCK — known fraud cards
```
Amount: $800   Hour: 7   Email: protonmail.com  Product: W  Card: 3342
Amount: $500   Hour: 2   Email: protonmail.com  Product: W  Card: 12473
Amount: $300   Hour: 7   Email: mail.com        Product: W  Card: 2675
```

### FLAG — suspicious signals
```
Amount: $800   Hour: 14  Email: protonmail.com  Product: W  Card: 12345
Amount: $12000 Hour: 10  Email: gmail.com       Product: S  Card: 11111
Amount: $300   Hour: 8   Email: outlook.es      Product: W  Card: 12345
```

### APPROVE — clean transactions
```
Amount: $45    Hour: 14  Email: att.net         Product: R  Card: 12345
Amount: $120   Hour: 11  Email: verizon.net     Product: S  Card: 99999
Amount: $200   Hour: 13  Email: gmail.com       Product: R  Card: 12345
```

### Velocity test — run in sequence with same card
```
Tx 1:  Amount: $97.50   Hour: 14  Email: gmail.com  Product: R  Card: 7777
Tx 2:  Amount: $183.25  Hour: 14  Email: gmail.com  Product: R  Card: 7777
Tx 3:  Amount: $342.00  Hour: 14  Email: gmail.com  Product: R  Card: 7777
Tx 4:  Amount: $650.75  Hour: 14  Email: gmail.com  Product: R  Card: 7777
```
By Tx 3/4 you should see: `📈 Escalating amounts — card-testing pattern`

---

## 🖥️ Dashboard Features

- **Risk gauge** — animated dial showing calibrated fraud probability vs. optimal threshold
- **Score breakdown** — XGBoost raw vs. calibrated score with risk summary
- **7-step pipeline tracker** — what each node found per transaction
- **SHAP bar chart** — per-feature impact (red = increases risk, green = decreases)
- **Counterfactual panel** — "what would make this transaction safer?"
- **AI reasoning** — 4-step structured analysis from Groq LLM
- **Historical context** — card seen before, historical fraud rate, amount vs. average
- **Session history** — score trend + decision distribution + velocity alerts
- **Downloadable report** — full investigation report as `.txt`

---

## ⚙️ Setup & Running

### 1. Clone and install
```bash
git clone https://github.com/your-username/fraud-detection-agent.git
cd fraud-detection-agent
pip install -r requirements.txt
```

### 2. Set environment variables
```bash
# Required
export GROQ_API_KEY=your_groq_key_here

# LangSmith observability (optional but recommended)
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=your_langsmith_key_here
export LANGSMITH_PROJECT=fraud-agent
```

Get a free Groq key at [console.groq.com](https://console.groq.com).
Get a free LangSmith key at [smith.langchain.com](https://smith.langchain.com).

### 3. Set up n8n alerting (optional)
1. Install n8n: `npm install -g n8n` or use [n8n.io cloud](https://n8n.io)
2. Import `workflows/n8n_fraud_alert.json`
3. Add your Telegram bot token and MongoDB credentials
4. Set your n8n webhook URL in `agent/config.py`:
```python
N8N_WEBHOOK_URL = "https://your-n8n-instance/webhook/fraud-alert"
```

### 4. Train the model (Kaggle — first time only)
Run the Kaggle notebook end-to-end. Download all `.pkl` files from `/kaggle/working/` and place in `models/`.

### 5. Run the dashboard
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
groq
python-dotenv
langsmith
requests
```

---

## 🔬 Key Design Decisions

**Why 29 features instead of 430+?**
The full IEEE-CIS dataset has 430+ features but most are unavailable at inference without a live data pipeline. Training on all 430 but providing only 5 at inference caused training-serving skew — every transaction scored near zero. Retraining on 29 inferable features fixed this completely.

**Why isotonic calibration?**
Raw XGBoost scores are not true probabilities. Isotonic calibration ensures when the model says 80% fraud probability, approximately 80% of those cases were actually fraud in validation. This makes the confidence % statistically meaningful.

**Why LangGraph over a simple chain?**
LangGraph enables conditional routing — low-risk transactions (score < threshold AND zero rule flags) fast-approve at Node 1, skipping the LLM entirely. This reduces Groq API calls by ~40%. The velocity override in Node 6 shows how rule signals can override model signals without retraining.

**Why data-driven email risk scores?**
Initial assumption was `anonymous.com` = high risk. Data showed it has only 2.3% fraud rate — lower than `gmail.com` (4.4%). `protonmail.com` at 40.8% is the real highest-risk domain. All risk scores are derived from actual training data fraud rates, not domain-name heuristics.

**Why LangSmith?**
The fraud agent makes decisions that affect real transactions. Every LLM call, every rule that fires, every node's latency needs to be observable. LangSmith provides per-trace visibility including the `langsmith_run_id` that links MongoDB records directly back to the exact trace for any decision.

**Why n8n for alerting?**
n8n provides a no-code workflow layer between the fraud agent and downstream systems (Telegram, MongoDB, email). New alert destinations (Slack, PagerDuty, JIRA) can be added without touching the Python codebase.

**Why rules can override a low model score?**
A 0.005 model score on a ProtonMail + $800 + digital goods transaction looks clean to the model — it learned from historical base rates. But 3 business rules fire because the combination violates known fraud policy regardless of probability. The system correctly escalates and logs the model/rules conflict in LangSmith for retraining data collection.

---

## 📈 Roadmap

- [x] Real-time card velocity tracking (in-session memory)
- [x] Data-driven email risk scores from training data
- [x] Counterfactual explanations
- [x] Calibrated confidence scores
- [x] 8 business rules including AML threshold
- [x] LangSmith tracing + observability
- [x] n8n alerting via Telegram + MongoDB
- [x] Smart router — rule pre-check before fast approve
- [ ] Telegram bot with APPROVE/REJECT buttons (human-in-the-loop)
- [ ] Analyst feedback loop → LangSmith dataset → retrain pipeline
- [ ] Batch CSV upload mode for offline investigation
- [ ] Redis for persistent cross-session velocity memory
- [ ] LangSmith eval dataset auto-populated from flagged transactions
- [ ] Email/Slack alert channels via n8n

---

## 🙏 Acknowledgements

- [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)
- [Groq](https://groq.com) for ultra-fast LLM inference
- [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- [LangSmith](https://smith.langchain.com) for agent observability and tracing
- [n8n](https://n8n.io) for workflow automation and alerting
- [SHAP](https://github.com/slundberg/shap) for model explainability

---

<p align="center">
  Fraud Detection Agent · Calibrated XGBoost + LangGraph + Groq LLM + LangSmith + n8n · IEEE-CIS Dataset · Built for placement portfolio
</p>
