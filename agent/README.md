# Fraud Detection Agent — Project Structure

```
fraud_agent/
│
├── agent.py            ← Public entry point — call run_agent() from here
├── config.py           ← All constants & thresholds (edit here first when tuning)
│
├── models_loader.py    ← Loads all .pkl files once at startup
├── features.py         ← Raw transaction dict → model feature vector
├── scoring.py          ← XGBoost / ensemble inference + SHAP explanations
├── rules.py            ← Pattern flags & deterministic business rules
├── llm_client.py       ← Groq API wrapper, prompt builder, response parser
├── nodes.py            ← LangGraph node functions (one concern each)
├── graph.py            ← Assembles and compiles the StateGraph pipeline
├── report.py           ← Formats the investigation report string
│
└── models/             ← Pickled model files (not in source control)
    ├── xgb_model.pkl
    ├── shap_explainer.pkl
    ├── threshold.pkl
    ├── feature_cols.pkl
    ├── label_encoders.pkl
    ├── iso_model.pkl          (optional — enables ensemble)
    ├── iso_norm_params.pkl    (optional)
    └── ensemble_weights.pkl   (optional)
```

## Debugging guide

| Symptom | File to check |
|---|---|
| Wrong feature values | `features.py` → `_build_row()` |
| Model not loading | `models_loader.py` → `load_all()` |
| Score seems off | `scoring.py` → `score_transaction()` |
| Wrong risk level / thresholds | `config.py` → `RISK_BANDS`, `THRESHOLD` |
| Rules firing incorrectly | `rules.py` → `get_rule_flags()` |
| Pattern flags wrong | `rules.py` → `get_pattern_flags()` |
| LLM decision is wrong | `llm_client.py` → `build_decision_prompt()` / `parse_llm_response()` |
| Report formatting broken | `report.py` → `build_report()` |
| Graph flow wrong | `graph.py` → edges / `nodes.route_after_score()` |

## Data flow

```
run_agent(tx)
    └─ features.py    build_features(tx) → np.ndarray
    └─ graph.py       invoke(state)
           ├─ nodes.node_score         → scoring.score_transaction()
           ├─ [fast path] node_fast_approve
           └─ [full path]
                  ├─ nodes.node_patterns  → rules.get_pattern_flags()
                  ├─ nodes.node_rules     → rules.get_rule_flags()
                  ├─ nodes.node_explain   → scoring.get_shap_explanations()
                  ├─ nodes.node_decide    → llm_client.call_groq()
                  └─ nodes.node_report    → report.build_report()
```
