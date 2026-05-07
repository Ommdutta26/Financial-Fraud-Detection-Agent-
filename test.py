# Save as check.py — run: python check.py
import joblib

fc = joblib.load('models/feature_cols.pkl')
print(f"Feature count: {len(fc)}")
print(f"Features: {fc}")
# Replace test.py content with this
import sys
sys.path.insert(0, '.')

from agent import models_loader
from agent import features as feat_builder

m = models_loader.load_all()

test_cases = [
    {'TransactionAmt': 45.0,   'hour': 14, 'P_emaildomain': 'att.net',        'ProductCD': 'R', 'card1': 12345},
    {'TransactionAmt': 800.0,  'hour': 14, 'P_emaildomain': 'protonmail.com',  'ProductCD': 'W', 'card1': 12345},
    {'TransactionAmt': 800.0,  'hour': 7,  'P_emaildomain': 'protonmail.com',  'ProductCD': 'W', 'card1': 3342},
    {'TransactionAmt': 500.0,  'hour': 2,  'P_emaildomain': 'protonmail.com',  'ProductCD': 'W', 'card1': 12473},
    {'TransactionAmt': 300.0,  'hour': 7,  'P_emaildomain': 'mail.com',        'ProductCD': 'W', 'card1': 2675},
]

print(f"\nThreshold: {m['THRESHOLD']:.4f}")
print(f"\n{'Card':<8} {'Email':<22} {'Amt':>7} {'Hr':>3} {'XGB':>8} {'Cal':>8} {'Expected'}")
print("-" * 75)

for tx in test_cases:
    feats, prov = feat_builder.build_features_with_provenance(tx)
    xgb = float(m['xgb_model'].predict_proba(feats)[0][1])
    cal = float(m['calibrated_model'].predict_proba(feats)[0][1])
    thresh = m['THRESHOLD']
    decision = 'BLOCK' if cal >= thresh else 'FLAG' if cal >= thresh*0.5 else 'APPROVE'
    print(f"{tx['card1']:<8} {tx['P_emaildomain']:<22} "
          f"${tx['TransactionAmt']:>6.0f} {tx['hour']:>3} "
          f"{xgb:>8.4f} {cal:>8.4f}  {decision}")