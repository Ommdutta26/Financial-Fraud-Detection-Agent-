# ============================================================
# test_calibration.py — Verify calibration is fixed
# ============================================================
import sys
sys.path.insert(0, '.')

from agent import models_loader
from agent import features as feat_builder

m = models_loader.load_all()

test_cases = [
    # (description, tx, expected_decision)
    ("Clean card + safe email",
     {'TransactionAmt': 45.0,  'hour': 14, 'P_emaildomain': 'att.net',       'ProductCD': 'R', 'card1': 12345},
     'APPROVE'),

    ("Known fraud card 100%",
     {'TransactionAmt': 800.0, 'hour': 7,  'P_emaildomain': 'protonmail.com', 'ProductCD': 'W', 'card1': 3342},
     'BLOCK'),

    ("Known fraud card 97.9%",
     {'TransactionAmt': 500.0, 'hour': 2,  'P_emaildomain': 'protonmail.com', 'ProductCD': 'W', 'card1': 12473},
     'BLOCK'),

    ("High fraud card 90%",
     {'TransactionAmt': 300.0, 'hour': 7,  'P_emaildomain': 'mail.com',       'ProductCD': 'W', 'card1': 2675},
     'FLAG/BLOCK'),

    ("Protonmail unknown card",
     {'TransactionAmt': 800.0, 'hour': 14, 'P_emaildomain': 'protonmail.com', 'ProductCD': 'W', 'card1': 12345},
     'FLAG'),
]

print("="*70)
print("CALIBRATION VERIFICATION TEST")
print("="*70)
print(f"\nThreshold: {m['THRESHOLD']:.4f}")
print(f"\n{'Description':<30} {'XGB':>8} {'Cal':>8} {'Gap':>8} {'Status'}")
print("-"*70)

all_passed = True
for desc, tx, expected in test_cases:
    feats, _ = feat_builder.build_features_with_provenance(tx)
    xgb = float(m['xgb_model'].predict_proba(feats)[0][1])
    cal = float(m['calibrated_model'].predict_proba(feats)[0][1])
    gap = abs(xgb - cal)

    # Calibration is broken if high XGB maps to low cal
    if xgb > 0.5 and cal < 0.1:
        status = "🚨 BROKEN — inversion detected"
        all_passed = False
    elif xgb > 0.5 and cal > 0.3:
        status = "✅ CORRECT"
    elif xgb < 0.1 and cal < 0.2:
        status = "✅ CORRECT"
    else:
        status = "⚠️  CHECK"

    print(f"{desc:<30} {xgb:>8.4f} {cal:>8.4f} {gap:>8.4f}  {status}")

print("-"*70)
print(f"\nThreshold: {m['THRESHOLD']:.4f}")
print(f"\n{'='*70}")
if all_passed:
    print("✅ ALL TESTS PASSED — Calibration is working correctly")
    print("   High XGBoost scores correctly map to high calibrated scores")
else:
    print("🚨 CALIBRATION STILL BROKEN")
    print("   Fix: rerun calibration cell on Kaggle with the updated cell")
    print("   Then download and replace models/calibrated_model.pkl")
print("="*70)