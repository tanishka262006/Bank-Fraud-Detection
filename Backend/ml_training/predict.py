
import sys
import json
import numpy as np
import pandas as pd
import joblib
try:
    model_artifact = joblib.load("best_fraud_model_v2.pkl")
except FileNotFoundError as e:
    print(json.dumps({"error": f"Missing artifact: {e}"}))
    sys.exit(1)

model = model_artifact["model"]
threshold = float(model_artifact["threshold"])
feature_names = model_artifact["features"]
le = model_artifact["type_encoder"]
p99 = model_artifact["high_amount_threshold"]
night_hours = model_artifact["night_hours"]
max_train_amount = model_artifact.get("max_train_amount", 1e9)

def prepare_features(data: dict) -> pd.DataFrame:
    hour = int(data["step"])
    amount = float(data["amount"])
    tx_type = data["type"]
    old_bal_orig = float(data["oldbalanceOrg"])
    new_bal_orig = float(data["newbalanceOrig"])
    old_bal_dest = float(data["oldbalanceDest"])
    new_bal_dest = float(data["newbalanceDest"])
    if amount > max_train_amount * 2:
        raise ValueError(f"Amount exceeds realistic bounds (Max typically ~{max_train_amount:.0f}). Please enter a realistic transaction amount.")
    log_amount = np.log1p(amount)
    is_high_amount = 1 if amount > p99 else 0
    is_night = 1 if hour in night_hours else 0
    balance_diff_orig = old_bal_orig - new_bal_orig
    balance_diff_dest = new_bal_dest - old_bal_dest

    sender_error = abs((old_bal_orig - amount) - new_bal_orig)
    receiver_error = abs((old_bal_dest + amount) - new_bal_dest)

    try:
        type_enc = int(le.transform([tx_type])[0])
    except ValueError:
        type_enc = 0

    row = {
        "amount": amount,
        "log_amount": log_amount,
        "is_high_amount": is_high_amount,
        "hour": hour,
        "is_night": is_night,
        "balance_diff_orig": balance_diff_orig,
        "balance_diff_dest": balance_diff_dest,
        "type_enc": type_enc,
        "sender_error": sender_error,
        "receiver_error": receiver_error
    }
    df = pd.DataFrame([row])[feature_names]
    return df

def predict_transaction(data: dict) -> dict:
    input_df = prepare_features(data)
    fraud_prob = float(model.predict_proba(input_df)[0, 1])
    is_fraud = fraud_prob >= threshold

    return {
        "fraud_probability": round(fraud_prob, 6),
        "is_fraud": bool(is_fraud),
        "threshold": round(threshold, 6),
    }
if __name__ == "__main__":
    try:
        raw = sys.stdin.read()
        data = json.loads(raw)
        result = predict_transaction(data)
        print(json.dumps(result))
    except ValueError as e:
        print(json.dumps({"error": f"Invalid transaction type or data: {e}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
