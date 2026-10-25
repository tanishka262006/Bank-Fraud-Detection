import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
import xgboost as xgb

print("Loading dataset...")
df = pd.read_csv('Fraud.csv')

# ── 1. Basic feature engineering (computed before split for convenience, independent of train/test)
# In reality, these don't leak information as they are row-level.
print("Applying basic row-level feature engineering...")
df['log_amount'] = np.log1p(df['amount'])
df['hour'] = df['step'] % 24
night_hours = [0, 1, 2, 3, 4, 5, 22, 23]
df['is_night'] = df['hour'].isin(night_hours).astype(int)
df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

df['sender_error'] = abs((df['oldbalanceOrg'] - df['amount']) - df['newbalanceOrig'])
df['receiver_error'] = abs((df['oldbalanceDest'] + df['amount']) - df['newbalanceDest'])

# Label encode type
le = LabelEncoder()
df['type_enc'] = le.fit_transform(df['type'])

# ── 2. Train/Val/Test Split (60/20/20)
print("Splitting dataset into Train/Val/Test (60/20/20)...")
X = df.drop(columns=['isFraud', 'isFlaggedFraud'] if 'isFlaggedFraud' in df.columns else ['isFraud'])
y = df['isFraud'].values

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
# X_temp is 80% of total. We want Val to be 20% of total, which is 0.25 of 80%
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

# ── 3. Calculate Train-only artifacts
print("Calculating training artifacts (p99 amount, scale_pos_weight)...")
p99_amount = float(X_train['amount'].quantile(0.99))
scale_pos = float((y_train == 0).sum() / (y_train == 1).sum())

print(f"p99 Amount threshold: {p99_amount}")
print(f"Scale Pos Weight: {scale_pos}")
print(f"Max train amount for bounds check: {X_train['amount'].max()}")

X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()

X_train['is_high_amount'] = (X_train['amount'] > p99_amount).astype(int)
X_val['is_high_amount'] = (X_val['amount'] > p99_amount).astype(int)
X_test['is_high_amount'] = (X_test['amount'] > p99_amount).astype(int)

# ── 4. Train Model A (Without Errors)
features_A = ['amount', 'log_amount', 'is_high_amount', 'hour', 'is_night', 'balance_diff_orig', 'balance_diff_dest', 'type_enc']
print(f"\n--- Training Model A (No Errors) ---")
print(f"Features: {features_A}")

xgb_A = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    scale_pos_weight=scale_pos,
    tree_method='hist',         
    n_jobs=-1
)

xgb_A.fit(X_train[features_A], y_train)

# Evaluate A on Val
probs_val_A = xgb_A.predict_proba(X_val[features_A])[:, 1]
precisions_A, recalls_A, thresholds_A = precision_recall_curve(y_val, probs_val_A)
valid_idx_A = np.where(recalls_A[:-1] >= 0.90)[0]
best_idx_A = valid_idx_A[np.argmax(precisions_A[valid_idx_A])] if len(valid_idx_A) > 0 else np.argmax(recalls_A[:-1])
thresh_A = thresholds_A[best_idx_A]
pred_val_A = (probs_val_A >= thresh_A).astype(int)
print(f"Model A (Val) - Thresh: {thresh_A:.4f}, Recall: {recall_score(y_val, pred_val_A):.4f}, Precision: {precision_score(y_val, pred_val_A):.4f}")

# ── 5. Train Model B (With Errors)
features_B = ['amount', 'log_amount', 'is_high_amount', 'hour', 'is_night', 'balance_diff_orig', 'balance_diff_dest', 'type_enc', 'sender_error', 'receiver_error']
print(f"\n--- Training Model B (With Errors) ---")
print(f"Features: {features_B}")

xgb_B = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    scale_pos_weight=scale_pos,
    tree_method='hist',         
    n_jobs=-1
)

xgb_B.fit(X_train[features_B], y_train)

# Evaluate B on Val
probs_val_B = xgb_B.predict_proba(X_val[features_B])[:, 1]
precisions_B, recalls_B, thresholds_B = precision_recall_curve(y_val, probs_val_B)
valid_idx_B = np.where(recalls_B[:-1] >= 0.90)[0]
best_idx_B = valid_idx_B[np.argmax(precisions_B[valid_idx_B])] if len(valid_idx_B) > 0 else np.argmax(recalls_B[:-1])
thresh_B = thresholds_B[best_idx_B]
pred_val_B = (probs_val_B >= thresh_B).astype(int)
print(f"Model B (Val) - Thresh: {thresh_B:.4f}, Recall: {recall_score(y_val, pred_val_B):.4f}, Precision: {precision_score(y_val, pred_val_B):.4f}")

# ── 6. Final Evaluation on Untouched Test Set
print(f"\n--- Final Evaluation (Model B on Untouched Test Set) ---")
probs_test = xgb_B.predict_proba(X_test[features_B])[:, 1]
y_pred_test = (probs_test >= thresh_B).astype(int)

acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test)
rec = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

# ── 7. Save the Artifact
print("\nSaving new single artifact to best_fraud_model_v2.pkl...")
artifact = {
    "version": "2.0",
    "model": xgb_B,
    "threshold": float(thresh_B),
    "features": features_B,
    "type_encoder": le,
    "high_amount_threshold": float(p99_amount),
    "night_hours": night_hours,
    "max_train_amount": float(X_train['amount'].max())
}
joblib.dump(artifact, 'best_fraud_model_v2.pkl')

# ── 8. Extract Validation Examples
X_test_examples = X_test.copy()
X_test_examples['actual_label'] = y_test
X_test_examples['fraud_probability'] = probs_test
X_test_examples['predicted_label'] = y_pred_test

fraud_examples = X_test_examples[X_test_examples['actual_label'] == 1].head(5)
safe_examples = X_test_examples[X_test_examples['actual_label'] == 0].head(5)

val_examples = pd.concat([fraud_examples, safe_examples])
# Drop the high dimensionality for display
cols_to_show = ['hour', 'type', 'amount', 'actual_label', 'fraud_probability', 'predicted_label']
val_examples = val_examples[cols_to_show]

print("\n--- 10 Validation Examples ---")
print(val_examples.to_string())
print("Done.")
