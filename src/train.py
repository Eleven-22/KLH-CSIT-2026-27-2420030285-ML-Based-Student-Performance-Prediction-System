import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tabpfn import TabPFNClassifier

from src.components.data_transformation import DataTransformation
from src.utils import save_object

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "student_data.csv")
ARTIFACTS = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS, "model.pkl")
META_PATH = os.path.join(ARTIFACTS, "best_tabpfn_metadata.pkl")
RESULTS_PATH = os.path.join(ARTIFACTS, "model_results.csv")

def load_xy():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["student_id"], errors="ignore")
    if "performance" not in df.columns:
        raise ValueError("student_data.csv must contain the 'performance' target column.")
    X = DataTransformation.transform_dataframe(df)
    # DataTransformation uses 1 = At Risk, 0 = Good Performance.
    y = (1 - df["performance"].astype(int)).to_numpy()
    return X, y

def cv_config(X, y, n_estimators):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_prob = np.zeros(len(y), dtype=float)

    for tr, va in skf.split(X, y):
        model = TabPFNClassifier(
            device="cpu",
            n_estimators=n_estimators,
            random_state=RANDOM_STATE
        )
        model.fit(X.iloc[tr], y[tr])
        oof_prob[va] = model.predict_proba(X.iloc[va])[:, 1]

    # Choose a threshold that gives the strongest F1 for At Risk.
    best = None
    for threshold in np.arange(0.20, 0.81, 0.01):
        pred = (oof_prob >= threshold).astype(int)
        f1 = f1_score(y, pred, pos_label=1, zero_division=0)
        acc = accuracy_score(y, pred)
        recall = recall_score(y, pred, pos_label=1, zero_division=0)
        # Primary objective: F1, secondary: accuracy, with recall encouraged.
        score = 0.60 * f1 + 0.25 * acc + 0.15 * recall
        candidate = (score, f1, acc, recall, threshold)
        if best is None or candidate > best:
            best = candidate

    return best

def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    print("Loading ALT student dataset...")
    X, y = load_xy()
    print(f"Dataset: {len(y)} rows, {X.shape[1]} engineered features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    print("\nSelecting the BEST TabPFN configuration using 5-fold CV...")
    configs = [4, 8, 16]
    rows = []
    best = None

    for n_estimators in configs:
        score, f1, acc, recall, threshold = cv_config(X_train, y_train, n_estimators)
        row = {
            "model": "TabPFN",
            "n_estimators": n_estimators,
            "cv_selection_score": score,
            "cv_f1_at_risk": f1,
            "cv_accuracy": acc,
            "cv_recall_at_risk": recall,
            "threshold": threshold,
        }
        rows.append(row)
        print(row)
        if best is None or (score, f1, acc) > (
            best["cv_selection_score"], best["cv_f1_at_risk"], best["cv_accuracy"]
        ):
            best = row

    print("\n===== BEST TABPFN =====")
    print(best)

    final_model = TabPFNClassifier(
        device="cpu",
        n_estimators=int(best["n_estimators"]),
        random_state=RANDOM_STATE
    )
    final_model.fit(X_train, y_train)

    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= float(best["threshold"])).astype(int)

    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, pos_label=1, zero_division=0)
    test_precision = precision_score(y_test, test_pred, pos_label=1, zero_division=0)
    test_recall = recall_score(y_test, test_pred, pos_label=1, zero_division=0)

    print("\n===== FINAL HOLD-OUT TEST =====")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall   : {test_recall:.4f}")
    print(f"F1-score : {test_f1:.4f}")

    save_object(MODEL_PATH, final_model)
    save_object(META_PATH, {
        "model": "TabPFN",
        "n_estimators": int(best["n_estimators"]),
        "threshold": float(best["threshold"]),
        "target": "1=At Risk (G3<10), 0=Good Performance (G3>=10)",
        "test_accuracy": float(test_acc),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
    })
    pd.DataFrame(rows).sort_values("cv_selection_score", ascending=False).to_csv(
        RESULTS_PATH, index=False
    )

    print("\nSaved:")
    print(MODEL_PATH)
    print(META_PATH)
    print(RESULTS_PATH)

if __name__ == "__main__":
    main()
