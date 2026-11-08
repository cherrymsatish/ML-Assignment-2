import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Bank Marketing Model Evaluation", page_icon="📊", layout="wide")
st.title("Bank Marketing – Classification Model Evaluation")
st.write("Upload a held-out test CSV containing the input features and target column `y`.")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TARGET = "y"
POSITIVE = "yes"
NEGATIVE = "no"

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

missing = [f for f in model_files.values() if not (MODEL_DIR / f).exists()]
if missing:
    st.error("Missing saved models: " + ", ".join(missing) + ". Run the training notebook first.")
    st.stop()

uploaded = st.file_uploader("Upload test CSV", type=["csv"])
model_name = st.selectbox("Select a model", list(model_files))

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()

    st.subheader("Dataset preview")
    st.dataframe(data.head(10), use_container_width=True)

    if TARGET not in data.columns:
        st.error("The uploaded CSV must include the target column 'y'.")
        st.stop()

    y_test = data[TARGET].astype(str).str.strip().str.lower()
    X_test = data.drop(columns=[TARGET])

    if not set(y_test.unique()).issubset({NEGATIVE, POSITIVE}):
        st.error("Target column y must contain only 'yes' and 'no'.")
        st.stop()

    model = joblib.load(MODEL_DIR / model_files[model_name])
    preprocessor = model.named_steps["preprocessing"]
    expected_columns = list(preprocessor.feature_names_in_)

    missing_columns = [c for c in expected_columns if c not in X_test.columns]
    if missing_columns:
        st.error("Missing required features: " + ", ".join(missing_columns))
        st.stop()

    extra_columns = [c for c in X_test.columns if c not in expected_columns]
    if extra_columns:
        st.warning("Extra columns will be ignored: " + ", ".join(extra_columns))

    X_test = X_test[expected_columns]

    try:
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        positive_index = list(model.classes_).index(POSITIVE)
        y_binary = (y_test == POSITIVE).astype(int)

        scores = {
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": roc_auc_score(y_binary, probabilities[:, positive_index]),
            "Precision": precision_score(y_test, predictions, pos_label=POSITIVE, zero_division=0),
            "Recall": recall_score(y_test, predictions, pos_label=POSITIVE, zero_division=0),
            "F1 Score": f1_score(y_test, predictions, pos_label=POSITIVE, zero_division=0),
            "MCC": matthews_corrcoef(y_test, predictions),
        }

        st.subheader(f"Evaluation – {model_name}")
        metric_cols = st.columns(3)
        for i, (name, value) in enumerate(scores.items()):
            metric_cols[i % 3].metric(name, f"{value:.4f}")

        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_test, predictions, labels=[NEGATIVE, POSITIVE])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted No", "Predicted Yes"],
            yticklabels=["Actual No", "Actual Yes"], ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Classification report")
        report = classification_report(
            y_test, predictions, labels=[NEGATIVE, POSITIVE],
            zero_division=0, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

        output = X_test.copy()
        output["Actual"] = y_test.to_numpy()
        output["Predicted"] = predictions
        output["Probability_Yes"] = probabilities[:, positive_index]
        st.subheader("Predictions")
        st.dataframe(output.head(100), use_container_width=True)
        st.download_button(
            "Download predictions CSV",
            output.to_csv(index=False).encode("utf-8"),
            file_name="model_predictions.csv",
            mime="text/csv"
        )
    except Exception as exc:
        st.exception(exc)
else:
    st.info("Upload the generated test_data.csv file to evaluate a saved model.")
