import json
from datetime import datetime
from langchain_core.tools import tool


@tool
def get_current_datetime() -> str:
    """Returns the current date and time.
    Use this when the user asks about today's date, current time, or day of the week.
    """
    now = datetime.now()
    return json.dumps({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "iso_format": now.isoformat(),
    })


@tool
def explain_ml_metric(metric_name: str) -> str:
    """Returns a concise Data Science explanation for a common ML evaluation metric.
    Use this when the user asks to explain, define, or compare ML metrics such as
    accuracy, precision, recall, F1, AUC-ROC, RMSE, MAE, R2, MSE, etc.

    Args:
        metric_name: The name of the ML metric to explain (e.g. 'F1', 'AUC-ROC').
    """
    metrics = {
        "accuracy": "Accuracy = (TP + TN) / Total. Fraction of correct predictions. Misleading on imbalanced datasets — a model predicting all-majority-class can score 99% accuracy with zero real utility.",
        "precision": "Precision = TP / (TP + FP). Of all predicted positives, how many are actually positive. Use when false positives are costly (e.g., spam detection).",
        "recall": "Recall (Sensitivity) = TP / (TP + FN). Of all actual positives, how many did we catch. Use when false negatives are costly (e.g., cancer screening).",
        "f1": "F1 = 2 × (Precision × Recall) / (Precision + Recall). Harmonic mean that balances precision and recall. Best for imbalanced datasets.",
        "f1 score": "F1 = 2 × (Precision × Recall) / (Precision + Recall). Harmonic mean that balances precision and recall. Best for imbalanced datasets.",
        "f1-score": "F1 = 2 × (Precision × Recall) / (Precision + Recall). Harmonic mean that balances precision and recall. Best for imbalanced datasets.",
        "auc-roc": "AUC-ROC measures classifier performance across ALL classification thresholds. AUC=1.0 is perfect; AUC=0.5 is random guessing. Robust to class imbalance.",
        "auc": "AUC-ROC measures classifier performance across ALL classification thresholds. AUC=1.0 is perfect; AUC=0.5 is random guessing. Robust to class imbalance.",
        "roc": "ROC curve plots True Positive Rate vs False Positive Rate at various thresholds. AUC (area under curve) summarises the full curve into one number.",
        "rmse": "RMSE = √mean((ŷ − y)²). Root Mean Squared Error. Same units as target variable. Penalises large errors heavily. Use for regression.",
        "mae": "MAE = mean(|ŷ − y|). Mean Absolute Error. Average absolute deviation. More robust to outliers than RMSE. Easier to interpret.",
        "mse": "MSE = mean((ŷ − y)²). Mean Squared Error. Heavily penalises large errors. Units are squared — harder to interpret directly; use RMSE instead.",
        "r2": "R² = 1 − SS_res/SS_tot. Proportion of variance in target explained by the model. R²=1 is perfect; R²=0 means the model is no better than the mean baseline; R²<0 is worse than baseline.",
        "r-squared": "R² = 1 − SS_res/SS_tot. Proportion of variance in target explained by the model. R²=1 is perfect; R²=0 means model equals mean baseline.",
        "log loss": "Log Loss (Cross-Entropy) = −mean(y·log(p) + (1−y)·log(1−p)). Penalises confident wrong predictions heavily. Standard metric for probabilistic classifiers.",
        "logloss": "Log Loss (Cross-Entropy) = −mean(y·log(p) + (1−y)·log(1−p)). Penalises confident wrong predictions heavily. Standard metric for probabilistic classifiers.",
    }
    key = metric_name.lower().strip()
    explanation = metrics.get(key)
    if explanation:
        return explanation
    return (
        f"'{metric_name}' not in built-in database. "
        "Supported metrics: accuracy, precision, recall, F1, AUC-ROC, RMSE, MAE, MSE, R2, Log Loss."
    )


TOOLS = [get_current_datetime, explain_ml_metric]
