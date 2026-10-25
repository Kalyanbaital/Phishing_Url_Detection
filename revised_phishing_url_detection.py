"""
Feature-Efficient Phishing URL Detection: Feature Reduction and
Domain-Aware Generalization for E-Health Security

Reproducible experiments for the associated manuscript.

Dataset: Hannousse–Yahiouche Web page phishing detection, Version 3.
Expected CSV columns: url, status, and 87 engineered numeric features.

Protocol:
- 80:20 stratified holdout, random_state=42.
- Labels: legitimate=0, phishing=1 (phishing is the positive class).
- Median imputation is fitted inside each pipeline.
- Numeric coercion failures and infinities are treated as missing values.
- Mutual-information selection is fitted inside the pipeline for 30, 20,
  and 10 features; the 87-feature setting uses all features.
- StandardScaler is used for LR, KNN, SVM, and MLP, not tree models.
- PR-AUC is average_precision_score (average precision).
- Domain-group experiment: five-fold StratifiedGroupKFold for RF, XGBoost,
  and SVM, using all 87 features and approximate hostname-derived groups.
  The grouping heuristic is not a maintained Public Suffix List.
- Optional --full-cv runs five-fold StratifiedKFold on the 80% training
  partition for all models and feature counts. This is additional analysis,
  separate from the manuscript's domain-group experiment.

Install:
    pip install pandas numpy scikit-learn xgboost matplotlib

Examples:
    python revised_phishing_url_detection.py --csv dataset_B_05_2020.csv --output results
    python revised_phishing_url_detection.py --csv dataset_B_05_2020.csv --output results --full-cv
"""
import argparse
import os
import re
import time
import warnings
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
EXPECTED_FEATURE_COUNT = 87
FEATURE_COUNTS = [87, 30, 20, 10]
POSITIVE_LABEL = 1

# Approximate list of common multi-label suffixes. This is not a full PSL.
TWO_LEVEL_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au",
    "co.in", "firm.in", "net.in", "org.in", "gen.in", "ind.in", "co.jp",
    "ne.jp", "or.jp", "co.nz", "net.nz", "org.nz", "com.br", "com.cn",
    "com.sg", "com.hk", "com.mx", "com.tr", "com.tw", "co.za", "com.ar",
    "com.pl", "co.kr", "com.my", "com.ua",
}


def get_hostname(url):
    """Return a normalized URL hostname, or an empty string if unavailable."""
    value = str(url).strip()
    if not value:
        return ""
    if not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
        value = "http://" + value
    try:
        return (urlparse(value).hostname or "").lower().strip(".")
    except (TypeError, ValueError):
        return ""


def approximate_registrable_domain(url):
    """
    Approximate domain-group rule used only for grouped evaluation:
    keep one/two-label hosts as-is; otherwise retain the final two labels,
    except when those labels are in TWO_LEVEL_SUFFIXES, in which case retain
    the final three labels. This is not guaranteed PSL-based registration.
    """
    host = get_hostname(url)
    if not host:
        return ""
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    last_two = ".".join(labels[-2:])
    if last_two in TWO_LEVEL_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last_two


def make_models():
    """Create the six classifiers used in the manuscript."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, solver="liblinear", random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=False,
                   random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=1e-4, max_iter=250, early_stopping=True,
            validation_fraction=0.10, n_iter_no_change=12,
            random_state=RANDOM_STATE,
        ),
    }


def make_pipeline(model, k):
    """Build a pipeline that fits preprocessing only on the training fold."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if k < EXPECTED_FEATURE_COUNT:
        steps.append(("selector", SelectKBest(
            score_func=lambda X, y: mutual_info_classif(
                X, y, random_state=RANDOM_STATE, n_jobs=-1
            ),
            k=k,
        )))
    if isinstance(model, (LogisticRegression, KNeighborsClassifier, SVC, MLPClassifier)):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", clone(model)))
    return Pipeline(steps)


def score_model(estimator, X_test, y_test):
    """Calculate metrics; phishing (label 1) is the positive class."""
    pred = estimator.predict(X_test)
    scores = (estimator.predict_proba(X_test)[:, 1]
              if hasattr(estimator, "predict_proba")
              else estimator.decision_function(X_test))
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "Recall": recall_score(y_test, pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "F1": f1_score(y_test, pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, scores),
        "PR-AUC": average_precision_score(y_test, scores, pos_label=POSITIVE_LABEL),
        "MCC": matthews_corrcoef(y_test, pred),
    }
    return metrics, pred, scores


def save_confusion_matrix(cm, path, title):
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        xlabel="Predicted label", ylabel="Actual label", title=title,
    )
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_holdout_figures(holdout, cm_xgb_87, ranking, output_dir):
    """Save manuscript figures 1, 2, 3, and 5 using the agreed filenames."""
    for metric, filename, ylabel, title in [
        ("Accuracy", "fig1_accuracy_features.png", "Accuracy (%)",
         "Accuracy versus feature count"),
        ("F1", "fig2_f1_features.png", "F1-score (%)",
         "F1-score versus feature count"),
    ]:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        for model_name in holdout["Model"].drop_duplicates():
            sub = holdout[holdout["Model"] == model_name].sort_values("Features")
            ax.plot(sub["Features"], sub[metric] * 100, marker="o",
                    linewidth=1.5, label=model_name)
        ax.set_xlabel("Number of features")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks([10, 20, 30, 87])
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close(fig)

    save_confusion_matrix(
        cm_xgb_87, os.path.join(output_dir, "fig3_confusion_xgb87.png"),
        "XGBoost confusion matrix (87 features)"
    )

    top = ranking.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh(top["feature"], top["mutual_information"])
    ax.set_xlabel("Mutual information")
    ax.set_title("Top 15 features ranked on training data")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig5_top_features_mi.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_domain_disjoint_evaluation(X, y, urls, models, output_dir):
    """Run five-fold approximate domain-group-disjoint evaluation."""
    groups = urls.map(approximate_registrable_domain)
    # Do not force unrelated missing/invalid URLs into a single shared group.
    missing = groups.eq("")
    if missing.any():
        groups = groups.copy()
        groups.loc[missing] = [
            f"__invalid_url_row_{idx}" for idx in groups.index[missing]
        ]

    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    rows = []
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "MCC"]

    for model_name in ["Random Forest", "XGBoost", "SVM"]:
        fold_values = {metric: [] for metric in metric_names}
        for train_idx, test_idx in splitter.split(X, y, groups=groups):
            estimator = make_pipeline(models[model_name], EXPECTED_FEATURE_COUNT)
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
            metrics, _, _ = score_model(estimator, X.iloc[test_idx], y.iloc[test_idx])
            for metric, value in metrics.items():
                fold_values[metric].append(value)

        row = {"Model": model_name}
        # Lowercase snake_case matches the manuscript's committed results CSV.
        for metric, values in fold_values.items():
            key = metric.lower().replace("-", "_")
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_sd"] = float(np.std(values, ddof=0))
        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(output_dir, "domain_disjoint_results.csv"), index=False)

    # Figure 4: mean F1-score and fold standard deviation.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(results["Model"], results["f1_mean"] * 100,
           yerr=results["f1_sd"] * 100, capsize=4)
    ax.set_ylabel("Mean F1-score (%)")
    ax.set_title("Approximate domain-group-disjoint five-fold evaluation")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_domain_disjoint.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    return results


def run_optional_full_cv(X_train, y_train, models, output_dir):
    """Optional five-fold CV on the 80% training partition; not domain grouped."""
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "MCC"]
    for k in FEATURE_COUNTS:
        for model_name, model in models.items():
            fold_values = {metric: [] for metric in metric_names}
            for train_idx, test_idx in splitter.split(X_train, y_train):
                estimator = make_pipeline(model, k)
                estimator.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                metrics, _, _ = score_model(
                    estimator, X_train.iloc[test_idx], y_train.iloc[test_idx]
                )
                for metric, value in metrics.items():
                    fold_values[metric].append(value)
            row = {"Features": k, "Model": model_name}
            for metric, values in fold_values.items():
                key = metric.lower().replace("-", "_")
                row[f"{key}_mean"] = float(np.mean(values))
                row[f"{key}_sd"] = float(np.std(values, ddof=0))
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "cross_validation_results.csv"), index=False
    )


def main(csv_path, output_dir, run_full_cv=False):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Download the Hannousse–Yahiouche "
            "Version 3 dataset from Mendeley Data and provide its CSV path with --csv."
        )

    df = pd.read_csv(csv_path)
    missing_cols = {"url", "status"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV must contain 'url' and 'status'; missing: {sorted(missing_cols)}")

    feature_cols = [c for c in df.columns if c not in ["url", "status"]]
    if len(feature_cols) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} engineered features after excluding "
            f"'url' and 'status'; found {len(feature_cols)}."
        )

    # Count infinities before replacement. Invalid numeric strings become NaN.
    X_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    infinite_count = int(np.isinf(X_raw.to_numpy(dtype=float, na_value=np.nan)).sum())
    X = X_raw.replace([np.inf, -np.inf], np.nan)

    y = df["status"].astype(str).str.strip().str.lower().map(
        {"legitimate": 0, "phishing": 1}
    )
    if y.isna().any():
        unexpected = sorted(df.loc[y.isna(), "status"].astype(str).unique().tolist())
        raise ValueError(
            "Expected only 'legitimate' and 'phishing' labels; unexpected values: "
            f"{unexpected}"
        )
    y = y.astype(int)
    if y.nunique() != 2:
        raise ValueError("Both legitimate and phishing examples are required.")
    all_missing = X.columns[X.isna().all(axis=0)].tolist()
    if all_missing:
        raise ValueError(f"Feature columns with no usable numeric values: {all_missing}")

    audit = pd.DataFrame({
        "records": [len(df)],
        "features": [len(feature_cols)],
        "legitimate": [int((y == 0).sum())],
        "phishing": [int((y == 1).sum())],
        "duplicate_rows": [int(df.duplicated().sum())],
        "duplicate_urls": [int(df["url"].duplicated().sum())],
        "missing_feature_cells": [int(X.isna().sum().sum())],
        "infinite_feature_cells_before_replacement": [infinite_count],
    })
    audit.to_csv(os.path.join(output_dir, "dataset_audit.csv"), index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    models = make_models()
    holdout_rows = []
    cm_xgb_87 = None

    for k in FEATURE_COUNTS:
        for model_name, model in models.items():
            estimator = make_pipeline(model, k)
            start = time.perf_counter()
            estimator.fit(X_train, y_train)
            train_seconds = time.perf_counter() - start

            start = time.perf_counter()
            metrics, pred, _ = score_model(estimator, X_test, y_test)
            inference_seconds = (time.perf_counter() - start) / len(X_test)

            holdout_rows.append({
                "Features": k, "Model": model_name, **metrics,
                "TrainTime_s": train_seconds,
                "Inference_s_per_URL": inference_seconds,
            })
            if model_name == "XGBoost" and k == EXPECTED_FEATURE_COUNT:
                cm_xgb_87 = confusion_matrix(y_test, pred, labels=[0, 1])

    holdout = pd.DataFrame(holdout_rows)
    holdout.to_csv(os.path.join(output_dir, "holdout_results.csv"), index=False)

    # Feature ranking is based only on the holdout training partition.
    ranking_imputer = SimpleImputer(strategy="median")
    X_train_imputed = ranking_imputer.fit_transform(X_train)
    mi_scores = mutual_info_classif(
        X_train_imputed, y_train, random_state=RANDOM_STATE, n_jobs=-1
    )
    ranking = pd.DataFrame({
        "feature": feature_cols, "mutual_information": mi_scores
    }).sort_values("mutual_information", ascending=False, kind="mergesort").reset_index(drop=True)
    ranking.to_csv(
        os.path.join(output_dir, "feature_ranking_mutual_information.csv"), index=False
    )
    for k in [30, 20, 10]:
        ranking.head(k).to_csv(
            os.path.join(output_dir, f"selected_features_{k}.csv"), index=False
        )

    if cm_xgb_87 is None:
        raise RuntimeError("Could not generate the XGBoost 87-feature confusion matrix.")
    save_holdout_figures(holdout, cm_xgb_87, ranking, output_dir)

    domain_results = run_domain_disjoint_evaluation(
        X, y, df["url"].astype(str), models, output_dir
    )
    holdout.sort_values(["F1", "Accuracy"], ascending=[False, False]).head(10).to_csv(
        os.path.join(output_dir, "best_holdout_models.csv"), index=False
    )

    if run_full_cv:
        run_optional_full_cv(X_train, y_train, models, output_dir)

    print(f"Completed. Results written to: {os.path.abspath(output_dir)}")
    print(f"Holdout result rows: {len(holdout)}")
    print(f"Domain-disjoint result rows: {len(domain_results)}")
    print("Positive class: phishing (label 1). PR-AUC: average_precision_score.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce phishing URL holdout, feature-reduction, and domain-group experiments."
    )
    parser.add_argument("--csv", default="dataset_B_05_2020.csv",
                        help="Dataset CSV path (default: dataset_B_05_2020.csv).")
    parser.add_argument("--output", default="results",
                        help="Output directory (default: results).")
    parser.add_argument("--full-cv", action="store_true",
                        help="Also run five-fold StratifiedKFold CV on the 80% training partition.")
    args = parser.parse_args()
    main(args.csv, args.output, args.full_cv)
