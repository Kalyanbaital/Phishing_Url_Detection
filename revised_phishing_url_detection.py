"""
Leakage-controlled phishing URL detection experiments.
Dataset: Hannousse & Yahiouche (Mendeley Data V3), dataset_B_05_2020.csv

Outputs:
  holdout_results.csv
  cross_validation_results.csv (if RUN_FULL_CV=True)
  domain_disjoint_results.csv
  selected_features_*.csv
  confusion matrices and performance figures

Main protocol:
  - 87 engineered numeric features; URL is never used as a classifier feature.
  - 80/20 stratified holdout, random_state=42.
  - Feature selection is fitted only on training data, never on the test set.
  - Optional 5-fold stratified CV for all models.
  - 5-fold StratifiedGroupKFold using an approximate registrable-domain group
    to test generalization to unseen domains.

Install:
  pip install pandas numpy scikit-learn xgboost matplotlib
"""
import os, re, time, argparse, warnings
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             matthews_corrcoef, confusion_matrix)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
FEATURE_COUNTS = [87, 30, 20, 10]

# Common multi-label public suffixes. For a production deployment, a maintained
# Public Suffix List / tldextract is preferable. The grouping is used only for
# the domain-disjoint robustness experiment, not for model input.
TWO_LEVEL_SUFFIXES = {
    'co.uk','org.uk','ac.uk','gov.uk','com.au','net.au','org.au','co.in','firm.in',
    'net.in','org.in','gen.in','ind.in','co.jp','ne.jp','or.jp','co.nz','net.nz',
    'org.nz','com.br','com.cn','com.sg','com.hk','com.mx','com.tr','com.tw','co.za',
    'com.ar','com.pl','co.kr','com.my','com.ua'
}

def get_hostname(url):
    s = str(url)
    if not re.match(r'^[A-Za-z][A-Za-z0-9+.-]*://', s):
        s = 'http://' + s
    try:
        return (urlparse(s).hostname or '').lower().strip('.')
    except Exception:
        return ''

def approximate_registrable_domain(url):
    host = get_hostname(url)
    if not host:
        return ''
    labels = host.split('.')
    if len(labels) <= 2:
        return host
    suffix2 = '.'.join(labels[-2:])
    if suffix2 in TWO_LEVEL_SUFFIXES and len(labels) >= 3:
        return '.'.join(labels[-3:])
    return suffix2

def make_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1500, solver='liblinear', random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9,
                                 eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1),
        'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                             alpha=1e-4, max_iter=250, early_stopping=True,
                             validation_fraction=0.10, n_iter_no_change=12,
                             random_state=RANDOM_STATE)
    }

def make_pipeline(model, k):
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if k < 87:
        steps.append(('selector', SelectKBest(
            lambda X, y: mutual_info_classif(X, y, random_state=RANDOM_STATE, n_jobs=-1), k=k)))
    if isinstance(model, (LogisticRegression, KNeighborsClassifier, SVC, MLPClassifier)):
        steps.append(('scaler', StandardScaler()))
    steps.append(('model', model))
    return Pipeline(steps)

def score_model(estimator, X_test, y_test):
    pred = estimator.predict(X_test)
    score = estimator.predict_proba(X_test)[:, 1] if hasattr(estimator, 'predict_proba') else estimator.decision_function(X_test)
    return {
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1': f1_score(y_test, pred),
        'ROC-AUC': roc_auc_score(y_test, score),
        'PR-AUC': average_precision_score(y_test, score),
        'MCC': matthews_corrcoef(y_test, pred),
    }, pred, score

def main(csv_path, output_dir, run_full_cv=False):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if 'url' not in df.columns or 'status' not in df.columns:
        raise ValueError("CSV must contain 'url' and 'status' columns.")

    feature_cols = [c for c in df.columns if c not in ['url', 'status']]
    if len(feature_cols) != 87:
        raise ValueError(f"Expected 87 engineered features; found {len(feature_cols)}.")

    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    y = df['status'].astype(str).str.lower().map({'legitimate': 0, 'phishing': 1})
    if y.isna().any():
        raise ValueError("Unexpected labels in status. Expected legitimate/phishing.")

    # Dataset audit
    audit = pd.DataFrame({
        'records': [len(df)],
        'features': [len(feature_cols)],
        'legitimate': [(y == 0).sum()],
        'phishing': [(y == 1).sum()],
        'duplicate_rows': [df.duplicated().sum()],
        'duplicate_urls': [df['url'].duplicated().sum()],
        'missing_feature_cells': [X.isna().sum().sum()],
        'infinite_feature_cells_before_replacement': [0],
    })
    audit.to_csv(os.path.join(output_dir, 'dataset_audit.csv'), index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    all_models = make_models()
    holdout_rows = []
    fitted = {}
    confusion = {}

    for k in FEATURE_COUNTS:
        for name, model in all_models.items():
            est = make_pipeline(model, k)
            t0 = time.perf_counter(); est.fit(X_train, y_train); train_time = time.perf_counter() - t0
            t1 = time.perf_counter(); metrics, pred, score = score_model(est, X_test, y_test); infer_time = (time.perf_counter() - t1) / len(X_test)
            row = {'Features': k, 'Model': name, **metrics,
                   'TrainTime_s': train_time, 'Inference_s_per_URL': infer_time}
            holdout_rows.append(row)
            fitted[(k, name)] = est
            confusion[(k, name)] = confusion_matrix(y_test, pred)

    holdout = pd.DataFrame(holdout_rows)
    holdout.to_csv(os.path.join(output_dir, 'holdout_results.csv'), index=False)

    # Save final training-set feature rankings for reproducibility.
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    mi = mutual_info_classif(X_train_imp, y_train, random_state=RANDOM_STATE, n_jobs=-1)
    ranking = pd.DataFrame({'feature': feature_cols, 'mutual_information': mi}).sort_values('mutual_information', ascending=False)
    ranking.to_csv(os.path.join(output_dir, 'feature_ranking_mutual_information.csv'), index=False)
    for k in [30, 20, 10]:
        ranking.head(k).to_csv(os.path.join(output_dir, f'selected_features_{k}.csv'), index=False)

    # Confusion matrices for XGBoost.
    for k in FEATURE_COUNTS:
        cm = confusion[(k, 'XGBoost')]
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        im = ax.imshow(cm)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['Legitimate','Phishing'],
               yticklabels=['Legitimate','Phishing'], xlabel='Predicted', ylabel='Actual',
               title=f'XGBoost confusion matrix ({k} features)')
        for i in range(2):
            for j in range(2): ax.text(j, i, int(cm[i,j]), ha='center', va='center')
        fig.tight_layout(); fig.savefig(os.path.join(output_dir, f'confusion_xgb_{k}.png'), dpi=300); plt.close(fig)

    # Performance vs feature count.
    for metric in ['Accuracy', 'F1']:
        fig, ax = plt.subplots(figsize=(7,4.2))
        for name in all_models:
            sub = holdout[holdout.Model == name].sort_values('Features')
            ax.plot(sub.Features, sub[metric] * 100, marker='o', label=name)
        ax.set_xlabel('Number of selected features'); ax.set_ylabel(f'{metric} (%)')
        ax.set_title(f'{metric} versus feature count'); ax.set_xticks([10,20,30,87]); ax.grid(alpha=.25); ax.legend(fontsize=8, ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(output_dir, f'{metric.lower()}_vs_features.png'), dpi=300); plt.close(fig)

    # Top feature plot.
    top = ranking.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7,5)); ax.barh(top.feature, top.mutual_information)
    ax.set_xlabel('Mutual information'); ax.set_title('Top 15 features ranked on training data')
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, 'top15_mutual_information.png'), dpi=300); plt.close(fig)

    # Domain-disjoint robustness evaluation. The URL is used ONLY to create groups.
    groups = df['url'].map(approximate_registrable_domain)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    domain_rows = []
    for name in ['Random Forest', 'XGBoost', 'SVM']:
        vals = {m: [] for m in ['Accuracy','Precision','Recall','F1','ROC-AUC','PR-AUC','MCC']}
        for tr, te in sgkf.split(X, y, groups=groups):
            est = make_pipeline(all_models[name], 87)
            est.fit(X.iloc[tr], y.iloc[tr])
            metrics, _, _ = score_model(est, X.iloc[te], y.iloc[te])
            for m,v in metrics.items(): vals[m].append(v)
        row = {'Model': name}
        for m,v in vals.items(): row[m+'_mean'] = np.mean(v); row[m+'_sd'] = np.std(v)
        domain_rows.append(row)
    domain_df = pd.DataFrame(domain_rows)
    domain_df.to_csv(os.path.join(output_dir, 'domain_disjoint_results.csv'), index=False)

    if run_full_cv:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_rows = []
        for k in FEATURE_COUNTS:
            for name, model in all_models.items():
                vals = {m: [] for m in ['Accuracy','Precision','Recall','F1','ROC-AUC','PR-AUC','MCC']}
                for tr, te in cv.split(X_train, y_train):
                    est = make_pipeline(model, k)
                    est.fit(X_train.iloc[tr], y_train.iloc[tr])
                    metrics, _, _ = score_model(est, X_train.iloc[te], y_train.iloc[te])
                    for m,v in metrics.items(): vals[m].append(v)
                row = {'Features': k, 'Model': name}
                for m,v in vals.items(): row[m+'_mean'] = np.mean(v); row[m+'_sd'] = np.std(v)
                cv_rows.append(row)
        pd.DataFrame(cv_rows).to_csv(os.path.join(output_dir, 'cross_validation_results.csv'), index=False)

    # Save a compact summary.
    best = holdout.sort_values('F1', ascending=False).head(10)
    best.to_csv(os.path.join(output_dir, 'best_holdout_models.csv'), index=False)
    print('Completed. Results written to:', output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='dataset_B_05_2020.csv')
    parser.add_argument('--output', default='results')
    parser.add_argument('--full-cv', action='store_true', help='Run 5-fold CV for every model and feature count; can be time-consuming.')
    args = parser.parse_args()
    main(args.csv, args.output, args.full_cv)
