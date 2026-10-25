# Feature-Efficient Phishing URL Detection: Feature Reduction and Domain-Aware Generalization for E-Health Security

This repository contains the reproducible source code, experimental results, and figures associated with the research paper **“Feature-Efficient Phishing URL Detection: Feature Reduction and Domain-Aware Generalization for E-Health Security.”**

The study evaluates machine-learning approaches for phishing URL detection using the Hannousse–Yahiouche *Web page phishing detection* dataset, Version 3, comprising 11,430 URLs and 87 engineered features.

## Repository Contents

### 1. `revised_phishing_url_detection.py`

Python implementation of the experimental framework, including:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest (RF)
- XGBoost
- Multilayer Perceptron (MLP)
- Experiments using 87, 30, 20, and 10 features
- Mutual-information-based feature selection within the machine-learning pipeline
- Standard 80:20 stratified holdout evaluation
- Five-fold approximate domain-group-disjoint evaluation
- Accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and Matthews Correlation Coefficient (MCC)
- Training-time and inference-time measurement
- Feature ranking, dataset auditing, and figure generation

The original URL string is excluded from the predictive feature set. It is used only to derive approximate domain groups for the domain-group-disjoint evaluation.

### 2. Experimental result files

| File | Description |
|---|---|
| `holdout_results.csv` | Results for all six classifiers across the 87-, 30-, 20-, and 10-feature configurations using the standard holdout protocol. |
| `domain_disjoint_results.csv` | Five-fold approximate domain-group-disjoint results for Random Forest, XGBoost, and SVM using all 87 features. |
| `dataset_audit.csv` | Dataset summary, class counts, duplicate counts, missing feature cells, and infinite feature values detected before replacement. |
| `feature_ranking_mutual_information.csv` | Mutual-information ranking of the 87 features calculated using the holdout training partition. |
| `selected_features_30.csv` | Top 30 features in the training-partition mutual-information ranking. |
| `selected_features_20.csv` | Top 20 features in the training-partition mutual-information ranking. |
| `selected_features_10.csv` | Top 10 features in the training-partition mutual-information ranking. |
| `best_holdout_models.csv` | Ten highest-ranked holdout model/configuration results by F1-score, with accuracy used as a secondary sorting criterion. |
| `cross_validation_results.csv` | Additional five-fold stratified cross-validation results, generated only when `--full-cv` is specified. |

### 3. Figures

The script generates the following figures using the corresponding filenames:

| File | Description |
|---|---|
| `fig1_accuracy_features.png` | Accuracy versus the number of features. |
| `fig2_f1_features.png` | F1-score versus the number of features. |
| `fig3_confusion_xgb87.png` | Confusion matrix for XGBoost using all 87 features. |
| `fig4_domain_disjoint.png` | Mean F1-score comparison under approximate domain-group-disjoint five-fold evaluation, with fold standard deviations. |
| `fig5_top_features_mi.png` | Top 15 features ranked by mutual information using the training partition. |

The generated output files are saved in the directory specified by the `--output` argument. The files committed to the repository should be regenerated and checked against the manuscript whenever the experimental code or results are updated.

## Dataset

The experiments use the *Web page phishing detection* dataset developed by Hannousse and Yahiouche.

The dataset contains:

- **11,430 URLs**
- **87 engineered features**
- **5,715 legitimate URLs**
- **5,715 phishing URLs**
- **Balanced class distribution (50:50)**

Obtain Version 3 from the dataset authors' official Mendeley Data record. The dataset is not included in this repository. Please cite the dataset and its associated publication according to the original authors' recommended citation details.

After obtaining the dataset, place the CSV file in the repository directory or specify its location with the `--csv` argument. The default filename expected by the script is `dataset_B_05_2020.csv`.

## Methodology and Reproducibility

### Holdout evaluation

The principal holdout experiment uses an 80:20 stratified train-test split with `random_state=42`. All six classifiers are evaluated using 87, 30, 20, and 10 features.

The original URL is excluded from the model inputs. The target label is encoded as:

- `legitimate` = 0
- `phishing` = 1

Therefore, **phishing is the positive class** for the reported precision, recall, F1-score, and related classification metrics.

### Preprocessing and feature selection

The preprocessing pipeline converts feature values to numeric form, treats invalid numeric entries and positive or negative infinity as missing values, and applies median imputation. Imputation is fitted within each model's training pipeline.

For the 30-, 20-, and 10-feature experiments, mutual-information-based feature selection is also performed within the pipeline. This ensures that feature selection is fitted on the relevant training partition rather than on the held-out test data. The 87-feature configuration uses all engineered features without feature selection.

Standard scaling is applied to Logistic Regression, KNN, SVM, and MLP. Tree-based models are not standardized.

### Evaluation metrics and timing

The script calculates accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and MCC. In this implementation, **PR-AUC is calculated using scikit-learn's `average_precision_score`**, rather than trapezoidal integration of a precision-recall curve.

Training time measures pipeline fitting, including the preprocessing and feature selection applicable to each configuration. Inference time is recorded as the elapsed evaluation time divided by the number of test URLs; it should be interpreted as an empirical per-URL timing measurement in the execution environment, not as a hardware-independent latency guarantee.

### Approximate domain-group-disjoint evaluation

The study also conducts a five-fold `StratifiedGroupKFold` evaluation for Random Forest, XGBoost, and SVM using all 87 engineered features.

Approximate domain groups are derived from URL hostnames using a predefined suffix heuristic. URLs assigned to the same group are kept in the same fold, so a group is not deliberately split between training and testing within a fold.

The heuristic does **not** use a maintained Public Suffix List and therefore cannot guarantee correct identification of registrable domains for every hostname. Results should consequently be interpreted as approximate domain-group-disjoint evaluation, not as a guarantee of complete registrable-domain separation.

This experiment complements the standard holdout evaluation by assessing performance under a different partitioning protocol. Because the holdout and grouped five-fold protocols differ in partitioning and aggregation, their performance difference is descriptive and should not be interpreted as an isolated causal estimate of the effect of domain overlap.

## Main Findings

In the reported standard holdout experiment, XGBoost using all 87 features achieves approximately **96.63% accuracy and 96.65% F1-score**.

With 30 features, XGBoost achieves approximately **96.17% F1-score**, a reduction of 0.48 percentage points compared with the 87-feature configuration. This result indicates that the feature representation can be reduced substantially while retaining performance close to that of the full feature set.

The domain-group-disjoint evaluation provides a complementary assessment of model generalization across approximate domain groups. Its results should be interpreted separately from the standard holdout results because the evaluation protocols differ.

These values describe the results reported in the manuscript; they should be reconfirmed against the CSV files generated by the current code before publication or reuse.

## Reproducing the Experiments

### Requirements

Python 3.11 is recommended. Install the required packages from the repository's `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The script uses Python packages including pandas, NumPy, scikit-learn, XGBoost, and Matplotlib. Ensure that `requirements.txt` is present in the repository and reflects the package versions used for the experiments.

### Obtain the dataset

Download Version 3 of the Hannousse–Yahiouche *Web page phishing detection* dataset from its official Mendeley Data record. Place the CSV in the repository directory or note its full path.

### Run the principal experiments

Execute the standard holdout and approximate domain-group-disjoint experiments with:

```bash
python revised_phishing_url_detection.py --csv dataset_B_05_2020.csv --output results
```

Replace `dataset_B_05_2020.csv` with the actual dataset path if necessary.

The output directory contains the holdout results, domain-group-disjoint results, dataset audit, feature rankings, selected-feature lists, best holdout model summary, and Figures 1–5.

### Run optional cross-validation

An additional five-fold stratified cross-validation analysis can be run on the 80% training partition:

```bash
python revised_phishing_url_detection.py --csv dataset_B_05_2020.csv --output results --full-cv
```

This optional analysis evaluates all six classifiers and all four feature configurations. It generates `cross_validation_results.csv`.

**Note:** This optional cross-validation is separate from both the standard holdout evaluation and the principal approximate domain-group-disjoint experiment. It is additional analysis and should not be confused with the five-fold grouped evaluation reported in the manuscript.

## Reported Results and Citation

The manuscript's standard holdout results are represented by `holdout_results.csv`, and the approximate domain-group-disjoint results are represented by `domain_disjoint_results.csv`. Figures 1–5 correspond to the figure files listed above.

When using this repository or reproducing the experiments, please cite the associated research paper and the original dataset publication. Dataset attribution should follow the citation information provided by the dataset authors.
