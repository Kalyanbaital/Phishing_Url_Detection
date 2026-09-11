This repository contains the reproducible source code, experimental results and figures associated with the research paper:

“Trustworthy and Resource-Efficient Machine Learning for Phishing URL Detection: Feature Reduction and Generalization Analysis”

The study evaluates machine-learning approaches for phishing URL detection using the Hannousse–Yahiouche Web page phishing detection dataset, containing 11,430 URLs and 87 engineered features.

Repository Contents:

1. revised_phishing_url_detection.py

Complete Python implementation of the experimental framework.

The code implements:

Logistic Regression
K-Nearest Neighbors
Support Vector Machine
Random Forest
XGBoost
Multilayer Perceptron
87-, 30-, 20- and 10-feature experiments
Leakage-controlled mutual-information feature selection
Standard 80:20 stratified holdout evaluation
Domain-disjoint five-fold evaluation
Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC and MCC
Training and inference-time measurement
Feature analysis and visualization

2. holdout_results.csv

Contains the numerical results from the standard 80:20 stratified holdout experiments for all evaluated classifiers and feature configurations.

3. domain_disjoint_results.csv

Contains the results from the five-fold domain-disjoint generalization experiment. URLs belonging to the same approximate registrable domain are kept within the same fold to evaluate performance on previously unseen domain groups.

4. fig1_accuracy_features.png

Accuracy versus the number of selected features.

5. fig2_f1_features.png

F1-score versus the number of selected features.

6. fig3_confusion_xgb87.png

Confusion matrix of the XGBoost model using all 87 features.

7. fig4_top_features_mi.png

Top features identified using mutual-information analysis on the training data.

8. fig5_domain_disjoint.png

Performance comparison under domain-disjoint five-fold evaluation.

Dataset

The experiments use the Web page phishing detection dataset developed by Hannousse and Yahiouche.

The dataset contains:

11,430 URLs
87 engineered features
5,715 legitimate URLs
5,715 phishing URLs
50:50 class distribution

The original dataset should be cited according to the dataset authors' publication and Mendeley Data record.

Reproducibility

The experimental pipeline is designed to avoid test-set information leakage. Feature selection is performed within the training pipeline rather than before train-test splitting.

The study additionally evaluates domain-level generalization because conventional random splits may produce optimistic results when related URLs or domains occur in both training and testing data.

Main Finding

XGBoost provides the best performance among the evaluated models on the standard holdout experiment, achieving approximately 96.63% accuracy and 96.65% F1-score using all 87 features.

Importantly, the model retains approximately 96.17% F1-score using only 30 features, demonstrating that a substantial reduction in feature complexity is possible with only a small loss in predictive performance.

The domain-disjoint evaluation provides an additional assessment of generalization to previously unseen domain groups.
