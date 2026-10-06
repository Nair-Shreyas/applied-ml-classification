# Applied ML: Entropy & Bank Term Deposit Classification

![Project Overview](docs/images/1_project_overview.png)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python: 3"/>
  <img src="https://img.shields.io/badge/Runs_on-Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white" alt="Runs on: Google Colab"/>
  <img src="https://img.shields.io/badge/Data-UCI_Bank_Marketing-c9440c?style=flat-square" alt="Data: UCI Bank Marketing"/>
</p>

Two related notebooks on classification fundamentals and a full applied pipeline.

## `entropy_information_gain.ipynb`
A from-scratch implementation of entropy and information gain: the math decision trees use to pick which feature to split on. Good for understanding what's happening under the hood before reaching for `sklearn`.

## `bank_term_deposit_classification.ipynb`
An end-to-end classification pipeline predicting whether a bank client will subscribe to a term deposit:
- Feature encoding and standardization
- **SMOTE** for class imbalance
- **Decision Tree, Random Forest, and SVM** models compared
- **GridSearchCV** for hyperparameter tuning
- Evaluated on accuracy, precision, recall, and F1

`bank.csv` (the UCI Bank Marketing dataset, 4,521 records) is bundled so the notebook runs standalone.

### Results
Random Forest and SVM (both tuned via GridSearchCV) reach ~95.5% accuracy, well above the baseline Decision Tree:

![Model Comparison](docs/images/2_model_comparison.png)

## Tech
Python, pandas, scikit-learn, imbalanced-learn (SMOTE)
