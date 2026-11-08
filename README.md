
# Machine Learning Assignment 2

## BITS M.Tech AIML – Classification Models

## 1. Problem Statement

The objective of this project is to implement and evaluate
multiple machine learning classification algorithms using
a publicly available dataset.

The project uses the Bank Marketing dataset to predict
whether a customer will subscribe to a term deposit based
on demographic, financial and campaign-related features.

Five classification algorithms are implemented and
compared using six evaluation metrics.

## 2. Dataset Description

- **Dataset:** Bank Marketing Dataset
- **Source:** UCI Machine Learning Repository
- **Dataset URL:** https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Total instances:** 45,211
- **Input features:** 16
- **Target column:** y
- **Target classes:** yes and no
- **Problem type:** Binary Classification

The dataset contains customer information such as age,
job, marital status, education, account balance and
campaign-related details.

Preprocessing includes categorical encoding,
numerical standardization and stratified train-test splitting.

## 3. GitHub Repository Link

**GitHub Repository:** https://github.com/cherrymsatish/ML-Assignment-2/tree/main

The repository contains the source code, trained models,
test dataset, requirements file and documentation.

## 4. Models Used and Performance Comparison

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

### Model Comparison Table

| Model               |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:--------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.8457 | 0.9079 |      0.4182 |   0.8147 | 0.5527 | 0.5092 |
| Decision Tree       |     0.8796 | 0.6961 |      0.4845 |   0.4565 | 0.4701 | 0.4024 |
| KNN                 |     0.8962 | 0.8277 |      0.5990 |   0.3403 | 0.4340 | 0.4001 |
| Naive Bayes         |     0.8548 | 0.8101 |      0.4059 |   0.5198 | 0.4559 | 0.3774 |
| SVM                 |     0.8999 | 0.9076 |      0.6393 |   0.3318 | 0.4368 | 0.4132 |

### Evaluation Metrics

- **Accuracy:** Overall proportion of correct predictions.
- **AUC:** Measures the model's ability to distinguish between classes.
- **Precision:** Proportion of positive predictions that are correct.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.
- **MCC:** Measures the quality of binary classifications,
  considering all four confusion-matrix outcomes.

## 5. Model Performance Observations

- **Logistic Regression:** Accuracy = 0.8457, AUC = 0.9079, Precision = 0.4182,
  Recall = 0.8147, F1 = 0.5527, MCC = 0.5092. Achieved the highest F1 score.
- **Decision Tree:** Accuracy = 0.8796, AUC = 0.6961, Precision = 0.4845,
  Recall = 0.4565, F1 = 0.4701, MCC = 0.4024. Review its metrics against the
  other classifiers.
- **KNN:** Accuracy = 0.8962, AUC = 0.8277, Precision = 0.5990, Recall = 0.3403,
  F1 = 0.4340, MCC = 0.4001. Review its metrics against the other classifiers.
- **Naive Bayes:** Accuracy = 0.8548, AUC = 0.8101, Precision = 0.4059,
  Recall = 0.5198, F1 = 0.4559, MCC = 0.3774. Review its metrics against the
  other classifiers.
- **SVM:** Accuracy = 0.8999, AUC = 0.9076, Precision = 0.6393, Recall = 0.3318,
  F1 = 0.4368, MCC = 0.4132. Achieved the highest precision.

### Overall Winner

**Best-performing model: Logistic Regression**

The model was selected based on F1 score, with AUC and
MCC used as tie-breakers. Its performance should be
considered alongside the other evaluation metrics.


