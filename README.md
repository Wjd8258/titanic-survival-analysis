# Titanic Passenger Survival Analysis

**Author:** Jiadi Wu

**Course:** DSCI 6000 — Final Project (May 2026)

**Tools:** JASP (with R code), Kaggle Titanic dataset

## Overview

What decided who survived the Titanic? This project answers that question three ways on the same data — classical statistics, Bayesian statistics, and machine learning — and checks whether the three approaches agree.

The data are Kaggle's Titanic training set: **891 passengers, 342 survived (38.4%)**. Missing values were handled up front (age imputed by class median, embarked by mode, cabin dropped at 77% missing), a log fare and family-size / age-group / fare-category features were engineered, and the data were split 70/30 for the machine-learning models.

## The three approaches agree

| Method | Key finding |
| --- | --- |
| Classical statistics | Sex, class, age group, and fare category all significantly associated with survival (chi-square, all p < 0.01); logistic regression McFadden pseudo-R² = 0.3405 |
| Bayesian statistics | Best model is Pclass + Sex + Age (posterior probability 0.311, Bayes factor BF₁₀ = 2.198 × 10⁶³ vs. the null); R-hat = 1.01, converged |
| Machine learning | Lasso, Ridge, and Elastic Net all rank the same three predictors — **Sex > Pclass > Age**; Elastic Net performs best |

**Headline numbers:**

- **Sex:** women survived at 74.2% vs. men at 18.9% — women 3.9× more likely to survive. Adjusted odds ratio for men (vs. women) in the logistic model: **0.064**.
- **Class:** 1st class 63.0%, 2nd class 47.3%, 3rd class 24.2%. Adjusted ORs vs. 1st class: 2nd 0.440, 3rd 0.152.
- **Age:** each additional year lowers survival odds by ~4.2% (OR 0.958); children survived best (58.0%), seniors worst (22.7%).
- **Fare:** survivors paid on average £48.40 vs. £22.12 for non-survivors (p < 0.001); log fare OR 1.504.
- **Elastic Net (best ML model):** MSE 0.123, RMSE 0.351, R² 0.472 — ahead of Lasso (0.138 / 0.371 / 0.418) and Ridge (0.150 / 0.387 / 0.357).

## Caveat

Titanic data are historical observational data: the results show statistical association, not causation.

## Repo layout

```
├── README.md     # this file
└── REPORT.md     # full report (English): data & EDA, classical statistics,
                  #   Bayesian statistics, machine learning, conclusions, references
```

The analysis was performed in JASP; the report reproduces the key R code snippets used for imputation, feature engineering, and the train/test split. The dataset itself is public on Kaggle: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data).
