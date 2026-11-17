# Titanic Passenger Survival Analysis: An Integrated Study of Classical Statistics, Bayesian Statistics, and Machine Learning

**Author:** Jiadi Wu

**Course:** DSCI 6000 — Final Project

**Date:** May 5, 2026

## Abstract

This study uses the Titanic dataset from Kaggle to systematically analyze the key factors affecting passenger survival from three angles: classical statistics, Bayesian statistics, and data science & machine learning (DSML). In the classical-statistics part, data preprocessing, descriptive statistics, correlation analysis, and hypothesis testing are used to examine how sex, passenger class, age, and fare relate to survival. In the Bayesian-statistics part, Bayesian logistic regression and Bayes factors are used for model comparison, and Bayesian t-tests verify the key variables. In the machine-learning part, Lasso, Ridge, and Elastic Net models are used to screen and compare variable importance. The results show a high degree of agreement across the three methods: sex, passenger class, and age are the most important factors, and the Elastic Net model performs best in prediction. This study demonstrates the complementarity and effectiveness of different statistical methods in data analysis.

## Part 1: Data Description and Exploratory Analysis

### 1. Data Source

The Titanic dataset is one of the classic public datasets, compiled and published by Kaggle and widely used for machine-learning introduction and teaching. It can be downloaded directly from Kaggle (Titanic: Machine Learning from Disaster, https://www.kaggle.com/competitions/titanic/data). The data are observational, drawn from a real historical event. They record passengers' basic information, including personal identity, passenger class, sex and age, and whether they ultimately survived. The data were not collected through a survey or experiment of our own; they are secondary data compiled from historical records, so there is no questionnaire design or sampling method to describe.

### 2. Data Description

#### 2.1 Dataset Composition

The Kaggle Titanic dataset contains two files: a training set (`train`) and a test set (`test`). The training set contains the true survival outcomes and is used to build models; the test set contains no true outcomes and is used to evaluate model performance on unseen data. A `gender_submission` file is also provided, containing a set of predictions that assume "all female passengers survived and all male passengers did not," as an example of the submission-file format.

| File | Records |
| --- | --- |
| `train.csv` | 891 records for model training, with survival labels |
| `test.csv` | 418 records for model prediction, without survival labels |

#### 2.2 Variables

The `train.csv` dataset contains 12 variables:

| Variable | Type | Meaning | Values |
| --- | --- | --- | --- |
| survived | Binary (target) | Whether survived | 0 = did not survive, 1 = survived |
| pclass | Ordered categorical | Passenger class | 1st, 2nd, 3rd |
| sex | Binary | Sex | male, female |
| age | Continuous numeric | Age | Years; has missing values |
| sibsp | Discrete numeric | Number of siblings / spouses aboard | Integer |
| parch | Discrete numeric | Number of parents / children aboard | Integer |
| fare | Continuous numeric | Ticket price | Pounds sterling |
| embarked | Nominal categorical | Port of embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |
| name | String | Passenger name | Contains title information |
| ticket | String | Ticket number | String format |
| cabin | String | Cabin number | Large amount of missing data |
| passengerid | Identifier | Unique passenger ID | Does not affect the analysis |

#### 2.3 Target Variable Definition

`survived` is the target variable of this analysis: 0 means the passenger did not survive (perished), 1 means the passenger survived.

### 3. Data Processing

#### 3.1 Data Import

Data were opened in JASP (File → Open → Computer → select `train.csv`).

#### 3.2 Missing-Value Handling

The dataset has 891 records. Missing values are concentrated in Age (177 missing), Cabin (687 missing), and Embarked (2 missing); all other variables are complete.

| Variable | Missingness | Treatment |
| --- | --- | --- |
| age | ~20% missing (177/891) | Median imputation |
| embarked | 2 missing | Mode imputation (S) |
| cabin | ~77% missing | Variable dropped |

**1) Age imputation.** A new column `age_imputed` was created (Data → Compute Column) using the following R code, imputing with the median age within each passenger class:

```r
age_imputed <- ifelse(is.na(Age),
  ifelse(Pclass == 1, median(Age[Pclass == 1], na.rm = TRUE),
  ifelse(Pclass == 2, median(Age[Pclass == 2], na.rm = TRUE),
         median(Age[Pclass == 3], na.rm = TRUE))), Age)
```

**2) Embarked imputation.** Embarked is categorical, so mean imputation does not apply; mode imputation is the more reasonable choice. The mode of the Embarked column is "S". A new column `embarked_filled` was created with the following R code:

```r
ifelse(is.na(Embarked), "S", as.character(Embarked))
```

**3) Cabin dropped.** Because its missing rate is too high, the variable was excluded from the analysis.

#### 3.3 Outlier Detection and Handling

Boxplots (Descriptives → Plots → Boxplots) were drawn for Age, Fare, SibSp, and Parch.

- **Age** has a small number of high-age outliers (around 80 years old); **Fare** has many high-fare outliers (maximum 512). The outliers in Age and Fare are genuine variation in the data, not data errors, and should not be deleted. The recommendation is to keep Age and apply a log transformation to Fare rather than deleting outliers.
- A new column `Fare_Log` was created (Data → Compute Column) with the following R code:

```r
log(Fare + 1)
```

#### 3.4 Feature Engineering

**1) Family size.** A new column `family_size` was created with the formula SibSp + Parch + 1.

**2) Age groups.** Age was divided into bands: 12 and under = Child, 13–18 = Teenager, 19–60 = Adult, 61 and over = Senior. A new column `age_group` was created with the following R code:

```r
ifelse(as.numeric(age_imputed) <= 12, "Child",
ifelse(as.numeric(age_imputed) <= 18, "Teenager",
ifelse(as.numeric(age_imputed) <= 60, "Adult", "Senior")))
```

**3) Fare categories.** Fares were divided into three levels: 8.05 and below = Low, 8.06–31.00 = Medium, above 31.00 = High. A new column `fare_category` was created with the following R code:

```r
ifelse(Fare <= 8.05, "Low", ifelse(Fare <= 31.00, "Medium", "High"))
```

#### 3.5 Train/Test Split

The data were randomly split 70/30 into a training set and a test set to evaluate model generalization, using the following R code (seed 123):

```r
set.seed(123)
train_size <- floor(0.7 * nrow(data))
train_indices <- sample(seq_len(nrow(data)), size = train_size)
train_set <- data[train_indices,]
test_set <- data[-train_indices,]
```

Result: 891 records total; training set 623, test set 268.

### 4. Exploratory Data Analysis

#### 4.1 Target Variable Distribution

About **38.4%** of passengers in the training set survived and 61.6% perished. The classes are imbalanced, but only moderately.

#### 4.2 Key Feature Distributions

**a. Age distribution.** Passenger age is slightly right-skewed, with a median of 28 and a mean of 30; passengers are mainly young and middle-aged, ranging from a 5-month-old infant to an 80-year-old, overall close to a normal distribution.

**b. Fare distribution.** Fare is severely right-skewed (skewness 4.79) and extremely peaked (kurtosis 33.40). The mean of £32.20 is pulled far upward by expensive tickets, while the median of £14.45 better represents the typical passenger. The data contain many high-fare outliers (~4–5%), the highest reaching £512 — about 35 times the median. A log transformation or binning of fare is recommended to reduce the influence of extreme values on later modeling.

**c. Passenger class distribution.** Third class accounts for the largest share at **55.1%**, more than half of all passengers. First class and second class account for 24.2% and 20.7% respectively — similar proportions, together less than half.

#### 4.3 Relationships between Key Variables and Survival

**a. Sex and survival.** **74.2%** of female passengers survived and only 25.8% perished; **81.1%** of male passengers perished and only 18.9% survived. The female survival rate is 3.9 times the male rate, fully reflecting the "women and children first" evacuation principle.

**b. Passenger class and survival.** First-class survival is 63%, second class 47.3%, third class only 24.2% — a clear positive association. First-class survival is 2.6 times third-class survival, fully reflecting how social class affected the chance of survival.

**c. Age and survival.** Children have the highest survival rate at **58%**, the only group in which survivors outnumber the dead. Seniors have the lowest at only 22.7%; adults and teenagers are 36.6% and 42.9% respectively.

**d. Fare and survival.** High-fare passengers have the highest survival rate at **58.1%**, the only group in which survivors outnumber the dead. Low-fare passengers survive at only 20.4%; the dead outnumber survivors by nearly 4 to 1.

#### 4.4 Correlation Analysis

Pearson correlations (sex recoded numerically: female = 1, male = 0):

- Pclass and Fare: r ≈ **−0.549** (strong negative correlation) — the higher the class, the more expensive the ticket.
- Fare and Survived: r ≈ **0.257** (weak positive correlation) — the higher the fare, the higher the survival rate.
- Age and Survived: r ≈ **−0.077** (extremely weak negative correlation) — age and survival are very weakly related.
- Pclass and Survived: r ≈ **−0.338** (moderate negative correlation) — the lower the class, the lower the survival rate.
- sex_numeric and Survived: r ≈ **0.543** (strong correlation) — female survival is far higher than male survival.
- family_size and Survived: r ≈ **0.026** (no / extremely weak correlation) — family size is almost unrelated to survival.

The heatmap confirms: the main factors associated with survival are being female (r = 0.543), better class (r = −0.338), and higher fare (r = 0.257). **Multicollinearity warning:** family_size is highly correlated with SibSp (r = 0.868) and Parch (r = 0.804), so they should not be used together in modeling. Other notable relationships: higher class means more expensive tickets (Pclass and Fare, r = −0.549); first-class passengers are older (Pclass and Age, r = −0.369).

#### 4.5 Group Comparisons of Survival

**a. t-tests (binary groups).** Independent-samples t-tests (survived as grouping variable; age_imputed and fare as variables) show: age does not differ significantly between survivors and non-survivors (**p = 0.159**); fare differs extremely significantly (**p < 0.001**). The fare bar chart shows survivors' mean fare clearly higher than non-survivors'. Survivors' average fare is significantly higher than non-survivors', indicating that higher-fare passengers had a higher probability of survival.

**b. ANOVA (multiple groups).** Passenger class (Pclass) has an extremely significant effect on survival (**F = 57.96, p < 0.001**). Descriptive statistics show survival falling as class falls — first class 63.0%, second class 47.3%, third class only 24.2%. The higher the class, the higher the survival probability; first-class survival is about 2.6 times third-class survival.

## Part 2: Classical Statistics

### 1.1 Research Background and Research Questions

The Titanic dataset records passengers' sex, age, passenger class, fare, family situation, and whether they ultimately survived. Because the dependent variable "whether survived" is a clear binary variable, and the data contain both categorical and continuous variables, this dataset is well suited for classical statistical analysis.

Note that this part uses the team's uniformly cleaned data — that is, the version in which preprocessing, missing-value handling, and basic variable construction were already completed. On this unified basis, this part completes the research-question design, interval estimation, hypothesis testing, logistic regression analysis, and interpretation of results for classical statistics.

Given the characteristics of the Titanic data and the requirements of this part, the analysis centers on three research questions:

1. Are sex, passenger class, age group, and fare level significantly associated with passenger survival?
2. Do survivors and non-survivors differ significantly in age and fare?
3. After controlling for multiple variables simultaneously, which factors still significantly affect the probability of passenger survival?

To answer these questions, descriptive statistics, 95% confidence intervals, chi-square tests of independence, Welch t-tests, and a binary logistic regression model are used in turn.

### 1.2 Data Source, Unified Data Description, and Variable Definitions

The data come from the Titanic training set publicly released by Kaggle, containing 891 passenger records with a survival label for each passenger. The raw data include passenger identity information, passenger class, sex, age, fare, port of embarkation, and family size.

This part does not re-clean the raw data; it uses the team's uniformly cleaned data version directly. In that unified version, missing Age values have been imputed, forming the variable `age_imputed`; missing Embarked values have been filled, forming `embarked_filled`; the fare variable has been log-transformed to form `Fare_Log`; and `family_size`, `age_group`, `fare_category`, and `sex_numeric` were constructed as the basis for later analysis.

The analysis below mainly uses the following variables: Survived (whether survived, 0 = did not survive, 1 = survived), Pclass (passenger class), Sex (sex), age_imputed (imputed age), Fare_Log (log-transformed fare), family_size (family size), age_group (age group), and fare_category (fare level).

### 1.3 Exploratory Analysis Related to the Research Questions

Before formal statistical inference, a brief look at several key variables helps clarify the direction of the later analysis. Overall, the cleaned data contain 891 passengers, of whom **342 survived and 549 did not, an overall survival rate of 38.38%**. This shows that most passengers in the Titanic event were not rescued, and the overall chance of survival was not high.

By sex, the female survival rate is 74.20%, while the male rate is only 18.89%. By passenger class, first class survives at 62.96%, second class at 47.28%, and third class at only 24.24%. By age group, children survive at the highest rate, 57.97%; teenagers at 42.86%; adults at 36.58%; and seniors at the lowest, only 22.73%. By fare level, high-fare passengers survive at 58.11%, medium-fare passengers at 40.36%, and low-fare passengers at only 20.35%.

In addition, the raw fare distribution is clearly right-skewed, so the team applied a log transformation to it beforehand. The analysis below prioritizes `Fare_Log`, which reduces the influence of extreme high fares on the results and is more robust for statistical analysis. Overall, these exploratory results already suggest that Titanic passenger survival was not randomly distributed, but was likely related to sex, passenger class, age, and fare.

### 1.4 Research Methods

To answer the research questions above, several commonly used classical-statistics methods are adopted. First, descriptive statistics and 95% confidence intervals describe the overall and group survival rates. Giving only a sample proportion is not enough, because sample results are themselves subject to random fluctuation; adding a confidence interval more fully reflects the range in which the population proportion may fall.

Second, for categorical variables such as sex, passenger class, age group, and fare level, contingency tables and chi-square tests of independence are used to analyze whether they are significantly associated with survival. The chi-square test suits the relationship between two categorical variables and is therefore ideal for answering "is a categorical feature associated with survival." To further describe the strength of association, Cramér's V is used as an effect-size measure.

Third, for continuous variables such as age and fare, Welch t-tests compare the mean differences between survivors and non-survivors. The Welch t-test is used instead of the ordinary independent-samples t-test because the two groups' variances are not necessarily equal, which is the safer treatment. Given the clearly right-skewed raw fare distribution, this part focuses on the log-transformed `Fare_Log` for comparison.

Finally, to analyze which factors still significantly affect the probability of passenger survival after controlling for multiple variables simultaneously, a binary logistic regression model is built. Logistic regression suits a binary dependent variable: it can test whether variables are significant and can explain the direction and strength of their effects through odds ratios (OR). It is therefore one of the most central methods in this part.

log[P(Survived = 1) / (1 − P(Survived = 1))] = β₀ + β₁Sex + β₂Pclass + β₃age_imputed + β₄Fare_Log + β₅family_size

### 1.5 Results and Analysis

#### a. Descriptive Statistics and Survival-Rate Confidence Intervals

In the cleaned unified data, there are 891 passengers, of whom 342 survived and 549 did not, an overall survival rate of 38.38%. Constructing a 95% confidence interval shows the overall survival rate lies roughly between **35.25% and 41.62%**. This indicates that Titanic passengers' overall chance of survival was not high, and the number of non-survivors clearly exceeded the number of survivors.

By sex, the female survival rate is 74.20% with a 95% CI of [69.09%, 78.73%]; the male survival rate is only 18.89% with a 95% CI of [15.91%, 22.29%]. By passenger class, first class survives at 62.96%, second class at 47.28%, and third class at only 24.24%. These results show fairly clear structural differences in survival by sex and passenger class.

**Table 1 — Overall and group survival rates with 95% confidence intervals**

| Group | n | Survivors | Survival rate | 95% CI |
| --- | --- | --- | --- | --- |
| Overall | 891 | 342 | 38.38% | [35.25%, 41.62%] |
| Female | 314 | 233 | 74.20% | [69.09%, 78.73%] |
| Male | 577 | 109 | 18.89% | [15.91%, 22.29%] |
| Pclass 1 | 216 | 136 | 62.96% | [56.35%, 69.12%] |
| Pclass 2 | 184 | 87 | 47.28% | [40.20%, 54.48%] |
| Pclass 3 | 491 | 119 | 24.24% | [20.66%, 28.22%] |

Table 1 summarizes the overall and group survival rates. Female and higher-class passengers survive at clearly higher rates, suggesting passenger survival may be related to sex and passenger class.

#### b. Chi-Square Tests of Categorical Variables and Survival

To further judge whether these differences are statistically significant, chi-square tests of independence were run between several categorical variables and survival. The results show that sex, passenger class, age group, and fare level are all significantly associated with whether a passenger survived. Sex has the strongest effect, passenger class and fare level are next, and age group is relatively weaker but still statistically significant.

**Table 2 — Chi-square test results for categorical variables and survival**

| Variable | χ² | df | p value | Cramér's V | Conclusion |
| --- | --- | --- | --- | --- | --- |
| Sex | 260.72 | 1 | < 0.001 | 0.541 | Significant, strongest association |
| Pclass | 102.89 | 2 | < 0.001 | 0.340 | Significant; higher class, higher survival |
| age_group | 15.08 | 3 | 0.0018 | 0.130 | Significant, but relatively weaker |
| fare_category | 76.34 | 2 | < 0.001 | 0.293 | Significant; higher fare, higher survival |

Table 2 shows that sex, passenger class, age group, and fare level are all significantly associated with passenger survival. Sex has the largest association strength, indicating its most pronounced effect on the survival outcome.

#### c. Between-Group Tests for Continuous Variables

Beyond categorical variables, differences between survivors and non-survivors on continuous variables were compared, focusing on age and fare. For age, survivors average 28.27 years and non-survivors 29.56 years. A Welch t-test shows the mean age difference is not significant (**p = 0.167**), meaning that from the angle of a single-variable mean comparison alone, age itself does not show a particularly clear difference. (Part 1 reported p = 0.159 for the same comparison.)

For fare, survivors average £48.40 and non-survivors £22.12, a highly significant difference (p < 0.001). Given the clearly right-skewed raw fare distribution, the log-transformed `Fare_Log` was compared as well: survivors average 3.367, non-survivors 2.710, still significantly different (p < 0.001). This shows that the fare–survival difference is not an accidental product of extreme values, but remains stably significant after log transformation.

#### d. Binary Logistic Regression Results

To further analyze which factors still significantly affect the probability of passenger survival after controlling for multiple variables simultaneously, a binary logistic regression model was built, with Survived as the dependent variable and sex, passenger class, age_imputed, Fare_Log, and family_size as explanatory variables. Overall, the logistic regression model is significantly better than the null model (p < 0.001), with a McFadden pseudo-R² of **0.3405**, indicating the variables in the model collectively explain whether a passenger survived reasonably well.

For sex, with female as the reference group, the odds ratio (OR) for males is **0.064 (p < 0.001)**, meaning that after controlling for other variables, male survival odds remain far below female odds. For passenger class, with first class as the reference, the OR for second class is **0.440 (p = 0.011)** and for third class **0.152 (p < 0.001)**, showing that lower-class passengers' survival probability remains significantly below first-class passengers'. For age, the OR for age_imputed is **0.958 (p < 0.001)**, meaning that, holding other variables constant, each additional year of age lowers survival odds by about 4.2%. For fare, the OR for Fare_Log is **1.504 (p = 0.013)** — that is, after controlling for other factors, higher fare levels mean a higher chance of survival. Finally, the OR for family_size is **0.716 (p < 0.001)**, meaning each additional family member lowers survival odds.

**Table 3 — Binary logistic regression results for Titanic passenger survival probability**

| Variable | OR | 95% CI | p value | Interpretation |
| --- | --- | --- | --- | --- |
| Male (ref. Female) | 0.064 | [0.044, 0.095] | < 0.001 | Male survival odds significantly lower than female |
| Pclass 2 (ref. 1st) | 0.440 | [0.235, 0.836] | 0.011 | 2nd-class survival probability significantly lower than 1st class |
| Pclass 3 (ref. 1st) | 0.152 | [0.075, 0.316] | < 0.001 | 3rd-class survival probability significantly lower than 1st class |
| age_imputed | 0.958 | [0.943, 0.974] | < 0.001 | Older age, lower survival probability |
| Fare_Log | 1.504 | [1.094, 2.090] | 0.013 | Higher fare, higher survival probability |
| family_size | 0.716 | [0.617, 0.838] | < 0.001 | Larger family size, lower survival probability |

Table 3 shows the main results of the binary logistic regression model. After controlling for other variables, sex, passenger class, age, fare, and family size all significantly affect the probability of passenger survival, with sex and passenger class the most prominent, fare a stable positive effect, and age and family size negative effects. Compared with single-variable tests, logistic regression provides a more complete multi-factor perspective and makes this part's conclusions more convincing.

#### e. Model Diagnostics and Reliability of Results

When interpreting logistic regression results, the model's stability and the reliability of its results must also be considered. Because several explanatory variables enter the same model, multicollinearity is a concern: if the independent variables are too strongly correlated, coefficient estimates can become unstable and affect interpretation. In this study's variable setup, SibSp, Parch, and family_size were not all put into the model; only family_size was kept as the family-size indicator, reducing the impact of high correlation among family variables.

Also, based on the earlier correlation analysis, although passenger class and fare are related, there is no severe overlap among multiple variables overall. From the variable structure, therefore, the current logistic regression model has no obvious serious multicollinearity risk. Combined with the model being significantly better than the null model overall, the logistic regression analysis in this study can be considered to have good explanatory power, and its conclusions are reasonably reliable.

### 1.6 Conclusions and Evaluation

Based on the team's uniformly cleaned Titanic data, this part analyzes the main factors affecting passenger survival from the angle of classical statistics. The overall results show that Titanic passenger survival was not random, but clearly related to several variables. Whether descriptive statistics, chi-square tests, or logistic regression analysis, all consistently show that sex and passenger class are the most important factors: female passengers and higher-class passengers have clearly higher survival probabilities. At the same time, the higher the fare, the greater the chance of survival; age and family size show some negative effect.

Methodologically, classical statistical methods have two fairly clear advantages in this study. First, they are highly interpretable, and the results map directly onto the research questions — for example, the chi-square test suits judging whether a categorical variable is related to survival, while logistic regression better suits analyzing the combined effect of multiple variables acting together. Second, the outputs are clear — confidence intervals, significance levels, and odds ratios all help interpret the conclusions more intuitively.

However, this study also has limitations. First, Titanic data are observational historical data, so statistical associations can be identified but strict causal conclusions cannot be drawn. Second, the age variable was imputed, which preserves sample size but may weaken the true variation originally in age. Third, some potentially valuable variables, such as Cabin, were not included in this part's model because of excessive missingness, which may cause some information loss. Finally, although the logistic regression model used here is highly interpretable, for more complex nonlinear relationships or variable interactions, the Bayesian statistics and machine-learning parts can explore further.

## Part 3: Bayesian Statistics

### 1. Dataset

The data record passengers' basic information at the time, including personal identity, passenger class, sex and age, and whether they ultimately survived. There are three main tables and variables:

- a. `train.csv`: 891 records for model training, with survival labels.
- b. `test.csv`: 418 records for model prediction, without survival labels.
- c. `gender_submission.xlsx`: a reference table of survival outcomes for the test set.

Variables:

#### 1.2 Descriptive Statistics of the Variables

#### 1.3 Missing-Value Handling

In this report on Bayesian statistical techniques, missing values are handled by removing them (listwise deletion). All variables' missing values are shown as "." and are screened out.

#### 1.4 Observations Are Independent, and the Data Are Close to Normally Distributed

### 2. Research Questions

2.1 Can Bayesian logistic regression find the best model to verify survival rates?

2.2 Can Bayesian statistical techniques verify survival rates and related parameters?

### 3. Analyzing the Questions

3.1 Use Bayesian logistic regression to analyze each relevant variable, using the models' posterior probability distributions and Bayes factors to select models and variables, and finally obtain the optimal model.

3.2 Bayesian t-tests can be used for verification.

### 4. Solving the Questions

#### 4.1 Obtaining the Best Model

Taking whether the passenger survived as the dependent variable and adding the other data items as factors or parameters into Bayesian logistic regression, the model containing Pclass, Sex, and Age is the best explanatory model. The model's posterior probability is **0.311**, and its Bayes factor **BF₁₀ is as high as 2.198 × 10⁶³**, significantly better than the null model and models containing other variables (such as fare and port of embarkation). This shows that passenger class, sex, and age are the three most critical factors determining Titanic survival probability.

#### 4.2 Posterior Results of the Bayesian Logistic Regression

| Variable | Posterior mean | 95% credible interval | Interpretation |
| --- | --- | --- | --- |
| Sex[male] | −2.418 | [−2.822, −2.013] | Male survival probability significantly lower than female; evidence extremely strong |
| Pclass[2] | −1.107 | [−1.723, −0.491] | Compared with first class, second-class survival probability lower |
| Pclass[3] | −2.324 | [−2.964, −1.683] | Compared with first class, third-class survival probability lowest |
| Age | −0.035 | [−0.050, −0.020] | Weak negative effect: older age, slightly lower survival probability |

All four effects are significantly negative: male survival is far below female survival, second- and third-class survival are below first class, and age has a small negative effect on survival.

#### 4.3 Convergence Diagnostics

The model's R-hat value is **1.01**, indicating the model has converged and the results are trustworthy.

#### 4.4 Bayesian t-Tests for Group Differences

**a. Sex and survival.** Sex shows an extreme difference in survival — BF₁₀ is extremely large, strongly supporting that survival status is related to sex, with female survival significantly higher than male survival.

**b. Age and survival.** The Bayesian t-test shows that the age of the survivor group is significantly lower.

**c. Passenger class and survival.** Passenger class has an extremely strong main effect on survival (BF extremely large); post-hoc comparisons show first-class survival significantly higher than second- and third-class survival.

### 5. Evaluation

#### 5.1 Bayesian Logistic Regression Can Find the Best Model to Verify Survival Rates

Bayesian logistic regression was used to explore the factors affecting Titanic passenger survival (Survived). Through model comparison, Bayes factors (BF₁₀) and model posterior probabilities (P(M|data)) were computed for all models containing different combinations of independent variables, and the Pclass + Sex + Age model was finally selected for subsequent inference.

#### 5.2 Bayesian t-Tests Further Verify the Core Variables

To further verify group differences in the core variables, Bayesian independent-samples t-tests and Bayesian one-way ANOVA were used to test differences in sex, age, and class between the survivor and death groups. The Bayesian t-tests and ANOVA found: the sex difference between the survived/death groups is extremely strong; the age difference is significant; the class difference is extremely strong. And Bayesian logistic regression found that putting the three variables together still yields the optimal model. The two approaches mutually verify each other, showing that sex, age, and class are both the key sources of group differences and important factors in predicting survival status.

## Part 4: Data Science and Machine Learning (DSML)

**Goal:** Use Lasso, Ridge, and Elastic Net to model Titanic passenger survival data and screen for the key variables significantly related to passenger survival probability.

### 1. Data Import

Open JASP (File → Open → Computer → select `train.csv`).

### 2. Data Preparation

#### 2.1 Recode Sex as a Binary Variable

A new column "Sex 1 or 2" was created, with male set to 1 and female set to 2, using the following R code:

```r
ifElse(Sex == "male", 1, 2)
```

#### 2.2 Age Missing-Value Imputation

A new column `age_imputed` was created (Data → Compute Column), imputing with the median, using the following R code:

```r
age_imputed <- ifelse(is.na(Age),
  ifelse(Pclass == 1, median(Age[Pclass == 1], na.rm = TRUE),
  ifelse(Pclass == 2, median(Age[Pclass == 2], na.rm = TRUE),
         median(Age[Pclass == 3], na.rm = TRUE))), Age)
```

#### 2.3 Recode Port of Embarkation Numerically

Set C = 1, Q = 2, S = 3, missing = 4, using the following R code:

```r
ifElse(is.na(Embarked), 4, ifElse(Embarked == "C", 1, ifElse(Embarked == "Q", 2, 3)))
```

### 3.1 Lasso Analysis

Machine Learning → Regularized Linear. Results:

The optimal λ chosen by the system is **0.010**. **MSE** (mean squared error) is **0.138** — the average of squared errors; lower is better, and this value is relatively low. **RMSE** (root mean squared error) is **0.371** — the model's predicted survival probability deviates from the true value by about 0.371 on average; this number is small, indicating small prediction error. **R²** (coefficient of determination) is **0.418** — the model explains about 41.8% of the variation in survival outcomes, indicating the included variables have strong explanatory power for survival.

Among the coefficients, **sex (SexB)** is the strongest at **0.239**, positive, meaning female survival is higher, consistent with history. Second is **passenger class (Pclass)** at **−0.130**, negative, meaning the lower the class (the larger the number), the lower the survival probability. **Age (age_imputed)** is relatively large at **−0.049**, negative, meaning older age lowers survival probability. **SibSp** (number of siblings/spouses aboard) is −0.017, negative but very small, indicating a weak relationship with survival probability and almost negligible influence. **Embarked B** is −0.016, negative and likewise very small, indicating a weak relationship. **Parch** (number of parents/children) is 0.000 — Lasso directly shrank it to 0, meaning the variable has no predictive role for survival probability. **Fare** is 0.000, also shrunk to 0 by Lasso, meaning it likewise has no predictive role.

The final equation is:

Survived = 0.388 + 0.239 × SexB − 0.049 × age_imputed − 0.016 × Embarked B − 0.13 × Pclass + 0 × Parch + 0 × Fare − 0.017 × SibSp

**Coefficient shrinkage path plot:** The horizontal axis is the regularization strength λ — larger values mean stronger penalties; the vertical axis is each variable's regression coefficient. The SexB coefficient stays positive and is the highest overall, the most influential core factor; the Pclass coefficient stays negative, the second-largest influence. The other variables' coefficients shrink toward 0 relatively quickly and are weak overall, indicating small explanatory power and practical influence.

**Lambda evaluation plot:** The horizontal axis is the regularization parameter λ — larger values mean stronger penalties; the vertical axis is cross-validated error, measuring model prediction performance. The curve first falls and then rises; the lowest point is where the overall error is smallest. The minimum λ is about 0.010.

### 3.4 Ridge Regression

Results: the optimal λ chosen by the system is **0.027**. **MSE** is **0.150**, slightly higher than the Lasso model, so error control is slightly weaker. **RMSE** is **0.387**, small, indicating small prediction error — but slightly higher than Lasso, so Ridge's prediction error is slightly larger. **R²** is **0.357** — the model explains about 35.7% of the variation in survival outcomes, lower than Lasso's explanatory power, though it still reflects the core variables' influence on survival.

Unlike Lasso, the Ridge model retains all variables' coefficients and removes no features, so it better reflects each variable's weak influence; but on this data, its overall prediction performance and explanatory power are both slightly weaker than Lasso's.

**Coefficient analysis:** **Sex (SexB)** is the strongest at **0.230**, positive — female survival is higher, consistent with history. Second is **passenger class (Pclass)** at **−0.145**, negative — the larger the class number (the lower the class), the lower the survival probability. **Age (age_imputed)** is third at **−0.073**, negative — older age lowers survival probability. SibSp is −0.038, Parch −0.025, Embarked B −0.024, and Fare +0.019 — all four are small, indicating weak relationships with survival probability and limited influence.

The final equation is:

Survived = 0.386 + 0.222 × SexB − 0.074 × age_imputed − 0.037 × Embarked B − 0.156 × Pclass − 0.011 × Parch + 0.013 × Fare − 0.033 × SibSp

**Coefficient shrinkage path plot:** The lines approach zero relatively quickly, but sex (SexB) and passenger class (Pclass) remain visibly the relatively larger influences.

**Lambda evaluation plot:** The horizontal axis is the regularization parameter λ and the vertical axis is cross-validated error. When λ is low, the error is lower, then it rises quickly and gradually levels off.

### Elastic Net

Results: the optimal λ chosen by the system is **0.001**, with the elastic-net parameter **α = 0.5** (combining the regularization characteristics of Lasso and Ridge). **MSE** is **0.123**, the lowest of the three models, with the best error control. **RMSE** is **0.351**, the smallest of the three models, so Elastic Net has the smallest prediction error. **R²** is **0.472** — the model explains about 47.2% of the variation in survival outcomes, the strongest explanatory power of the three models.

**Coefficient analysis:** **Sex (SexB)** is the strongest at **0.235**, positive — female survival is higher, consistent with history. Second is **passenger class (Pclass)** at **−0.137**, negative — the larger the class number (the lower the class), the lower the survival probability. **Age (age_imputed)** is third at **−0.079**, negative — older age lowers survival probability. SibSp is −0.036, Parch −0.033, Embarked B −0.031, and Fare +0.027 — all four are small, indicating weak relationships with survival probability and limited influence.

The final equation is:

Survived = 0.395 + 0.235 × SexB − 0.079 × age_imputed − 0.031 × Embarked B − 0.137 × Pclass − 0.033 × Parch + 0.027 × Fare − 0.036 × SibSp

**Coefficient shrinkage path plot:** The horizontal axis is the regularization strength λ and the vertical axis is each variable's regression coefficient. SexB stays positive and is the highest overall, the most influential core factor; Pclass stays negative, the second-largest influence. The other variables' coefficients shrink toward 0 relatively quickly and are weak overall, indicating small explanatory power and practical influence.

**Lambda evaluation plot:** The horizontal axis is the regularization parameter λ and the vertical axis is cross-validated error. The curve rises overall; the lowest point is where the overall error is smallest. The minimum λ is about 0.001.

### 4. Comparing the Three Models

- **MSE:** Elastic Net (0.123) < Lasso (0.138) < Ridge (0.150) — Elastic Net has the lowest prediction error of the three models.
- **RMSE:** Elastic Net (0.351) < Lasso (0.371) < Ridge (0.387) — Elastic Net controls prediction error best.
- **R²:** Elastic Net (0.472) > Lasso (0.418) > Ridge (0.357) — Elastic Net explains the variation in survival outcomes most strongly.
- **Validation-set MSE:** Elastic Net (0.138) < Lasso (0.143) < Ridge (0.144) — Elastic Net has the lowest error on the validation set used for tuning, indicating the best performance in the parameter-optimization stage and better generalization potential.
- **Test-set MSE:** Elastic Net (0.123) < Lasso (0.138) < Ridge (0.150) — Elastic Net still has the lowest error on the independent test set, confirming stable performance with no overfitting.

The Lasso model uses L1 regularization to automatically remove weakly influential variables such as Parch and Fare, streamlining variable selection. The Ridge model uses L2 regularization to retain all variables, smoothly shrinking coefficients and better reflecting variables' weak influences. Elastic Net combines the advantages of both regularizations, achieving both partial variable shrinkage and model stability.

Elastic Net combines Lasso's variable-selection ability with Ridge's coefficient-smoothing property, achieving the best balance of prediction performance and explanatory power on this data — the best-performing of the three models.

**Variables:** Across Lasso, Ridge, and Elastic Net, the ranking of variable importance is consistent: all three models show that **sex (SexB)** has the greatest effect on survival, ranking first; **passenger class (Pclass)** is second; **age (age_imputed)** is third. The only difference is the degree of coefficient shrinkage: Lasso removes irrelevant variables, making the core variables' coefficients stand out more; Ridge shrinks all variables' coefficients without changing their relative importance; Elastic Net combines the two. The three methods agree on the ranking of the key factors.

Lasso shrinks the coefficients of Parch (parents/children) and Fare directly to 0. Ridge removes no variables; all feature coefficients are shrunk but none are 0. SibSp, Parch, Embarked B, and Fare all have small coefficients.

In summary, combining the Lasso, Ridge, and Elastic Net results, the ranking of the key factors affecting Titanic passenger survival is highly consistent: sex (Sex) has the greatest effect, ranking first; passenger class (Pclass) is second; age (Age) is third.

Fare in the Titanic dataset has clear extreme high values — typical outliers. The three models handle outliers differently. In Lasso, Fare's coefficient is shrunk directly to 0, removing the outliers' interference. In Ridge, because Ridge only shrinks coefficients and does not remove variables, extreme high fares are still included in the model and increase the model's overall fluctuation. This is one reason Ridge has the highest prediction error (MSE = 0.150, RMSE = 0.387) and the weakest explanatory power (R² = 0.357) of the three models — it is most negatively affected by outliers. In Elastic Net, combining the advantages of L1 and L2 regularization, resistance to outliers sits between the two: it neither removes Fare directly like Lasso nor fully retains it and suffers outlier interference like Ridge. Its coefficient is kept at 0.027, retaining weak information while effectively suppressing fluctuation from extreme values, so its stability and prediction accuracy are best.

## Part 5: Overall Conclusions

In summary, this study analyzed the factors affecting passenger survival using the Titanic dataset, from the angles of classical statistics and Bayesian statistics. The two methods are highly consistent in their conclusions: sex, passenger class, and age are the core variables affecting survival probability. Female passengers survive at significantly higher rates than male passengers, higher-class passengers survive at clearly higher rates than lower-class passengers, and the older the age, the relatively lower the survival probability. In addition, fare level shows a positive association with survival in the classical statistical analysis, further reflecting how socioeconomic factors affected the chance of survival.

From a methodological angle, classical statistical methods have good explanatory power, clearly showing relationships between variables through significance tests and correlation analysis; Bayesian statistical methods provide a more complete expression of uncertainty through posterior distributions and Bayes factors, and are more flexible in model selection. The two methods complement each other, making the conclusions more robust and reliable.

It should be noted that the data used in this study are historical observational data, so the results only reflect statistical associations between variables and cannot directly support causal inference. Also, some variables underwent missing-value handling and feature-engineering transformations, which may affect the analysis results. Overall, this study not only reveals the significant social-structural differences in the Titanic event, but also demonstrates the applied value of different statistical methods in data analysis, laying a foundation for further research using data science and machine learning methods.

In addition, in the data science and machine learning (DSML) part, the comparison of Lasso, Ridge, and Elastic Net models found consistent variable-importance rankings across the three models, all showing that sex, passenger class, and age are the most critical predictive factors. The Elastic Net model performs best in prediction, with the lowest mean squared error (MSE) and the highest explanatory power (R²), further confirming the robustness of the first two parts' results.

From a method-comparison angle, classical statistical methods are highly interpretable and clearly reveal relationships between variables; Bayesian methods have advantages in uncertainty expression and model comparison; and data science and machine learning methods perform better in prediction, with the Elastic Net model achieving the best fit. The three methods each have their strengths and complement each other, making the conclusions more comprehensive and robust.

## Part 6: References

Kaggle. (n.d.). Titanic: Machine Learning from Disaster. https://www.kaggle.com/c/titanic

JASP Team. (2024). JASP (Version X.X) [Computer software]. https://jasp-stats.org

Wagenmakers, E. J., et al. (2018). Bayesian inference for psychology. Part I: Theoretical advantages and practical ramifications. *Psychonomic Bulletin & Review*, 25(1), 35–57.

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An introduction to statistical learning*. Springer.

## Appendix: Data Source and Processing

**Data source:** Kaggle public dataset — https://www.kaggle.com/c/titanic/data

**Processing:** See the data-cleaning descriptions in each part (Part 1, Section 3 for imputation and feature engineering; Part 3 for the Bayesian part's listwise deletion of missing values; Part 4, Section 2 for the DSML part's recoding).
