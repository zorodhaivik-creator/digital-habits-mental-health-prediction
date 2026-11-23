# digital-habits-mental-health-prediction
Machine Learning regression project analysing digital habits, screen time, sleep and stress to predict mood scores using multiple ML algorithms and Artificial Neural Networks.
# Digital Habits vs Mental Health

## Project Overview

This project investigates the relationship between digital habits and mental well-being by developing Machine Learning regression models to predict mood scores.

It analyses how screen time, social media usage, sleep duration and stress levels relate to self-reported mood scores.

**Problem Type:** Supervised Machine Learning – Regression

**Target Variable:** `mood_score`

## Objectives

- Analyse digital habits and their relationship with mood scores.
- Perform data cleaning and exploratory data analysis.
- Identify multicollinearity using Variance Inflation Factor (VIF).
- Develop and compare multiple regression algorithms.
- Experiment with different Artificial Neural Network architectures.
- Evaluate models using Mean Squared Error (MSE), R² and Mean Absolute Percentage Error (MAPE).

## Dataset Description

- Original records: 100,000
- Records after duplicate removal: 98,135
- Original features: 6
- Missing values: 0
- Duplicate records removed: 1,865

### Features

| Feature | Description |
|---|---|
| screen_time_hours | Daily screen time |
| social_media_platforms_used | Number of social media platforms used |
| hours_on_TikTok | Daily TikTok usage |
| sleep_hours | Average sleep duration |
| stress_level | Self-rated stress level |
| mood_score | Self-rated mood score (target) |

## Data Preprocessing

- Loaded and inspected the dataset using Pandas.
- Checked missing values and duplicate records.
- Removed 1,865 duplicate records.
- Checked feature ranges and potential outliers.
- Separated five independent variables and one target variable.
- Applied StandardScaler for ANN input features.

## Exploratory Data Analysis

Performed:
- Univariate analysis using histograms and KDE plots.
- Outlier detection using boxplots.
- Bivariate analysis using scatterplots and regression lines.
- Pearson correlation analysis.
- Correlation heatmap.
- Multicollinearity analysis using VIF.

The maximum VIF was 3.98, indicating that the predictors were below the chosen threshold of 5.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Statsmodels
- Scikit-learn
- XGBoost
- TensorFlow / Keras

## Machine Learning Models

1. Linear Regression
2. Decision Tree Regressor
3. XGBoost Regressor
4. K-Nearest Neighbors Regressor
5. Support Vector Regression (SVR)
6. Random Forest Regressor
7. Artificial Neural Network (ANN)

Models were compared using five train-test split configurations:
- 80:20
- 75:25
- 70:30
- 65:35
- 60:40

## Model Evaluation

The following evaluation metrics were used:

- Mean Squared Error (MSE)
- R-squared (R²)
- Mean Absolute Percentage Error (MAPE)

### Best Reported Results

| Model | R² |
|---|---:|
| XGBoost | 0.6668 |
| SVR | 0.6602 |
| ANN | 0.6551 |
| Random Forest (tuned) | 0.6428 |
| KNN (tuned) | 0.6110 |
| Linear Regression | 0.5731 |
| Decision Tree | 0.2954 |

**Best overall model:** XGBoost Regressor

**Best reported R²:** 0.6668 (80:20 split).

## Artificial Neural Network

Experimented with eight ANN configurations involving:
- Different hidden-layer architectures.
- ReLU, tanh and ELU activation functions.
- Different learning rates.
- Dropout and Batch Normalization.
- Different batch sizes and training epochs.

The best reported ANN configuration achieved an R² of 0.6551 using three hidden layers (128, 64, 32) with tanh activation.

## Key Findings

- XGBoost achieved the highest reported R² across the five train-test splits.
- Non-linear models performed better than Linear Regression.
- Digital habits and stress-related features explained approximately 66% of the variance in mood scores according to the best reported model.
- ANN experiments demonstrated that a larger network does not necessarily produce better generalisation.
- Other factors beyond the five selected predictors may influence mood.

## Limitations

- The dataset is based on self-reported, survey-style records.
- Predictive performance is limited by the available features.
- The models identify associations and cannot establish causation.
- Most model comparisons used a single random seed.
- ANN experiments were conducted using the 80:20 split only.

## Future Improvements

- Integrate real-time smartphone screen-time data.
- Explore longitudinal mood monitoring.
- Implement explainable AI techniques such as SHAP.
- Develop personalised habit recommendations.
- Validate models using independently collected datasets.
- Build a privacy-conscious application prototype.

This project was collaboratively developed by a team of 4 members as an exploration of Machine Learning and Deep Learning techniques across regression and classification problems.

The project involved data preprocessing, exploratory analysis, model development, experimentation, performance evaluation, and comparison of multiple algorithms.

Drive Link: https://drive.google.com/drive/folders/1gAQaC-0AtG178J6GA3ZrvYjrcWo9NjhG?usp=sharing
