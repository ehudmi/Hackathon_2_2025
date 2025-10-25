# Hackathon_2_2025
Machine learning model for medical costs

# Analysis of Medical Insurance Costs

## Introduction and Objective

This notebook analyzes the "Medical Cost Personal Datasets" to explore factors influencing individual medical insurance costs. The objective is to identify patterns and insights that can inform pricing strategies and risk assessment in the health insurance sector.

## Data Loading and Initial Exploration

The dataset was loaded into a pandas DataFrame. Initial checks confirmed that there were no missing values. The dataset includes features such as age, sex, BMI, number of children, smoking status, region, and the target variable, medical charges.

## Exploratory Data Analysis (EDA)

Comprehensive EDA was performed to understand variable distributions and relationships with medical charges:

*   **Age:** A general positive correlation was observed, with charges tending to increase with age. Scatter plots revealed distinct bands, particularly when considering smoking status.
*   **Smoking Status:** Smoking was found to be the most significant factor influencing charges, with smokers having substantially higher mean charges than non-smokers (statistically significant difference confirmed by t-test, p < 0.001).
*   **Sex:** A small but statistically significant difference in mean charges was observed between males and females (females having slightly lower mean charges, p < 0.05).
*   **BMI:** A weak but statistically significant positive correlation was found between BMI and charges (correlation coefficient ≈ 0.2, p < 0.001).
*   **Number of Children:** ANOVA test indicated a statistically significant difference in mean charges across different numbers of children (p < 0.05).
*   **Region:** ANOVA test also showed a statistically significant difference in mean charges across different regions, although this difference was marginal (p < 0.05).

Multivariate analysis, particularly focusing on age and smoking status, highlighted the strong impact of smoking on charges across all age groups.

## Data Preprocessing

Data preprocessing involved:

*   Splitting the data into training and testing sets (80/20 split).
*   Using a `ColumnTransformer` to apply different transformations to different feature types:
    *   Ordinal Encoding for binary features (`sex`, `smoker`).
    *   One-Hot Encoding for the categorical feature (`region`).
    *   Standard Scaling for numerical features (`age`, `bmi`, `children`).

## Model Building and Evaluation

Several regression models were evaluated using a pipeline that included the preprocessing steps. The models assessed were:

*   Linear Regression
*   Lasso
*   ElasticNet
*   Linear SVR
*   SVR
*   Random Forest Regressor
*   XGBoost Regressor

Initial evaluation on the test set using R2, MAE, and MSE metrics showed that the **Random Forest Regressor** and **XGBoost Regressor** performed best, with Random Forest having slightly better scores across the board (R2 ≈ 0.86, MAE ≈ 2550).

## Model Tuning and Selection

Hyperparameter tuning was performed on the best-performing model, the Random Forest Regressor, using `GridSearchCV` with a focus on optimizing the Mean Absolute Error (using negative MAE scoring).

The tuning process identified the following optimal hyperparameters:
*   `max_depth`: 10
*   `min_samples_leaf`: 4
*   `min_samples_split`: 2
*   `n_estimators`: 100

Evaluating the tuned Random Forest model on the test set yielded improved performance:
*   **Test set R2 score:** 0.8775
*   **Test set MAE:** 2427.77
*   **Test set MSE:** 19010468.42

The tuned Random Forest model demonstrates a strong ability to predict medical charges based on the provided features.

## Feature Importance

Analysis of the feature importances from the tuned Random Forest model revealed the relative contribution of each feature to the predictions:

*   **Smoker** was overwhelmingly the most important feature.
*   **BMI** and **Age** were the next most important features.
*   **Children** and **Region** had considerably lower importance.

This confirms that smoking status is the primary driver of medical costs in this dataset, followed by BMI and age.

## Conclusion and Insights

The analysis highlights that **smoking status, BMI, and age** are the most significant factors influencing medical insurance charges. While sex, number of children, and region show statistically significant relationships with charges, their impact is considerably less compared to smoking, BMI, and age in the developed model.

The tuned Random Forest Regressor model provides a robust tool for predicting medical costs, with an average absolute error of approximately 2428 USD on unseen data. These findings can be valuable for understanding risk factors and potentially informing insurance pricing and health intervention strategies. Further analysis could explore interaction effects in more detail and potentially investigate the unexplained variance in the middle charge band.
