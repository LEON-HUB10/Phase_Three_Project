Terry Stops Analysis and Predictive Modeling
📌 Project Overview

This project analyzes Terry Stops data to understand the factors influencing arrest decisions. Using exploratory data analysis (EDA) and machine learning models, the project seeks to identify patterns and improve predictive accuracy of arrest outcomes.

The key goal is to balance interpretability and predictive performance while addressing challenges such as class imbalance (far fewer arrests compared to non-arrests).

🗂 Dataset

Original Distribution

Arrests (1): 5,870

Non-Arrests (0): 45,830

Balanced Training Set (SMOTE applied)

Arrests (1): 45,830

Non-Arrests (0): 45,830

SMOTE (Synthetic Minority Oversampling Technique) was applied only to the training data to handle class imbalance, while the test set remained imbalanced for realistic evaluation.

🔍 Exploratory Data Analysis (EDA)

Conducted univariate, bivariate, and multivariate analysis.

Identified correlations between demographics, frisk/search flags, and arrest likelihood.

Applied chi-square tests to confirm significant associations.

⚙️ Models Implemented

Logistic Regression

Tuned hyperparameters and optimized classification threshold.

Strength: Simple, interpretable baseline model.

Weakness: Struggled with minority class recall.

XGBoost Classifier

Tuned using RandomizedSearchCV.

Threshold optimization applied for better minority class detection.

Strength: Higher accuracy and weighted F1 score, capturing more complex patterns.

📊 Model Performance
Logistic Regression (Tuned + Optimized Threshold)

Accuracy: 0.67

Recall (Arrests): 0.55

Precision (Arrests): 0.18

F1-Score (Arrests): 0.28

XGBoost (Tuned + Optimized Threshold)

Accuracy: 0.75

Recall (Arrests): 0.43

Precision (Arrests): 0.21

F1-Score (Arrests): 0.29

✅ Observations

Both models suffer from low precision for arrests (class 1) due to real-world class imbalance.

Logistic Regression provided interpretability but weaker predictive performance.

XGBoost outperformed Logistic Regression with higher accuracy and weighted F1 score, offering a better balance between detecting arrests and minimizing false positives.

📌 Why XGBoost Was Selected

Best overall performance on accuracy (0.75) and weighted F1 (0.79).

Handles non-linear relationships and interactions effectively.

More robust to imbalanced data (especially after SMOTE + threshold tuning).

Provides interpretable feature importance for actionable insights.

🚀 Recommendations

Overfitting control: Apply regularization (max_depth, min_child_weight, subsample).

Cost-sensitive learning: Assign higher misclassification costs to minority class.

Ensemble approaches: Combine XGBoost with other models for balanced predictions.

Threshold adjustment: Calibrate thresholds depending on whether recall or precision is prioritized.

Deployment monitoring: Continuously monitor performance in real-world data, as class imbalance may shift.

🛠️ Tech Stack

Python (Pandas, NumPy, Scikit-learn, Imbalanced-learn, XGBoost)

Jupyter Notebook for interactive analysis

Matplotlib/Seaborn for visualization
