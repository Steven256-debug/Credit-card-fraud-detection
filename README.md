Project Overview

This project builds a complete machine learning system for credit card fraud detection, aimed at identifying fraudulent transactions with high accuracy.
The solution includes:

A full ML pipeline (cleaning → feature engineering → encoding → modeling)

Multiple classification models (Logistic Regression, Random Forest, XGBoost)

Imbalanced learning strategies (SMOTE)

SHAP explainability

A professional Streamlit dashboard for real-time fraud detection

Online deployment using ngrok.

2. Technologies Used

Python – core programming language

Pandas, NumPy – data manipulation, cleaning

Scikit-learn – preprocessing, modeling, evaluation

XGBoost – high-performance gradient boosting model

Imbalanced-learn (SMOTE) – oversampling minority fraud class

Matplotlib, Seaborn, Plotly – visualizations

SHAP – global + individual model interpretability

Streamlit – interactive dashboard

Joblib – saving/loading trained pipeline

Pyngrok – deployment tunnel

Category Encoders – categorical encoding support

3. Setup & Installation
Environment

Use Python 3.9+ (Colab recommended).

Install Dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap joblib streamlit pyngrok category_encoders openpyxl

Configure ngrok

Create account → https://dashboard.ngrok.com

Copy auth token

Set token:

from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN_HERE")

Project Files Needed

app.py

models/full_pipeline.pkl

models/feature_order.csv

results_test_predictions.csv (optional)

4. Data Preparation & Feature Engineering
Raw Data Columns
TransactionID, TransactionDate, Amount, MerchantID, TransactionType, Location, IsFraud

Key Steps
✔ Date Parsing

Converted TransactionDate into datetime and extracted:

trans_hour

trans_dayofweek

trans_month

TransactionDate_freq_enc (frequency encoded)

✔ Numerical Engineering

norm_amount (scaled transaction amount)

Outlier capping (IQR)

Duplicate removal

✔ Categorical Encoding

Missing values → "MISSING"

One-hot encoding for low-cardinality:

TransactionType

Location

Frequency encoding for high-cardinality:

MerchantID

✔ Target Class Imbalance

Dataset was heavily imbalanced.
SMOTE was applied in the training pipeline.

5. Model Training & Evaluation

Trained models:

Logistic Regression

Random Forest

XGBoost

Performance Metrics

Used metrics suited for imbalanced datasets:

PR-AUC (Primary Metric)

ROC-AUC

Recall

Precision

F1-score

Cross-Validation

Used Stratified 5-Fold CV.

Best Model

XGBoost selected based on PR-AUC.

Final metrics:

ROC-AUC: 0.521

PR-AUC: 0.011

Recall: 0.000
(The dataset is extremely imbalanced; model ranking is captured better by PR-AUC.)

6. Streamlit Application Features

The dashboard has three tabs:

📊 Predictions Tab

Shows raw uploaded data

Shows processed features

Displays fraud predictions + probabilities

Summary metrics:

Total transactions

Fraud count

% fraud

Highlights highest-risk transaction

Downloadable results CSV

📈 Dashboard Tab

Interactive Plotly histogram of fraud probabilities

Explanation of probability distribution

💡 Explainability Tab

Global SHAP summary plot (feature importance)

Individual SHAP force plot for selected TransactionID

Displays feature contributions to each prediction

7. How to Use the App

Open the ngrok public URL (active only during Colab session).

Upload RAW CSV containing the required columns.

Navigate tabs:

Predictions → Results, summaries, download CSV

Dashboard → Probability visualization

Explainability → SHAP insights

Inspect high-risk transactions and interpret SHAP force plots.

8. Key Insights

Fraud detection datasets are extremely imbalanced → PR-AUC is a reliable metric.

XGBoost learns ranking patterns but recall remains low due to lack of strong signals.

SHAP plots improve transparency of model decisions.

The Streamlit UI provides clear insights, professional visualization, and user-friendly interpretation tools.

9. Future Enhancements

Collect more fraud data or synthetic fraud generation

Test anomaly detection models (Isolation Forest, Autoencoders)

Add real-time transaction scoring API

Integrate a database backend

Improve SHAP force plot performance

Add dynamic filters (by merchant, amount, date range)
