# Credit-card-fraud-detection
Project Overview
Purpose
This project aims to develop a robust machine learning solution for detecting credit card fraud. The primary objective is to identify fraudulent transactions with high accuracy to minimize financial losses and protect customers from unauthorized activities. The solution focuses on building a predictive model and presenting its insights through an interactive web application.

Technologies Used
Python: The core programming language for data manipulation, model development, and application building.
Pandas: Utilized for efficient data loading, cleaning, preprocessing, and feature engineering.
Numpy: Essential for numerical operations and array manipulation.
Scikit-learn: Employed for machine learning model development, including data splitting, preprocessing (e.g., StandardScaler), and various classification algorithms (LogisticRegression, RandomForestClassifier).
XGBoost: A powerful gradient boosting library used for building highly performant classification models (XGBClassifier).
Imbalanced-learn: Used to handle imbalanced datasets (e.g., SMOTE for oversampling) common in fraud detection.
Matplotlib & Seaborn: For static data visualization during exploratory data analysis and model evaluation.
Plotly Express: Integrated for creating interactive data visualizations within the Streamlit application, such as the fraud probability distribution.
SHAP (SHapley Additive exPlanations): Employed for model interpretability, providing insights into feature importance both globally and for individual predictions.
Streamlit: The framework used to build the interactive web application (dashboard) for real-time fraud prediction and explanation.
Joblib: For saving and loading the trained machine learning pipeline and model.
Pyngrok: Used to create secure tunnels to expose the local Streamlit application to the internet, making it accessible via a public URL.
Category Encoders: Potentially used for advanced categorical feature encoding (though basic one-hot and frequency encoding were primarily used in this notebook).
Solution Summary
The solution involves a comprehensive machine learning pipeline designed to detect credit card fraud:

Data Ingestion & Cleaning: Raw transaction data (CSV) is uploaded and undergoes initial cleaning, including handling duplicates and parsing TransactionDate robustly.
Feature Engineering: New features are created, such as norm_amount (scaled transaction amount), date-derived features (trans_hour, trans_dayofweek, trans_month), and frequency encodings (TransactionDate_freq_enc, MerchantID_freq_enc). Missing value flags and outlier capping are also applied.
Categorical Encoding: Categorical features like TransactionType and Location are one-hot encoded, while MerchantID is frequency encoded to manage high cardinality, ensuring consistency with the model's expectations.
Model Training: Multiple classification models (LogisticRegression, RandomForest, XGBoost) are trained on the preprocessed data, incorporating SMOTE to address class imbalance. The models are cross-validated, and the best-performing model (based on PR-AUC) is selected and saved.
Interactive Streamlit Dashboard: A user-friendly Streamlit application is developed to demonstrate the fraud detection capabilities. Users can upload new raw CSV data, and the app performs the same preprocessing steps as the training pipeline. The dashboard then displays:
Samples of raw and processed data.
Transaction-level predictions (fraud status and probability).
Summary statistics of predictions using st.metric.
An interactive Plotly histogram of fraud probabilities.
A fraud probability gauge highlighting the highest-risk transaction.
Global and individual SHAP explanations for model interpretability.
A downloadable CSV of all prediction results.
This end-to-end solution provides both a predictive model and a transparent, interactive interface for understanding and utilizing its fraud detection capabilities.

Detail Setup and Installation Instructions
Subtask:
Provide clear step-by-step instructions on how to set up the environment, install required libraries, and configure ngrok for deploying the Streamlit application.

Detail Setup and Installation Instructions
Subtask:
Provide clear step-by-step instructions on how to set up the environment, install required libraries, and configure ngrok for deploying the Streamlit application.

Instructions
Python Environment Setup:

Python Version: Ensure you are using Python 3.9 or higher.
Recommended Environment: This notebook is designed for Google Colab, which comes with many dependencies pre-installed and manages environments effectively. Alternatively, you can use a local Python environment (e.g., venv or conda).
Install Required Libraries:

All necessary Python libraries can be installed using a single pip install command. Execute the following in your environment:
!pip install -q pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap joblib streamlit pyngrok category_encoders openpyxl
Obtain ngrok Authentication Token:

ngrok is used to create a secure tunnel to your local Streamlit application, making it accessible via a public URL.
Go to the ngrok dashboard and sign up for a free account.
After signing up, navigate to the "Your Authtoken" section (usually found under "Getting Started" or directly at ngrok.com/dashboard/your-authtoken).
Copy your authentication token.
Set ngrok Authentication Token:

In your environment (e.g., a code cell in Google Colab), replace "YOUR_NGROK_AUTH_TOKEN_HERE" with the token you copied:
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN_HERE")
Acquire Project Files:

To get the necessary files, such as app.py, models/full_pipeline.pkl, models/feature_order.csv, etc., you would typically clone a Git repository (e.g., git clone [repository_url]) or download them from a provided link into your working directory.
Describe Data Preparation and Feature Engineering
Subtask:
Summarize the data loading process, the initial EDA findings (e.g., class imbalance), and the feature engineering steps applied to prepare the data for modeling.

Data Preparation and Feature Engineering Summary
1. Data Loading and Initial Overview
The process began by loading the raw credit card transaction data from a CSV file into a pandas DataFrame. Initial exploratory data analysis (EDA) revealed the data's shape and data types, highlighting that TransactionDate was an object type needing parsing and that the IsFraud target variable was highly imbalanced, with a significant majority of transactions being legitimate (class 0).

2. Date Parsing and Feature Extraction
To leverage temporal information, the TransactionDate column was robustly parsed into a datetime format. From this parsed date, several new time-based features were extracted:

trans_hour: The hour of the transaction.
trans_dayofweek: The day of the week of the transaction.
trans_month: The month of the transaction.
TransactionDate_freq_enc: A frequency encoding representing the normalized count of transactions occurring on that specific date.
3. Numerical Feature Engineering and Cleaning
Numerical features underwent further processing:

norm_amount: A normalized version of the Amount feature was created using StandardScaler to bring it to a common scale.
Outlier Capping: An Interquartile Range (IQR)-based capping method was applied to Amount and norm_amount to mitigate the influence of extreme outliers.
Duplicate Handling: Initial checks for duplicate rows were performed, and any exact duplicates were removed to ensure data integrity.
4. Categorical Feature Encoding
To prepare categorical features for modeling, a selective encoding strategy was employed:

Missing Value Imputation: For categorical columns (TransactionType, Location, MerchantID), any missing values were explicitly filled with the string 'MISSING' to ensure they could be processed without errors.
Low-Cardinality Encoding: Columns with low cardinality (i.e., a small number of unique values), specifically TransactionType and Location, were transformed using one-hot encoding. This creates new binary columns for each category, preventing the model from inferring spurious ordinal relationships.
High-Cardinality Encoding: For MerchantID, which typically has many unique values, frequency encoding was used. This method replaces each category with its normalized occurrence frequency, which can be effective in capturing information from high-cardinality features while avoiding the dimensionality explosion of one-hot encoding.
Data Preparation and Feature Engineering Summary
1. Data Loading and Initial Overview
The process began by loading the raw credit card transaction data from a CSV file into a pandas DataFrame. Initial exploratory data analysis (EDA) revealed the data's shape and data types, highlighting that TransactionDate was an object type needing parsing and that the IsFraud target variable was highly imbalanced, with a significant majority of transactions being legitimate (class 0).

2. Date Parsing and Feature Extraction
To leverage temporal information, the TransactionDate column was robustly parsed into a datetime format. From this parsed date, several new time-based features were extracted:

trans_hour: The hour of the transaction.
trans_dayofweek: The day of the week of the transaction.
trans_month: The month of the transaction.
TransactionDate_freq_enc: A frequency encoding representing the normalized count of transactions occurring on that specific date.
3. Numerical Feature Engineering and Cleaning
Numerical features underwent further processing:

norm_amount: A normalized version of the Amount feature was created using StandardScaler to bring it to a common scale.
Outlier Capping: An Interquartile Range (IQR)-based capping method was applied to Amount and norm_amount to mitigate the influence of extreme outliers.
Duplicate Handling: Initial checks for duplicate rows were performed, and any exact duplicates were removed to ensure data integrity.
4. Categorical Feature Encoding
To prepare categorical features for modeling, a selective encoding strategy was employed:

Missing Value Imputation: For categorical columns (TransactionType, Location, MerchantID), any missing values were explicitly filled with the string 'MISSING' to ensure they could be processed without errors.
Low-Cardinality Encoding: Columns with low cardinality (i.e., a small number of unique values), specifically TransactionType and Location, were transformed using one-hot encoding. This creates new binary columns for each category, preventing the model from inferring spurious ordinal relationships.
High-Cardinality Encoding: For MerchantID, which typically has many unique values, frequency encoding was used. This method replaces each category with its normalized occurrence frequency, which can be effective in capturing information from high-cardinality features while avoiding the dimensionality explosion of one-hot encoding.
Data Preparation and Feature Engineering Summary
1. Data Loading and Initial Overview
The process began by loading the raw credit card transaction data from a CSV file into a pandas DataFrame. Initial exploratory data analysis (EDA) revealed the data's shape and data types, highlighting that TransactionDate was an object type needing parsing and that the IsFraud target variable was highly imbalanced, with a significant majority of transactions being legitimate (class 0).

2. Date Parsing and Feature Extraction
To leverage temporal information, the TransactionDate column was robustly parsed into a datetime format. From this parsed date, several new time-based features were extracted:

trans_hour: The hour of the transaction.
trans_dayofweek: The day of the week of the transaction.
trans_month: The month of the transaction.
TransactionDate_freq_enc: A frequency encoding representing the normalized count of transactions occurring on that specific date.
3. Numerical Feature Engineering and Cleaning
Numerical features underwent further processing:

norm_amount: A normalized version of the Amount feature was created using StandardScaler to bring it to a common scale.
Outlier Capping: An Interquartile Range (IQR)-based capping method was applied to Amount and norm_amount to mitigate the influence of extreme outliers.
Duplicate Handling: Initial checks for duplicate rows were performed, and any exact duplicates were removed to ensure data integrity.
4. Categorical Feature Encoding
To prepare categorical features for modeling, a selective encoding strategy was employed:

Missing Value Imputation: For categorical columns (TransactionType, Location, MerchantID), any missing values were explicitly filled with the string 'MISSING' to ensure they could be processed without errors.
Low-Cardinality Encoding: Columns with low cardinality (i.e., a small number of unique values), specifically TransactionType and Location, were transformed using one-hot encoding. This creates new binary columns for each category, preventing the model from inferring spurious ordinal relationships.
High-Cardinality Encoding: For MerchantID, which typically has many unique values, frequency encoding was used. This method replaces each category with its normalized occurrence frequency, which can be effective in capturing information from high-cardinality features while avoiding the dimensionality explosion of one-hot encoding.
Data Preparation and Feature Engineering Summary
1. Data Loading and Initial Overview
The process began by loading the raw credit card transaction data from a CSV file into a pandas DataFrame. Initial exploratory data analysis (EDA) revealed the data's shape and data types, highlighting that TransactionDate was an object type needing parsing and that the IsFraud target variable was highly imbalanced, with a significant majority of transactions being legitimate (class 0).

2. Date Parsing and Feature Extraction
To leverage temporal information, the TransactionDate column was robustly parsed into a datetime format. From this parsed date, several new time-based features were extracted:

trans_hour: The hour of the transaction.
trans_dayofweek: The day of the week of the transaction.
trans_month: The month of the transaction.
TransactionDate_freq_enc: A frequency encoding representing the normalized count of transactions occurring on that specific date.
3. Numerical Feature Engineering and Cleaning
Numerical features underwent further processing:

norm_amount: A normalized version of the Amount feature was created using StandardScaler to bring it to a common scale.
Outlier Capping: An Interquartile Range (IQR)-based capping method was applied to Amount and norm_amount to mitigate the influence of extreme outliers.
Duplicate Handling: Initial checks for duplicate rows were performed, and any exact duplicates were removed to ensure data integrity.
4. Categorical Feature Encoding
To prepare categorical features for modeling, a selective encoding strategy was employed:

Missing Value Imputation: For categorical columns (TransactionType, Location, MerchantID), any missing values were explicitly filled with the string 'MISSING' to ensure they could be processed without errors.
Low-Cardinality Encoding: Columns with low cardinality (i.e., a small number of unique values), specifically TransactionType and Location, were transformed using one-hot encoding. This creates new binary columns for each category, preventing the model from inferring spurious ordinal relationships.
High-Cardinality Encoding: For MerchantID, which typically has many unique values, frequency encoding was used. This method replaces each category with its normalized occurrence frequency, which can be effective in capturing information from high-cardinality features while avoiding the dimensionality explosion of one-hot encoding.
Outline Model Training and Evaluation
Subtask:
Explain the machine learning models trained, the evaluation metrics used (especially for imbalanced datasets like PR-AUC), and the selection of the best model.

Models Trained and Evaluation
In this project, three different machine learning models were trained to detect credit card fraud:

Logistic Regression: A linear model used for binary classification, often serving as a strong baseline.
Random Forest Classifier: An ensemble learning method that constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
XGBoost Classifier: An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework.
Handling Class Imbalance
Given the highly imbalanced nature of fraud detection datasets (where fraudulent transactions are very rare compared to legitimate ones), SMOTE (Synthetic Minority Over-sampling Technique) from the imblearn library was integrated into each model's pipeline. SMOTE works by creating synthetic samples from the minority class, helping the models to learn more effectively from the infrequent fraud examples without simply duplicating existing ones.

Evaluation Metrics
For a robust evaluation, especially with imbalanced datasets, the following metrics were used:

PR-AUC (Average Precision Score): This was the primary metric for model selection. PR-AUC is particularly important for imbalanced datasets because it focuses on the positive class (fraud). A high PR-AUC indicates that the model is performing well in identifying fraud cases while minimizing false positives. Unlike ROC-AUC, which can be misleading on imbalanced data, PR-AUC provides a more realistic view of classifier performance when the positive class is rare.
ROC-AUC (Receiver Operating Characteristic Area Under the Curve): Measures the ability of a classifier to distinguish between classes. While useful, it can be less informative than PR-AUC for imbalanced datasets.
Recall: Measures the proportion of actual positive cases that are correctly identified by the model. High recall is critical in fraud detection to catch as many fraudulent transactions as possible.
Precision: Measures the proportion of positive identifications that were actually correct. High precision means fewer legitimate transactions are flagged as fraud.
F1-Score: The harmonic mean of precision and recall, providing a balance between the two.
Cross-Validation Strategy
To ensure the models' robustness and generalize well to unseen data, Stratified K-Fold Cross-Validation (with n_splits=5, shuffle=True, random_state=42) was employed. This method ensures that each fold maintains the same proportion of the target class (fraudulent vs. legitimate transactions) as the overall dataset, which is crucial for imbalanced problems.

Best Model Selection
The best model was selected based on the highest PR-AUC (Average Precision Score) during cross-validation and subsequent evaluation on the test set. This choice prioritizes the model's ability to effectively identify fraud with a low false positive rate, which is paramount in real-world fraud detection systems.

Selected Best Model and Final Performance
After cross-validation and evaluation on the held-out test set, XGBoost emerged as the best performing model based on its PR-AUC score. Its final performance on the test set was:

ROC-AUC: 0.521
PR-AUC: 0.011
Recall: 0.000
While the recall for the positive class is 0 (meaning it didn't identify any fraud correctly in the test set after SMOTE), the PR-AUC provides a better indication of how well the model ranks potential fraud cases. The model's PR-AUC of 0.011 is slightly above random (which would be the proportion of fraud in the dataset), indicating some learning, but also highlights the extreme difficulty of fraud detection on this particular, highly imbalanced dataset with the current features and model parameters.

Explain Streamlit Dashboard Features
Subtask:
Detail all the interactive features implemented in the Streamlit application, including the tabbed interface, sidebar, metrics, interactive charts, fraud probability gauge, SHAP explanations (global and individual), and downloadable results.

Explain Streamlit Dashboard Features
Overview of the Streamlit Application Structure
The Streamlit application is designed with an intuitive, interactive dashboard to facilitate credit card fraud detection and analysis. Its core structure leverages st.tabs to organize content into distinct sections: 'Predictions', 'Dashboard', and 'Explainability'. A st.sidebar is also incorporated to provide general information about the application.

Predictions Tab (📊)
The 'Predictions' tab serves as the primary interface for users to upload data, view preprocessing steps, and examine the model's predictions. It includes the following sections:

Raw Uploaded Data Sample (📝): Displays the initial raw data as uploaded by the user, providing a quick overview of the input.
Processed Data Sample (for model input) (⚙️): Shows how the raw data is transformed and engineered into features that the machine learning model can understand and process. This step is crucial for transparency in the preprocessing pipeline.
Prediction Results Sample (✨): Presents a sample of the output, including the original TransactionID, Amount, Location, TransactionType, alongside the model's Predicted_IsFraud (0 for legitimate, 1 for fraudulent) and the Fraud_Probability for each transaction.
Prediction Summary (📋): Utilizes st.metric widgets within a st.columns layout to provide key summary statistics at a glance:
Total Transactions: The total number of transactions processed.
Fraudulent Transactions (⚠️): The count of transactions predicted as fraudulent.
% Fraudulent: The percentage of transactions predicted as fraudulent.
Highest Risk Transaction (⚠️): This section highlights the transaction with the highest predicted fraud probability. It uses st.metric to display the maximum fraud probability and a st.progress bar to visually represent the risk level. Detailed information for this specific transaction is also displayed.
Download Prediction Results CSV: A st.download_button is available, allowing users to easily download the complete results_df (which includes original data, predicted fraud status, and probabilities) as a CSV file for further analysis or record-keeping.
Dashboard Tab (📈)
The 'Dashboard' tab is dedicated to visualizing the overall model performance and insights:

Distribution of Fraud Probabilities (📉): This section features an interactive histogram, powered by plotly.express, that visualizes the distribution of predicted fraud probabilities across all uploaded transactions. Unlike static plots, this interactive chart allows users to zoom, pan, and hover for detailed insights into the probability spread.
Implications of Probability Distribution (🔎): Provides a textual explanation of what the shape of the fraud probability distribution indicates about the model's confidence and the nature of the dataset's fraud cases.
Explainability Tab (💡)
For deep diving into why the model makes certain predictions, the 'Explainability' tab offers:

Global Feature Importance (SHAP Summary Plot) (🌳): Displays a SHAP (SHapley Additive exPlanations) summary plot, which visually represents the overall importance and impact of each feature on the model's predictions. This provides a general understanding of which features are most influential in determining fraud.
Individual Transaction Explanation (🔎): This interactive feature allows users to select a specific TransactionID from a dropdown menu. Upon selection, the application generates and displays a detailed SHAP force plot for that individual transaction using st.components.v1.html. The force plot breaks down how each feature contributed to the final fraud probability for that particular transaction, offering a transparent, in-depth explanation of the model's decision-making process.
Visual Enhancements
Throughout the application, Streamlit emojis/icons (e.g., 🛡️, 📊, 📈, 💡, ⚠️) are strategically used in st.title, st.header, st.subheader, and st.metric labels to enhance visual appeal and user engagement. The layout also employs st.columns in sections like the Prediction Summary to arrange metrics side-by-side, improving readability and information density.

Explain Streamlit Dashboard Features
Overview of the Streamlit Application Structure
The Streamlit application is designed with an intuitive, interactive dashboard to facilitate credit card fraud detection and analysis. Its core structure leverages st.tabs to organize content into distinct sections: 'Predictions', 'Dashboard', and 'Explainability'. A st.sidebar is also incorporated to provide general information about the application.

Predictions Tab (📊)
The 'Predictions' tab serves as the primary interface for users to upload data, view preprocessing steps, and examine the model's predictions. It includes the following sections:

Raw Uploaded Data Sample (📝): Displays the initial raw data as uploaded by the user, providing a quick overview of the input.
Processed Data Sample (for model input) (⚙️): Shows how the raw data is transformed and engineered into features that the machine learning model can understand and process. This step is crucial for transparency in the preprocessing pipeline.
Prediction Results Sample (✨): Presents a sample of the output, including the original TransactionID, Amount, Location, TransactionType, alongside the model's Predicted_IsFraud (0 for legitimate, 1 for fraudulent) and the Fraud_Probability for each transaction.
Prediction Summary (📋): Utilizes st.metric widgets within a st.columns layout to provide key summary statistics at a glance:
Total Transactions: The total number of transactions processed.
Fraudulent Transactions (⚠️): The count of transactions predicted as fraudulent.
% Fraudulent: The percentage of transactions predicted as fraudulent.
Highest Risk Transaction (⚠️): This section highlights the transaction with the highest predicted fraud probability. It uses st.metric to display the maximum fraud probability and a st.progress bar to visually represent the risk level. Detailed information for this specific transaction is also displayed.
Download Prediction Results CSV: A st.download_button is available, allowing users to easily download the complete results_df (which includes original data, predicted fraud status, and probabilities) as a CSV file for further analysis or record-keeping.
Dashboard Tab (📈)
The 'Dashboard' tab is dedicated to visualizing the overall model performance and insights:

Distribution of Fraud Probabilities (📉): This section features an interactive histogram, powered by plotly.express, that visualizes the distribution of predicted fraud probabilities across all uploaded transactions. Unlike static plots, this interactive chart allows users to zoom, pan, and hover for detailed insights into the probability spread.
Implications of Probability Distribution (🔎): Provides a textual explanation of what the shape of the fraud probability distribution indicates about the model's confidence and the nature of the dataset's fraud cases.
Explainability Tab (💡)
For deep diving into why the model makes certain predictions, the 'Explainability' tab offers:

Global Feature Importance (SHAP Summary Plot) (🌳): Displays a SHAP (SHapley Additive exPlanations) summary plot, which visually represents the overall importance and impact of each feature on the model's predictions. This provides a general understanding of which features are most influential in determining fraud.
Individual Transaction Explanation (🔎): This interactive feature allows users to select a specific TransactionID from a dropdown menu. Upon selection, the application generates and displays a detailed SHAP force plot for that individual transaction using st.components.v1.html. The force plot breaks down how each feature contributed to the final fraud probability for that particular transaction, offering a transparent, in-depth explanation of the model's decision-making process.
Visual Enhancements
Throughout the application, Streamlit emojis/icons (e.g., 🛡️, 📊, 📈, 💡, ⚠️) are strategically used in st.title, st.header, st.subheader, and st.metric labels to enhance visual appeal and user engagement. The layout also employs st.columns in sections like the Prediction Summary to arrange metrics side-by-side, improving readability and information density.

Provide Usage Instructions for the App
Subtask:
Give clear instructions on how to interact with the Streamlit app, upload data, navigate through the dashboard, and interpret the predictions and explanations.

Provide Usage Instructions for the App
Subtask:
Give clear instructions on how to interact with the Streamlit app, upload data, navigate through the dashboard, and interpret the predictions and explanations.

Instructions
Welcome to the Credit Card Fraud Detection application! Follow these steps to use the dashboard:

Accessing the Application:

Open your web browser and navigate to the public URL: https://bianca-bedfast-barrett.ngrok-free.dev.
Uploading Data:

On the main page, locate the "Upload RAW CSV" file uploader (📂 icon).
Click on it and select a CSV file from your local machine.
Important: Your CSV file must contain the following columns: TransactionID, TransactionDate, Amount, MerchantID, TransactionType, Location, and IsFraud.
Once uploaded, the app will display "Uploaded columns: [...]" to confirm the columns found in your file.
Navigating the Dashboard:

After a successful upload and processing, the application will display three main tabs:
"Predictions 📊": This tab provides an overview of the raw data, processed data, individual prediction results, and key summary metrics.
"Dashboard 📈": This tab focuses on visual analytics, primarily showing the distribution of predicted fraud probabilities.
"Explainability 💡": This tab delves into the model's decision-making process using SHAP plots, both global and individual.
Interpreting Predictions (Predictions Tab 📊):

Raw Uploaded Data Sample 📝: Shows the first few rows of the CSV file you uploaded, exactly as it was read.
Processed Data Sample (for model input) ⚙️: Displays how your raw data was transformed into features suitable for the machine learning model. This includes new features like norm_amount, trans_hour, frequency encodings, and one-hot encoded categorical variables.
Prediction Results Sample ✨: Presents a summary table of transaction IDs, original amounts, locations, transaction types, and the model's output: Predicted_IsFraud (0 for legitimate, 1 for fraud) and Fraud_Probability (the model's confidence score).
Prediction Summary 📋: This section uses st.metric widgets to highlight key statistics:
Total Transactions: The total number of transactions processed.
Fraudulent Transactions ⚠️: The count of transactions predicted as fraudulent.
% Fraudulent: The percentage of fraudulent transactions out of the total.
Highest Risk Transaction ⚠️: This section pinpoints the transaction with the highest predicted fraud probability. It shows the Max Fraud Probability as a metric and a Fraud Risk Level progress bar, along with the detailed raw data for that specific high-risk transaction.
Interpreting Dashboard Visualizations (Dashboard Tab 📈):

Distribution of Fraud Probabilities 📉: This interactive Plotly histogram visualizes the spread of fraud probabilities for all uploaded transactions. You can hover over bars for details, zoom, and pan.
Implications of Probability Distribution 🔎: Read the accompanying text for insights into what the shape of the probability distribution tells you about the model's performance and the rarity of fraud.
Interpreting Explainability (Explainability Tab 💡):

Global Feature Importance (SHAP Summary Plot) 🌳: This bar plot shows which features, on average, had the most impact on the model's predictions across a sample of your data. Longer bars indicate higher importance, and the color often indicates the feature's value (e.g., high value, low value).
Individual Transaction Explanation 🔎: Use the dropdown menu to select a specific Transaction ID. The app will then generate a SHAP force plot for that transaction. This plot visually breaks down how each feature's value pushed the prediction from the base value (average prediction) to the final prediction for that specific transaction. Features pushing the prediction higher (towards fraud) are typically shown in red, and those pushing it lower (towards legitimate) are in blue.
Downloading Results:

In the "Predictions 📊" tab, locate the "Download Prediction Results CSV" button. Click it to download a CSV file containing all the original transaction data, along with the Predicted_IsFraud and Fraud_Probability columns.
Summarize Key Insights and Future Enhancements
Subtask:
Present the main insights derived from the model and identify potential next steps or future enhancements for the project.

Summary: Key Insights and Future Enhancements
Key Insights
The app.py script was extensively refactored to incorporate a tab-based navigation (st.tabs for "Predictions", "Dashboard", "Explainability") and a sidebar (st.sidebar) for application information, significantly improving the user interface and content organization.
Key summary statistics, including "Total Transactions", "Fraudulent Transactions", and "% Fraudulent", are now displayed using st.metric widgets within a three-column layout, enhancing visibility and readability.
The static fraud probability distribution plot was upgraded to an interactive Plotly histogram, offering a more dynamic and engaging visualization experience.
A fraud probability gauge (st.progress) and detailed information for the highest-risk transaction were implemented, allowing users to quickly identify and investigate potentially fraudulent activities.
The explainability section was enhanced to include not only a global SHAP summary plot (sampled up to 500 rows for performance) but also an interactive individual SHAP force plot. This allows users to select a specific TransactionID and understand the feature contributions to its prediction.
A download button was added, enabling users to export the complete prediction results, including original transaction data, predicted fraud status, and fraud probabilities, as a CSV file.
Throughout the application, Streamlit emojis and icons were strategically integrated into titles, headers, and labels to improve visual appeal and user engagement.
The Streamlit application was successfully restarted and redeployed via ngrok after each set of modifications, ensuring continuous access to the latest features through the public URL.
Future Enhancements
The integration of SHAP plots, both global and individual, significantly enhances the interpretability of the fraud detection model, allowing users to understand the key drivers behind each prediction, which is crucial for building trust and enabling data-driven decision-making in fraud investigation.
The use of st.metric, interactive Plotly charts, and visual enhancements makes the dashboard more user-friendly and highlights key information effectively.
Future enhancements could include implementing user-specific filters or drill-down capabilities within the dashboard for the prediction results, allowing for deeper investigation of specific fraudulent or high-risk transactions.
