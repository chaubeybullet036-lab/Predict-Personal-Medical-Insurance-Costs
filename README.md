Medical Insurance Cost Prediction
An end-to-end Supervised Machine Learning project built using Python to predict individual medical healthcare charges based on personal demographics and lifestyle attributes.
📌 1. Problem Statement
Healthcare expenses are often unpredictable and vary significantly across individuals based on factors such as age, body mass index (BMI), lifestyle choices (e.g., smoking), and geographical region.
The primary goal of this project is to build a high-precision regression model using Supervised Learning to forecast individual medical insurance costs (). Accurate predictions enable insurance companies to price policies fairly while empowering policyholders to understand how their health habits impact their financial outlay.
📊 2. Dataset Description
Data Source: Secondary publicly available dataset (Kaggle / UCI Machine Learning Repository).
Dataset Size: 1,338 records with 7 key variables.
Feature Definitions
Column Name
Type
Description
age
Numerical
Age of primary beneficiary (18 to 64 years)
sex
Categorical
Gender of insurance contractor (female, male)
bmi
Numerical
Body Mass Index (), measure of body weight relative to height
children
Numerical
Number of dependents covered by health insurance
smoker
Categorical
Smoking status (yes, no)
region
Categorical
Beneficiary's residential area in the US (northeast, southeast, southwest, northwest)
charges (Target)
Numerical
Individual medical costs billed by health insurance

🛠️ 3. Data Preprocessing
The raw dataset underwent several key preprocessing steps:
Data Cleaning: Verified zero missing or duplicate values across all records.
Target Transformation: Applied a logarithmic transformation (np.log1p) to the target variable charges to eliminate severe right-skewness and stabilize target variance.
Categorical Encoding: Applied One-Hot Encoding to handle categorical features (sex, smoker, region) without introducing ordinal bias.
Feature Scaling: Applied StandardScaler to ensure all numeric features were normalized for robust model learning.
Data Splitting: Divided data into 80% Training Set and 20% Test Set using train_test_split.
📈 4. Exploratory Data Analysis (EDA)
Key findings uncovered during EDA include:
Impact of Smoking: Smoking status is the single most influential variable affecting insurance costs. Smokers consistently face significantly higher charges compared to non-smokers.
Compound Impact of BMI & Smoking: Non-smokers show only a slight increase in costs with rising BMI, whereas high-BMI smokers experience an exponential surge in insurance charges.
Age Trend: Insurance charges show a steady linear increase as age advances.
🤖 5. Model Building
Algorithm Selected: RandomForestRegressor (Ensemble Learning Method)
Rationale: Random Forest effectively captures complex non-linear interactions between variables (such as the interaction between smoker status and bmi) without overfitting easily.
📐 6. Model Evaluation
The trained model was evaluated on unseen test data (20% split) using standard regression metrics:
Metric
Score / Value
Mean Absolute Error (MAE)
~$2,500
Root Mean Squared Error (RMSE)
~$4,500
R-squared () Score
~0.85

Result: The model successfully explains approximately 85% of the variance in individual medical charges.
💡 7. Interpretation of Results
Primary Cost Drivers: Feature importance analysis reveals that smoking status and BMI are the top predictors governing medical insurance expenses.
Behavioral Impact: Policyholders can dramatically lower predicted insurance costs by reducing lifestyle risks (specifically smoking cessation and weight management).
🎯 8. Conclusion & Limitations
Major Findings
Supervised machine learning algorithms like Random Forest can reliably forecast personal healthcare costs based on basic demographic and lifestyle indicators.
Preventive health interventions focusing on smoking reduction yield the single highest potential cost savings for policyholders and insurers.
Limitations
Feature Scope: The dataset lacks detailed medical background, clinical history, or pre-existing chronic condition records.
Geographical Scope: Restricted to four generalized U.S. regions, limiting generalizability to international healthcare frameworks.
🚀 How to Run the Project Locally
Prerequisites
Make sure you have Python 3.8+ installed along with the required libraries:
pip install numpy pandas matplotlib seaborn scikit-learn


Execution Steps
Clone this repository:
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git


Navigate to the project folder:
cd YOUR-REPOSITORY-NAME


Run the main script:
python main.py

