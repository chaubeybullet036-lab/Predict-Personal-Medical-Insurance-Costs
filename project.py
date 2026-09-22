# ==========================================
# IMPORT REQUIRED LIBRARIES
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# STEP 2 & 3: LOAD DATA & DATA PREPROCESSING
# ==========================================
# Load dataset directly via URL
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)

print("--- Initial Data Snapshot ---")
print(df.head())
print("\n--- Summary Info ---")
print(df.info())

# 3.1 Handle Missing Values
print("\nMissing values per column:\n", df.isnull().sum())
# Note: This dataset is clean with zero missing values.

# 3.2 Handling Outliers & Skewness
# Target variable 'charges' is right-skewed, so log transform helps normalize distribution
df['log_charges'] = np.log1p(df['charges'])

# 3.3 Categorical Encoding (One-Hot Encoding)
df_encoded = pd.get_dummies(df.drop(columns=['charges']), columns=['sex', 'smoker', 'region'], drop_first=True)

# Separate Features (X) and Target (y)
X = df_encoded.drop(columns=['log_charges'])
y = df_encoded['log_charges']

# Split Data into Train (80%) and Test (20%) Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Standardizing numerical features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
plt.figure(figsize=(14, 5))

# Plot 1: Target Variable Distribution
plt.subplot(1, 2, 1)
sns.histplot(df['charges'], kde=True, color='teal')
plt.title("Distribution of Medical Charges")
plt.xlabel("Charges ($)")

# Plot 2: Smoker vs Charges (Key Relationship)
plt.subplot(1, 2, 2)
sns.boxplot(x='smoker', y='charges', data=df, palette='Set2')
plt.title("Impact of Smoking on Insurance Charges")
plt.xlabel("Smoker")
plt.ylabel("Charges ($)")

plt.tight_layout()
plt.show()

# Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df_encoded.corr(), annot=True, fmt='.2f', cmap='Blues')
plt.title("Feature Correlation Heatmap")
plt.show()

# ==========================================
# STEP 5: MODEL BUILDING
# ==========================================
# Model Choice: Random Forest Regressor (Handles non-linear relationships & feature interactions)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Model Predictions (Exponents reversed back to original dollar values)
y_pred_log = model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

# ==========================================
# STEP 6: MODEL EVALUATION
# ==========================================
mae = mean_absolute_error(y_test_orig, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
r2 = r2_score(y_test_orig, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"R-squared (R²) Score: {r2:.4f} ({r2*100:.2f}% variance explained)")

# Feature Importance Analysis
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- Feature Importance ---")
print(feature_importances)