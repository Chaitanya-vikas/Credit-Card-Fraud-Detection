import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix


# Load the dataset
# Ensure 'creditcard.csv' is in the same folder as this script
df = pd.read_csv('creditcard.csv')

# Quick check to see if data loaded correctly
print("First 5 rows of data:")
print(df.head())
print(f"\nDataset shape: {df.shape}")

# Scale Time and Amount using RobustScaler (good for outliers)
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

# Drop the original un-scaled columns to avoid duplication
df.drop(['Time','Amount'], axis=1, inplace=True)

# Reorder columns to put scaled features first (optional, for tidiness)
scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']
df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

print("Data scaled successfully.")



# Separate Features (X) and Target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data
# Stratify=y ensures we have the same % of fraud in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to the Training set ONLY
# We never touch the test set with synthetic data!
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"Original fraud count in Train: {y_train.sum()}")
print(f"Resampled fraud count in Train: {y_train_res.sum()}")


# Train the Model
# n_jobs=-1 uses all your computer cores to speed it up
print("Training Random Forest Model... (this may take a moment)")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_res, y_train_res)

print("Model training complete.")



# Predict on Test Data
y_pred = rf.predict(X_test)

# Print Classification Report
print("--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Fraud Detection Confusion Matrix')
plt.show()

import joblib

# Save the model to a file
joblib.dump(rf, 'fraud_model.pkl') 

# Save the scaler too (crucial for new data!)
joblib.dump(rob_scaler, 'scaler.pkl')

print("Model and Scaler saved successfully!")