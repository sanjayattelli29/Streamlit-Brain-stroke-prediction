import pandas as pd
import numpy as np
import os
import pickle
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load data
df = pd.read_csv("cleaned_brain_stroke.csv")

# Fill missing BMI values using interpolation
df['bmi'] = df['bmi'].interpolate(method='linear')

# Encode categorical variables
le = LabelEncoder()
for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    df[col] = le.fit_transform(df[col].astype(str))

# Split features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Save test sets for API evaluation
pd.DataFrame(X_test, columns=X.columns).to_csv("X_test.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

# Models to train
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(),
    "naive_bayes": GaussianNB(),
    "svm": SVC(probability=True),
    "knn": KNeighborsClassifier()
}

# Save scaler and label encoder
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Train and evaluate
metrics = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save model
    with open(f"{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": recall,
        "F1 Score": f1,
        "MSE": mse,
        "R2 Score": r2
    })

# Save metrics
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("model_performance_metrics.csv", index=False)

# Save metrics.pkl
with open("model_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

# Zip all model files
with zipfile.ZipFile("trained_models_bundle.zip", 'w') as zipf:
    files_to_zip = [
        "X_test.csv", "y_test.csv", "scaler.pkl", "label_encoder.pkl",
        "model_performance_metrics.csv", "model_metrics.pkl"
    ] + [f"{name}_model.pkl" for name in models.keys()]
    for file in files_to_zip:
        zipf.write(file)

print("Training complete. All models and files saved to 'trained_models_bundle.zip'.")
