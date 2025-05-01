import pickle
import pandas as pd
import numpy as np

# Example input
data_input = {
    'gender': 'Male',
    'age': 67,
    'hypertension': 0,
    'heart_disease': 1,
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 228.69,
    'bmi': 36,
    'smoking_status': 'formerly smoked'
}

# Convert to DataFrame
input_df = pd.DataFrame([data_input])

# Load LabelEncoder and transform categorical variables
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    input_df[col] = le.fit_transform(input_df[col].astype(str))  # Ensure match with training encoding

# Load Scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Scale inputs
scaled_input = scaler.transform(input_df)

# Load model performance metrics
metrics_df = pd.read_csv("model_performance_metrics.csv")

# Load models
model_files = [
    "logistic_regression_model.pkl",
    "random_forest_model.pkl",
    "naive_bayes_model.pkl",
    "svm_model.pkl",
    "knn_model.pkl"
]

# Predict and show metrics
print("\nPrediction Results with Metrics:\n")
for model_file in model_files:
    model_name = model_file.replace("_model.pkl", "")
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    prediction = model.predict(scaled_input)[0]
    result = "Stroke Risk" if prediction == 1 else "No Stroke Risk"

    # Fetch metrics
    metrics = metrics_df[metrics_df['Model'] == model_name].to_dict('records')
    if metrics:
        metrics_info = metrics[0]
        print(f"Model: {model_name} => Prediction: {result}")
        print(f"  Accuracy: {metrics_info['Accuracy']:.2f}")
        print(f"  Precision: {metrics_info['Precision']:.2f}")
        print(f"  Recall: {metrics_info['Recall']:.2f}")
        print(f"  F1 Score: {metrics_info['F1 Score']:.2f}")
        print(f"  MSE: {metrics_info['MSE']:.2f}")
        print(f"  R2 Score: {metrics_info['R2 Score']:.2f}\n")
    else:
        print(f"Model: {model_name} => Prediction: {result} (Metrics not found)\n")

# ---- Highlight Best Models Based on Recall and F1 Score ----

# Sort by Recall (most important for medical detection)
ranked_metrics = metrics_df.sort_values(by=["Recall", "F1 Score"], ascending=False).reset_index(drop=True)

print("\n\U0001F4CA Ranked Models by Recall & F1 Score (for Stroke Detection):\n")
for i, row in ranked_metrics.iterrows():
    highlight = "⬅️ Top Choice" if i == 0 else ""
    print(f"Rank {i+1}: {row['Model']}")
    print(f"  Accuracy: {row['Accuracy']:.2f}")
    print(f"  Precision: {row['Precision']:.2f}")
    print(f"  Recall: {row['Recall']:.2f}")
    print(f"  F1 Score: {row['F1 Score']:.2f}")
    print(f"  MSE: {row['MSE']:.2f}")
    print(f"  R2 Score: {row['R2 Score']:.2f} {highlight}\n")