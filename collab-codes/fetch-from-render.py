# Stroke Prediction Analysis and Visualization Dashboard
# Install required packages first
#pip install pandas numpy matplotlib seaborn scikit-learn plotly pickle5

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')
print(plt.style.available)
# Set style parameters
plt.style.use('ggplot')
#plt.style.use('seaborn-whitegrid')
sns.set_style("white")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Load the stroke dataset for analysis
# Sample data - replace with your actual dataset if available
try:
    # Try to load your dataset if it exists
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    print("Loaded existing dataset")
except:
    # If not available, load a sample dataset or create synthetic data
    print("Creating sample dataset for visualization")
    # Sample data creation based on stroke prediction dataset structure
    np.random.seed(42)
    n = 5000
    
    # Create synthetic data that resembles a stroke dataset
    gender = np.random.choice(['Male', 'Female'], size=n)
    age = np.random.normal(loc=55, scale=15, size=n).clip(min=18, max=100)
    hypertension = np.random.choice([0, 1], size=n, p=[0.75, 0.25])
    heart_disease = np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    ever_married = np.random.choice(['Yes', 'No'], size=n)
    work_type = np.random.choice(['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'], 
                              size=n, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    residence_type = np.random.choice(['Urban', 'Rural'], size=n)
    
    # Create slightly correlated data
    avg_glucose_level = np.random.normal(loc=100, scale=40, size=n)
    avg_glucose_level = np.where(hypertension == 1, 
                              avg_glucose_level + np.random.normal(loc=20, scale=10, size=n), 
                              avg_glucose_level)
    
    bmi = np.random.normal(loc=28, scale=7, size=n).clip(min=15, max=50)
    smoking_status = np.random.choice(['never smoked', 'formerly smoked', 'smokes', 'Unknown'], 
                                   size=n, p=[0.5, 0.2, 0.15, 0.15])
    
    # Synthetic stroke values with higher likelihood based on risk factors
    p_stroke = 0.05 + 0.1 * (age > 65) + 0.1 * hypertension + 0.1 * heart_disease + 0.05 * (avg_glucose_level > 140) + 0.05 * (bmi > 30)
    p_stroke = p_stroke.clip(min=0, max=0.9)
    stroke = np.random.binomial(1, p_stroke)
    
    # Create DataFrame
    df = pd.DataFrame({
        'id': range(1, n+1),
        'gender': gender,
        'age': age.astype(int),
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level.round(2),
        'bmi': bmi.round(2),
        'smoking_status': smoking_status,
        'stroke': stroke
    })

# Create images directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')
    print("Created 'images' directory for saving visualizations")

# Function to save matplotlib figures
def save_matplotlib_fig(fig, filename):
    fig.savefig(f'images/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}.png to images folder")

# Function to save plotly figures
def save_plotly_fig(fig, filename):
    fig.write_image(f'images/{filename}.png', scale=2)
    print(f"Saved {filename}.png to images folder")

# Function to get user input
def get_input():
    print("Enter the following details:")

    gender = input("Gender (Male/Female): ").strip()
    age = int(input("Age: ").strip())
    hypertension = int(input("Hypertension (0: No, 1: Yes): ").strip())
    heart_disease = int(input("Heart Disease (0: No, 1: Yes): ").strip())
    ever_married = input("Ever Married (Yes/No): ").strip()
    work_type = input("Work Type (Private/Government/Self-employed/Children): ").strip()
    residence_type = input("Residence Type (Urban/Rural): ").strip()
    avg_glucose_level = float(input("Average Glucose Level: ").strip())
    bmi = float(input("BMI: ").strip())
    smoking_status = input("Smoking Status (never smoked/formerly smoked/smokes): ").strip()

    # Return as a dictionary
    return {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }

# Process the API response for visualization
def process_and_visualize(user_input, api_response, df):
    """
    Process API response and create visualizations
    """
    # Extract prediction results and metrics
    prediction_data = api_response.json()
    model_predictions = prediction_data['predictions']
    final_recommendation = prediction_data['final_recommendation']
    model_metrics = prediction_data['model_metrics']
    
    # Create a DataFrame for model metrics
    metrics_df = pd.DataFrame(model_metrics)
    
    # Start visualization
    print("\n" + "="*50)
    print("\033[1m\033[4mSTROKE PREDICTION ANALYSIS DASHBOARD\033[0m")
    print("="*50)
    
    # 1. Display User Input in table format
    print("\n\033[1m👤 USER PROFILE\033[0m")
    user_df = pd.DataFrame([user_input])
    print("\nUser Profile:")
    print(user_df.to_string(index=False))
    
    # 2. Display Prediction Results
    print("\n\033[1m🔮 PREDICTION RESULTS\033[0m")
    pred_df = pd.DataFrame([model_predictions]).T.reset_index()
    pred_df.columns = ['Model', 'Prediction']
    print("\nModel Predictions:")
    print(pred_df.to_string(index=False))
    
    # 3. Display Final Recommendation
    print(f"\n\033[1m💡 FINAL RECOMMENDATION\033[0m")
    print(f"\033[1m{final_recommendation}\033[0m")
    
    # 4. Display Model Metrics in a styled table
    print("\n\033[1m📊 MODEL PERFORMANCE METRICS\033[0m")
    
    # Format metrics to 2 decimal places
    for col in ['Accuracy', 'F1 Score', 'Precision', 'Recall']:
        metrics_df[col] = metrics_df[col].map(lambda x: f"{x:.2f}")
    
    print("\nModel Metrics:")
    print(metrics_df.to_string(index=False))
    
    # 5. Model Comparison Visualization
    print("\n\033[1m📈 MODEL COMPARISON VISUALIZATION\033[0m")
    
    # Convert metrics back to float for plotting
    plot_metrics = metrics_df.copy()
    for col in ['Accuracy', 'F1 Score', 'Precision', 'Recall']:
        plot_metrics[col] = plot_metrics[col].astype(float)
    
    # Create subplots for metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']
    
    for i, (metric, ax) in enumerate(zip(metrics, axes.flatten())):
        bars = ax.bar(plot_metrics['Model'], plot_metrics[metric], color=colors)
        ax.set_title(f'Model {metric}', fontsize=16, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', fontsize=12)
    
    plt.tight_layout()
    save_matplotlib_fig(fig, 'model_comparison')
    
    # 6. Dataset Exploratory Analysis
    print("\n\033[1m🔍 DATASET EXPLORATORY ANALYSIS\033[0m")
    
    # 6.1 Distribution of Stroke vs Non-Stroke in Dataset
    plt.figure(figsize=(12, 6))
    stroke_counts = df['stroke'].value_counts()
    ax = sns.countplot(x='stroke', data=df, palette=['#99CC99', '#FF9999'])
    plt.title('Distribution of Stroke Cases in Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Stroke Status (0: No Stroke, 1: Stroke)')
    plt.ylabel('Count')
    
    # Add percentage annotations
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height() + 50
        ax.annotate(percentage, (x, y), ha='center', fontsize=12)
    
    plt.grid(axis='y', alpha=0.3)
    save_matplotlib_fig(plt.gcf(), 'stroke_distribution')
    
    # 6.2 Feature Analysis
    print("\n\033[1m📊 FEATURE DISTRIBUTION ANALYSIS\033[0m")
    
    # Age distribution by stroke
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='age', hue='stroke', kde=True, palette=['#99CC99', '#FF9999'])
    plt.title('Age Distribution by Stroke Status', fontsize=16, fontweight='bold')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    save_matplotlib_fig(plt.gcf(), 'age_distribution')
    
    # Glucose level distribution by stroke
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='avg_glucose_level', hue='stroke', kde=True, palette=['#99CC99', '#FF9999'])
    plt.title('Average Glucose Level Distribution by Stroke Status', fontsize=16, fontweight='bold')
    plt.xlabel('Average Glucose Level')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    save_matplotlib_fig(plt.gcf(), 'glucose_distribution')
    
    # BMI distribution by stroke
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='bmi', hue='stroke', kde=True, palette=['#99CC99', '#FF9999'])
    plt.title('BMI Distribution by Stroke Status', fontsize=16, fontweight='bold')
    plt.xlabel('BMI')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    save_matplotlib_fig(plt.gcf(), 'bmi_distribution')
    
    # 7. Correlation Matrix
    print("\n\033[1m🔄 CORRELATION MATRIX\033[0m")
    
    # Prepare data for correlation analysis
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Add encoded categorical features
    cat_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    cat_df = pd.get_dummies(df[cat_features])
    
    # Combine numeric and encoded categorical
    corr_df = pd.concat([numeric_df, cat_df], axis=1)
    
    # Compute correlation matrix
    corr_matrix = corr_df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title('Correlation Matrix of Features', fontsize=18, fontweight='bold')
    plt.tight_layout()
    save_matplotlib_fig(plt.gcf(), 'correlation_matrix')
    
    # 8. Individual Feature Analysis
    print("\n\033[1m🧩 INDIVIDUAL FEATURE ANALYSIS\033[0m")
    
    # Create a figure with subplots for categorical features
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    cat_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'smoking_status']
    
    for i, feature in enumerate(cat_features):
        if i < len(axes):
            # Calculate percentage of stroke cases by feature
            stroke_pct = df.groupby(feature)['stroke'].mean() * 100
            
            # Create bar chart
            sns.barplot(x=stroke_pct.index, y=stroke_pct.values, ax=axes[i], palette='viridis')
            axes[i].set_title(f'Stroke Percentage by {feature.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Stroke Percentage (%)')
            axes[i].set_ylim([0, max(stroke_pct.values) * 1.2])  # Add some padding
            
            # Add percentage labels
            for j, p in enumerate(axes[i].patches):
                axes[i].annotate(f'{p.get_height():.1f}%', 
                             (p.get_x() + p.get_width() / 2., p.get_height() + 0.5),
                             ha='center', fontsize=11)
            
            axes[i].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_matplotlib_fig(fig, 'feature_analysis')
    
    # 9. User Risk Factor Comparison
    print("\n\033[1m⚖️ USER RISK FACTOR COMPARISON\033[0m")
    
    # Create comparison of user values with population averages
    user_numeric = {
        'age': user_input['age'],
        'avg_glucose_level': user_input['avg_glucose_level'],
        'bmi': user_input['bmi'],
        'hypertension': user_input['hypertension'],
        'heart_disease': user_input['heart_disease']
    }
    
    # Calculate population averages
    pop_avg = {
        'age': df['age'].mean(),
        'avg_glucose_level': df['avg_glucose_level'].mean(),
        'bmi': df['bmi'].mean(),
        'hypertension': df['hypertension'].mean(),
        'heart_disease': df['heart_disease'].mean()
    }
    
    # Calculate high-risk averages (people who had strokes)
    high_risk_avg = {
        'age': df[df['stroke'] == 1]['age'].mean(),
        'avg_glucose_level': df[df['stroke'] == 1]['avg_glucose_level'].mean(),
        'bmi': df[df['stroke'] == 1]['bmi'].mean(),
        'hypertension': df[df['stroke'] == 1]['hypertension'].mean(),
        'heart_disease': df[df['stroke'] == 1]['heart_disease'].mean()
    }
    
    # Create comparison DataFrame
    comparison_data = pd.DataFrame({
        'Feature': list(user_numeric.keys()),
        'User Value': list(user_numeric.values()),
        'Population Average': [pop_avg[k] for k in user_numeric.keys()],
        'High Risk Average': [high_risk_avg[k] for k in user_numeric.keys()]
    })
    
    # Format to 2 decimal places
    for col in ['Population Average', 'High Risk Average']:
        comparison_data[col] = comparison_data[col].map(lambda x: f"{x:.2f}")
    
    print("\nUser vs Population Comparison:")
    print(comparison_data.to_string(index=False))
    
    # 10. Radar Chart for Risk Factors
    print("\n\033[1m📡 RISK FACTOR RADAR CHART\033[0m")
    
    # Create radar chart using matplotlib
    features = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']
    feature_names = [f.replace('_', ' ').title() for f in features]
    
    # Normalize values
    def normalize(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val) if max_val > min_val else 0
    
    max_vals = {f: df[f].max() for f in features}
    min_vals = {f: df[f].min() for f in features}
    
    user_norm = [normalize(user_numeric[f], min_vals[f], max_vals[f]) for f in features]
    pop_norm = [normalize(df[f].mean(), min_vals[f], max_vals[f]) for f in features]
    high_risk_norm = [normalize(df[df['stroke'] == 1][f].mean(), min_vals[f], max_vals[f]) for f in features]
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 10))
    
    # Create angles for each feature
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Add feature names
    feature_names += [feature_names[0]]  # Close the loop
    
    # Add data
    user_norm += [user_norm[0]]  # Close the loop
    pop_norm += [pop_norm[0]]
    high_risk_norm += [high_risk_norm[0]]
    
    # Create plot
    ax = fig.add_subplot(111, polar=True)
    
    # Plot data
    ax.plot(angles, user_norm, 'o-', linewidth=2, label='User Values', color='#FF9999')
    ax.fill(angles, user_norm, alpha=0.25, color='#FF9999')
    
    ax.plot(angles, pop_norm, 'o-', linewidth=2, label='Population Average', color='#99CC99')
    ax.fill(angles, pop_norm, alpha=0.25, color='#99CC99')
    
    ax.plot(angles, high_risk_norm, 'o-', linewidth=2, label='High Risk Average', color='#4472C4')
    ax.fill(angles, high_risk_norm, alpha=0.25, color='#4472C4')
    
    # Set feature labels
    ax.set_thetagrids(np.degrees(angles[:-1]), feature_names[:-1])
    
    # Set chart properties
    ax.set_ylim(0, 1)
    ax.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Risk Factor Comparison Radar Chart', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    save_matplotlib_fig(fig, 'risk_factor_radar')
    
    print("\nAll visualizations have been saved to the 'images' folder!")

# Main execution
if __name__ == "__main__":
    # Get user input
    print("\033[1m\033[4mSTROKE PREDICTION SYSTEM\033[0m")
    user_input = get_input()
    
    # Convert to DataFrame for preprocessing
    input_df = pd.DataFrame([user_input])
    
    # Create label encoder or load if available
    try:
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
    except:
        # Create new encoder if not available
        le = LabelEncoder()
        # We'll just use it without fitting since we're sending raw data to API
    
    # Create scaler or load if available
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except:
        # Create new scaler if not available
        scaler = StandardScaler()
        # We'll just use it without fitting since we're sending raw data to API
    
    # Send data to API
    api_url = "https://brain-stroke-prediction-8odp.onrender.com/predict"
    
    try:
        response = requests.post(api_url, json=user_input)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Process and visualize results
            process_and_visualize(user_input, response, df)
        else:
            print(f"Error in API call: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error connecting to API: {str(e)}")
        
        # If API fails, create mock response for visualization demo
        print("\nGenerating mock response for visualization demonstration...")
        
        mock_response = {
            "predictions": {
                "KNN": "No Stroke Risk",
                "Logistic Regression": "No Stroke Risk",
                "Naive Bayes": "Stroke Risk", 
                "Random Forest": "No Stroke Risk",
                "SVM": "No Stroke Risk"
            },
            "final_recommendation": "Best performing model: logistic_regression with accuracy 0.95",
            "model_metrics": [
                {"Model": "logistic_regression", "Accuracy": 0.95, "F1 Score": 0.00, "Precision": 0.00, "Recall": 0.00},
                {"Model": "random_forest", "Accuracy": 0.95, "F1 Score": 0.00, "Precision": 0.00, "Recall": 0.00},
                {"Model": "naive_bayes", "Accuracy": 0.86, "F1 Score": 0.20, "Precision": 0.14, "Recall": 0.34},
                {"Model": "svm", "Accuracy": 0.95, "F1 Score": 0.00, "Precision": 0.00, "Recall": 0.00},
                {"Model": "knn", "Accuracy": 0.94, "F1 Score": 0.00, "Precision": 0.00, "Recall": 0.00}
            ]
        }
        
        # Create mock Response object
        class MockResponse:
            def __init__(self, json_data, status_code):
                self.json_data = json_data
                self.status_code = status_code
                
            def json(self):
                return self.json_data
                
        mock_response_obj = MockResponse(mock_response, 200)
        
        # Call visualization function with mock response
        process_and_visualize(user_input, mock_response_obj, df)

# Create an interactive dashboard using Plotly for web-based visualization
def create_interactive_dashboard(user_input, api_response, df):
    """
    Create an interactive dashboard using Plotly
    """
    # Extract prediction results and metrics
    prediction_data = api_response.json()
    model_predictions = prediction_data['predictions']
    final_recommendation = prediction_data['final_recommendation']
    model_metrics = prediction_data['model_metrics']
    
    # Create a DataFrame for model metrics
    metrics_df = pd.DataFrame(model_metrics)
    
    # Calculate stroke percentage by demographics
    demographic_vars = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'smoking_status']
    stroke_by_demo = {}
    
    for var in demographic_vars:
        stroke_by_demo[var] = df.groupby(var)['stroke'].mean() * 100
    
    # Create the dashboard layout
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Model Performance Metrics', 'Feature Importance',
                       'Age vs. Stroke Risk', 'Glucose Level vs. Stroke Risk',
                       'Stroke Risk by Gender', 'Stroke Risk by Hypertension'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )
    
    # 1. Model Performance Metrics
    metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']
    
    for i, metric in enumerate(metric_names):
        fig.add_trace(
            go.Bar(
                x=metrics_df['Model'],
                y=metrics_df[metric],
                name=metric,
                marker_color=colors[i % len(colors)]
            ),
            row=1, col=1
        )
    
    # 2. Feature Importance (based on correlation with stroke)
    feature_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    feature_importance = {}
    
    for feature in feature_cols:
        # Use absolute correlation as simulated importance
        importance = abs(np.corrcoef(df[feature].values, df['stroke'].values)[0, 1])
        feature_importance[feature] = importance
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    }).sort_values('Importance', ascending=True)
    
    fig.add_trace(
        go.Bar(
            y=importance_df['Feature'],
            x=importance_df['Importance'],
            orientation='h',
            marker_color='#5A9BD4',
            name='Feature Importance'
        ),
        row=1, col=2
    )
    
    # 3. Age vs. Stroke Risk (Scatter plot with trend line)
    age_stroke = df.groupby('age')['stroke'].mean() * 100
    age_stroke_df = pd.DataFrame({'age': age_stroke.index, 'stroke_pct': age_stroke.values})
    
    fig.add_trace(
        go.Scatter(
            x=age_stroke_df['age'],
            y=age_stroke_df['stroke_pct'],
            mode='markers',
            marker=dict(color='#70AD47', size=8),
            name='Age vs. Stroke'
        ),
        row=2, col=1
    )
    
    # Add trend line
    z = np.polyfit(age_stroke_df['age'], age_stroke_df['stroke_pct'], 1)
    p = np.poly1d(z)
    
    fig.add_trace(
        go.Scatter(
            x=age_stroke_df['age'],
            y=p(age_stroke_df['age']),
            mode='lines',
            line=dict(color='red', width=2),
            name='Trend'
        ),
        row=2, col=1
    )
    
    # 4. Glucose Level vs. Stroke Risk
    # Bin glucose levels for better visualization
    df['glucose_bin'] = pd.cut(df['avg_glucose_level'], bins=10)
    glucose_stroke = df.groupby('glucose_bin')['stroke'].mean() * 100
    glucose_stroke_df = pd.DataFrame({
        'glucose_level': [str(interval) for interval in glucose_stroke.index],
        'stroke_pct': glucose_stroke.values
    })
    
    fig.add_trace(
        go.Scatter(
            x=[interval.mid for interval in glucose_stroke.index],
            y=glucose_stroke.values,
            mode='markers',
            marker=dict(color='#FFC000', size=8),
            name='Glucose vs. Stroke'
        ),
        row=2, col=2
    )
    
    # Add trend line for glucose
    valid_indices = ~np.isnan(glucose_stroke.values)
    if sum(valid_indices) > 1:  # Ensure we have at least 2 points for fitting
        x_values = [interval.mid for i, interval in enumerate(glucose_stroke.index) if valid_indices[i]]
        y_values = glucose_stroke.values[valid_indices]
        
        z_glucose = np.polyfit(x_values, y_values, 1)
        p_glucose = np.poly1d(z_glucose)
        
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=p_glucose(x_values),
                mode='lines',
                line=dict(color='red', width=2),
                name='Glucose Trend'
            ),
            row=2, col=2
        )
    
    # 5. Stroke Risk by Gender
    gender_stroke = df.groupby('gender')['stroke'].mean() * 100
    
    fig.add_trace(
        go.Bar(
            x=gender_stroke.index,
            y=gender_stroke.values,
            marker_color='#4472C4',
            name='Gender'
        ),
        row=3, col=1
    )
    
    # 6. Stroke Risk by Hypertension
    hypertension_stroke = df.groupby('hypertension')['stroke'].mean() * 100
    
    fig.add_trace(
        go.Bar(
            x=['No Hypertension', 'Hypertension'],
            y=hypertension_stroke.values,
            marker_color='#ED7D31',
            name='Hypertension'
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text='Stroke Prediction Analysis Dashboard',
        title_font_size=24,
        title_x=0.5,
        height=1000,
        width=1200,
        showlegend=False,
        template='plotly_white'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='Model', row=1, col=1)
    fig.update_yaxes(title_text='Score (0-1)', row=1, col=1)
    
    fig.update_xaxes(title_text='Importance Score', row=1, col=2)
    fig.update_yaxes(title_text='Feature', row=1, col=2)
    
    fig.update_xaxes(title_text='Age', row=2, col=1)
    fig.update_yaxes(title_text='Stroke Risk (%)', row=2, col=1)
    
    fig.update_xaxes(title_text='Average Glucose Level', row=2, col=2)
    fig.update_yaxes(title_text='Stroke Risk (%)', row=2, col=2)
    
    fig.update_xaxes(title_text='Gender', row=3, col=1)
    fig.update_yaxes(title_text='Stroke Risk (%)', row=3, col=1)
    
    fig.update_xaxes(title_text='Hypertension Status', row=3, col=2)
    fig.update_yaxes(title_text='Stroke Risk (%)', row=3, col=2)
    
    # Display User Information and Model Predictions
    user_table = go.Figure(data=[go.Table(
        header=dict(
            values=['Feature', 'User Value'],
            fill_color='#4472C4',
            align='center',
            font=dict(color='white', size=14)
        ),
        cells=dict(
            values=[
                ['Gender', 'Age', 'Hypertension', 'Heart Disease', 'Ever Married', 
                'Work Type', 'Residence Type', 'Avg Glucose Level', 'BMI', 'Smoking Status'],
                [user_input['gender'], user_input['age'], user_input['hypertension'], 
                user_input['heart_disease'], user_input['ever_married'], user_input['work_type'],
                user_input['Residence_type'], user_input['avg_glucose_level'], 
                user_input['bmi'], user_input['smoking_status']]
            ],
            fill_color='white',
            align='left',
            font=dict(color='black', size=12)
        )
    )])
    
    user_table.update_layout(
        title='User Profile Information',
        title_font_size=20,
        title_x=0.5,
        height=400,
        width=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Model Predictions Table
    predictions_table = go.Figure(data=[go.Table(
        header=dict(
            values=['Model', 'Prediction'],
            fill_color='#4472C4',
            align='center',
            font=dict(color='white', size=14)
        ),
        cells=dict(
            values=[
                list(model_predictions.keys()),
                list(model_predictions.values())
            ],
            fill_color=['white', 
                       ['#99CC99' if pred == 'No Stroke Risk' else '#FF9999' 
                        for pred in model_predictions.values()]],
            align='left',
            font=dict(color='black', size=12)
        )
    )])
    
    predictions_table.update_layout(
        title='Model Predictions',
        title_font_size=20,
        title_x=0.5,
        height=400,
        width=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Create ROC Curve (simulated)
    roc_fig = go.Figure()
    
    # Simulated ROC curves for each model
    fpr_base = np.linspace(0, 1, 100)
    
    model_names = [model['Model'] for model in model_metrics]
    accuracies = [model['Accuracy'] for model in model_metrics]
    
    for i, (model_name, accuracy) in enumerate(zip(model_names, accuracies)):
        # Create a simulated ROC curve based on accuracy
        # Higher accuracy = more curve (larger AUC)
        tpr = fpr_base ** (1 / (1 + 2 * (accuracy - 0.5)))
        
        roc_fig.add_trace(go.Scatter(
            x=fpr_base, 
            y=tpr,
            mode='lines',
            name=f"{model_name} (AUC={accuracy:.2f})",
            line=dict(width=2)
        ))
    
    # Add diagonal reference line
    roc_fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Reference',
        line=dict(dash='dash', color='gray')
    ))
    
    roc_fig.update_layout(
        title='ROC Curves (Simulated)',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        width=600,
        legend=dict(x=0.01, y=0.01),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Create confusion matrices visualization (simulated)
    conf_mat_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=tuple(model_names[:4]),  # Show first 4 models
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "heatmap"}, {"type": "heatmap"}]],
    )
    
    # Create simulated confusion matrices based on metrics
    for i, (model_name, accuracy) in enumerate(zip(model_names[:4], accuracies[:4])):
        # Simulated confusion matrix
        tn = 400 * accuracy  # True negatives
        fp = 400 * (1 - accuracy)  # False positives
        fn = 100 * (1 - accuracy)  # False negatives
        tp = 100 * accuracy  # True positives
        
        cm = [[tn, fp], [fn, tp]]
        
        row = i // 2 + 1
        col = i % 2 + 1
        
        conf_mat_fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Predicted No Stroke', 'Predicted Stroke'],
                y=['Actual No Stroke', 'Actual Stroke'],
                colorscale='Blues',
                showscale=False,
                text=[[f'{int(val)}' for val in row] for row in cm],
                texttemplate="%{text}",
                textfont={"size":14}
            ),
            row=row, col=col
        )
    
    conf_mat_fig.update_layout(
        title_text='Confusion Matrices (Simulated)',
        title_font_size=20,
        title_x=0.5,
        height=600,
        width=800
    )
    
    # Risk Factor Radar Chart
    # Create normalized values for radar chart
    features = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']
    
    # Function to normalize values between 0 and 1
    def normalize(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val) if max_val > min_val else 0
    
    # Get min and max values
    max_vals = {f: df[f].max() for f in features}
    min_vals = {f: df[f].min() for f in features}
    
    # Normalize values
    user_numeric = {
        'age': user_input['age'],
        'avg_glucose_level': user_input['avg_glucose_level'],
        'bmi': user_input['bmi'],
        'hypertension': user_input['hypertension'],
        'heart_disease': user_input['heart_disease']
    }
    
    user_norm = [normalize(user_numeric[f], min_vals[f], max_vals[f]) for f in features]
    pop_norm = [normalize(df[f].mean(), min_vals[f], max_vals[f]) for f in features]
    high_risk_norm = [normalize(df[df['stroke'] == 1][f].mean(), min_vals[f], max_vals[f]) for f in features]
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 10))
    
    # Create angles for each feature
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Add feature names
    feature_names = [f.replace('_', ' ').title() for f in features]
    feature_names += [feature_names[0]]  # Close the loop
    
    # Add data
    user_norm += [user_norm[0]]  # Close the loop
    pop_norm += [pop_norm[0]]
    high_risk_norm += [high_risk_norm[0]]
    
    # Create plot
    ax = fig.add_subplot(111, polar=True)
    
    # Plot data
    ax.plot(angles, user_norm, 'o-', linewidth=2, label='User Values', color='#FF9999')
    ax.fill(angles, user_norm, alpha=0.25, color='#FF9999')
    
    ax.plot(angles, pop_norm, 'o-', linewidth=2, label='Population Average', color='#99CC99')
    ax.fill(angles, pop_norm, alpha=0.25, color='#99CC99')
    
    ax.plot(angles, high_risk_norm, 'o-', linewidth=2, label='High Risk Average', color='#4472C4')
    ax.fill(angles, high_risk_norm, alpha=0.25, color='#4472C4')
    
    # Set feature labels
    ax.set_thetagrids(np.degrees(angles[:-1]), feature_names[:-1])
    
    # Set chart properties
    ax.set_ylim(0, 1)
    ax.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Risk Factor Comparison Radar Chart', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    save_matplotlib_fig(fig, 'risk_factor_radar')
    
    print("\nAll visualizations have been saved to the 'images' folder!")
    
    return fig, user_table, predictions_table, roc_fig, conf_mat_fig

# Main execution
if __name__ == "__main__":
    # Get user input
    print("\033[1m\033[4mSTROKE PREDICTION SYSTEM\033[0m")
    print("Enter the following details:")

    user_input = {
        'gender': "Male",
        'age': 85,
        'hypertension': 0,
        'heart_disease': 1,
        'ever_married': "Yes",
        'work_type': "Government",
        'Residence_type': "Urban",
        'avg_glucose_level': 23,
        'bmi': 35,
        'smoking_status': "smokes"
    }
    
    print("Using sample input:", user_input)
    
    # Convert to DataFrame for preprocessing
    input_df = pd.DataFrame([user_input])
    
    # Create label encoder or load if available
    try:
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
    except:
        # Create new encoder if not available
        le = LabelEncoder()
        # We'll just use it without fitting since we're sending raw data to API
    
    # Create scaler or load if available
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except:
        # Create new scaler if not available
        scaler = StandardScaler()
        # We'll just use it without fitting since we're sending raw data to API
    
    # Create mock API response for demonstration
    mock_response = {
        "predictions": {
            "KNN": "No Stroke Risk",
            "Logistic Regression": "No Stroke Risk",
            "Naive Bayes": "Stroke Risk", 
            "Random Forest": "No Stroke Risk",
            "SVM": "No Stroke Risk"
        },
        "final_recommendation": "Best performing model: logistic_regression with accuracy 0.95",
        "model_metrics": [
            {"Model": "logistic_regression", "Accuracy": 0.95, "F1 Score": 0.00, "Precision": 0.00, "Recall": 0.00},
            {"Model": "random_forest", "Accuracy": 0.95, "F1 Score": 0.00, "Precision": 0.00, "Recall": 0.00},
            {"Model": "naive_bayes", "Accuracy": 0.86, "F1 Score": 0.20, "Precision": 0.14, "Recall": 0.34},
            {"Model": "svm", "Accuracy": 0.95, "F1 Score": 0.00, "Precision": 0.00, "Recall": 0.00},
            {"Model": "knn", "Accuracy": 0.94, "F1 Score": 0.00, "Precision": 0.00, "Recall": 0.00}
        ]
    }
    
    # Create mock Response object
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code
            
        def json(self):
            return self.json_data
            
    mock_response_obj = MockResponse(mock_response, 200)
    
    # Process and visualize results
    print("\n\n" + "="*50)
    print("\033[1m\033[4mSTROKE PREDICTION VISUALIZATION DASHBOARD\033[0m")
    print("="*50 + "\n")
    
    # Call standard visualization
    process_and_visualize(user_input, mock_response_obj, df)
    
    # Create and show interactive dashboard
    print("\n\n" + "="*50)
    print("\033[1m\033[4mINTERACTIVE STROKE PREDICTION DASHBOARD\033[0m")
    print("="*50 + "\n")
    
    dashboard_outputs = create_interactive_dashboard(user_input, mock_response_obj, df)
    
    # In a Jupyter notebook, you'd display these with:
    # for fig in dashboard_outputs:
    #     fig.show()
    
    print("Visualization complete! Run this code in a Jupyter notebook or Google Colab to see all visualizations.")