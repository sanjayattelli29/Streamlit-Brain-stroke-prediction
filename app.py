import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Brain Stroke Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Custom CSS
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #2196F3;
        --secondary-color: #1976D2;
        --background-color: #0e1117;
        --card-background: #1e1e1e;
        --text-color: #ffffff;
        --text-muted: #a0a0a0;
        --border-color: #2d2d2d;
        --success-color: #00C851;
        --warning-color: #ffbb33;
        --danger-color: #ff4444;
    }

    /* Main Container */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Card Styling */
    div[data-testid="stVerticalBlock"] > div {
        background-color: var(--card-background);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }

    /* Dataframe Styling */
    .dataframe {
        background-color: var(--card-background) !important;
        color: var(--text-color) !important;
        font-family: 'Arial', sans-serif !important;
        border-collapse: collapse !important;
        width: 100% !important;
    }

    .dataframe th {
        background-color: var(--primary-color) !important;
        color: white !important;
        padding: 12px !important;
        border: none !important;
    }

    .dataframe td {
        padding: 12px !important;
        border-bottom: 1px solid var(--border-color) !important;
    }

    .dataframe tr:hover {
        background-color: rgba(33, 150, 243, 0.1) !important;
    }

    /* Button Styling */
    .stButton > button {
        width: 100%;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        color: white;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.3);
    }

    /* Form Fields */
    .stSelectbox [data-baseweb="select"] {
        background-color: var(--card-background) !important;
        z-index: 1000 !important;
    }

    .stSelectbox span {
        color: var(--text-color) !important;
    }

    .stSelectbox [data-baseweb="popover"] {
        background-color: var(--card-background) !important;
        z-index: 1001 !important;
    }

    .stSelectbox [data-baseweb="option"] {
        background-color: var(--card-background) !important;
        color: var(--text-color) !important;
    }

    .stSelectbox [data-baseweb="option"]:hover {
        background-color: var(--primary-color) !important;
        color: white !important;
    }

    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: var(--primary-color) !important;
    }

    /* Plot Styling */
    .js-plotly-plot {
        background-color: var(--card-background) !important;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--card-background);
        padding: 0.5rem;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: var(--text-color);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--primary-color);
        color: white;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }

    /* Info Box */
    .stAlert {
        background-color: var(--card-background);
        color: var(--text-color);
        border: 1px solid var(--border-color);
        padding: 1rem;
        border-radius: 8px;
    }

    /* Fix for empty spaces */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Number Input */
    .stNumberInput input {
        color: var(--text-color) !important;
        background-color: var(--card-background) !important;
        border: 1px solid var(--border-color) !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: var(--primary-color) !important;
    }
    </style>
""", unsafe_allow_html=True)

def home_page():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h1 style='color: var(--text-color); font-size: 2.5rem; margin-bottom: 1rem;'>
                    🧠 Brain Stroke Prediction System
                </h1>
                <p style='color: var(--text-muted); font-size: 1.1rem; line-height: 1.6;'>
                    Advanced machine learning system to predict stroke risk using multiple models
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Get Started"):
            st.session_state['page'] = 'input'
            st.rerun()

def input_page():
    st.markdown("""
        <h1 style='color: var(--text-color); font-size: 2rem; margin-bottom: 2rem;'>
            📝 Patient Information
        </h1>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container():
                gender = st.selectbox("Gender", ["Male", "Female"])
                age = st.number_input("Age", min_value=0, max_value=120, value=30)
                hypertension = st.selectbox("Hypertension", ["No", "Yes"])
                heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
                ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        
        with col2:
            with st.container():
                work_type = st.selectbox("Work Type", ["Private", "Government", "Self-employed", "Children"])
                residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
                avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, value=100.0)
                bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
                smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

        submitted = st.form_submit_button("Predict")
        if submitted:
            with st.spinner('Analyzing patient data...'):
                data = {
                    'gender': gender,
                    'age': age,
                    'hypertension': 1 if hypertension == "Yes" else 0,
                    'heart_disease': 1 if heart_disease == "Yes" else 0,
                    'ever_married': ever_married,
                    'work_type': work_type,
                    'Residence_type': residence_type,
                    'avg_glucose_level': avg_glucose_level,
                    'bmi': bmi,
                    'smoking_status': smoking_status
                }
                
                try:
                    response = requests.post(
                        "https://brain-stroke-prediction-8odp.onrender.com/predict",
                        json=data
                    )
                    if response.status_code == 200:
                        response_data = response.json()
                        # Store both the input data and the response
                        st.session_state['prediction_data'] = response_data
                        st.session_state['input_data'] = data
                        st.session_state['page'] = 'results'
                        st.rerun()
                    else:
                        st.error("Error in prediction. Please try again.")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

def results_page():
    if 'prediction_data' not in st.session_state or 'input_data' not in st.session_state:
        st.error("No prediction data available")
        return

    data = st.session_state['prediction_data']
    input_data = st.session_state['input_data']
    metrics_df = pd.DataFrame(data['model_metrics'])
    best_model = metrics_df.loc[metrics_df['Recall'].idxmax()]
    
    # Convert input data for visualization
    input_df = pd.DataFrame([input_data])
    
    # Header Section with tabs
    st.markdown("""
        <h1 style='color: var(--text-color); font-size: 2.5rem; margin-bottom: 2rem; text-align: center;'>
            🎯 Comprehensive Analysis Dashboard
        </h1>
    """, unsafe_allow_html=True)

    # Create tabs for different analyses
    tabs = st.tabs([
        "📊 Overview", 
        "🤖 Model Analysis", 
        "📈 Metrics Deep Dive",
        "🎯 Feature Analysis",
        "📉 Trend Analysis"
    ])

    # Tab 1: Overview
    with tabs[0]:
        st.markdown("""
            <div style='background-color: var(--card-background); padding: 1.5rem; border-radius: 10px; border: 1px solid var(--border-color); margin-bottom: 2rem;'>
                <h2 style='color: var(--text-color); margin-bottom: 1rem;'>Quick Overview</h2>
            </div>
        """, unsafe_allow_html=True)

        # Summary metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Best Model",
                value=best_model['Model'],
                delta=f"Recall: {best_model['Recall']:.3f}"
            )
        
        with col2:
            avg_accuracy = metrics_df['Accuracy'].mean()
            st.metric(
                label="Average Accuracy",
                value=f"{avg_accuracy:.3f}",
                delta=f"{(avg_accuracy - metrics_df['Accuracy'].min()):.3f} from lowest"
            )
        
        with col3:
            model_count = len(metrics_df)
            st.metric(
                label="Models Evaluated",
                value=str(model_count),
                delta="Ensemble Prediction"
            )

        # Model Predictions Table
        st.markdown("""
            <div style='background-color: var(--card-background); padding: 1.5rem; border-radius: 10px; border: 1px solid var(--border-color); margin-top: 2rem;'>
                <h3 style='color: var(--text-color); margin-bottom: 1rem;'>Model Predictions</h3>
            </div>
        """, unsafe_allow_html=True)
        
        predictions = pd.DataFrame({
            'Model': list(data['predictions'].keys()),
            'Prediction': list(data['predictions'].values())
        })
        
        # Style the predictions dataframe
        styled_predictions = predictions.style.apply(
            lambda x: ['background-color: rgba(33, 150, 243, 0.1)' if x['Prediction'] == 1 else '' for i in x], 
            axis=1
        )
        st.dataframe(styled_predictions, use_container_width=True)

        # Model Metrics Table
        st.markdown("""
            <div style='background-color: var(--card-background); padding: 1.5rem; border-radius: 10px; border: 1px solid var(--border-color); margin-top: 2rem;'>
                <h3 style='color: var(--text-color); margin-bottom: 1rem;'>Model Metrics</h3>
            </div>
        """, unsafe_allow_html=True)

        # Create a styled metrics table
        metrics_table = metrics_df.copy()
        
        # Round all numeric columns to 3 decimal places
        numeric_cols = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        metrics_table[numeric_cols] = metrics_table[numeric_cols].round(3)

        # Define the styling function
        def style_metrics(val):
            if isinstance(val, float):
                color = 'rgba(0, 200, 81, 0.2)' if val > 0.8 else 'rgba(255, 68, 68, 0.1)'
                return f'background-color: {color}'
            return ''

        # Apply styling
        styled_metrics = metrics_table.style.apply(lambda x: [style_metrics(v) for v in x])
        
        # Add tooltips
        tooltips = {
            'Accuracy': 'Overall prediction accuracy',
            'F1 Score': 'Harmonic mean of precision and recall',
            'Precision': 'True positives / (True positives + False positives)',
            'Recall': 'True positives / (True positives + False negatives)'
        }
        
        styled_metrics = styled_metrics.format(precision=3)
        
        # Display the styled table
        st.dataframe(
            styled_metrics,
            use_container_width=True,
            column_config={
                "Model": st.column_config.TextColumn(
                    "Model",
                    help="Machine Learning model used for prediction"
                ),
                "Accuracy": st.column_config.NumberColumn(
                    "Accuracy",
                    help=tooltips['Accuracy'],
                    format="%.3f"
                ),
                "F1 Score": st.column_config.NumberColumn(
                    "F1 Score",
                    help=tooltips['F1 Score'],
                    format="%.3f"
                ),
                "Precision": st.column_config.NumberColumn(
                    "Precision",
                    help=tooltips['Precision'],
                    format="%.3f"
                ),
                "Recall": st.column_config.NumberColumn(
                    "Recall",
                    help=tooltips['Recall'],
                    format="%.3f"
                )
            }
        )

        # Add a legend for the metrics table
        st.markdown("""
            <div style='background-color: var(--card-background); padding: 1rem; border-radius: 5px; border: 1px solid var(--border-color); margin-top: 1rem; font-size: 0.9em;'>
                <h4 style='color: var(--text-color); margin-bottom: 0.5rem;'>📊 Metrics Legend</h4>
                <ul style='color: var(--text-muted); list-style-type: none; padding-left: 0;'>
                    <li>🎯 <strong>Accuracy</strong>: Overall prediction accuracy of the model</li>
                    <li>⚖️ <strong>F1 Score</strong>: Balance between precision and recall</li>
                    <li>📈 <strong>Precision</strong>: Accuracy of positive predictions</li>
                    <li>🔍 <strong>Recall</strong>: Ability to find all positive cases</li>
                </ul>
                <p style='color: var(--text-muted); margin-top: 0.5rem; font-style: italic;'>
                    * Green highlighting indicates strong performance (>0.8)
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Tab 2: Model Analysis
    with tabs[1]:
        st.markdown("""
            <div style='background-color: var(--card-background); padding: 1.5rem; border-radius: 10px; border: 1px solid var(--border-color); margin-bottom: 2rem;'>
                <h2 style='color: var(--text-color); margin-bottom: 1rem;'>Model Performance Analysis</h2>
            </div>
        """, unsafe_allow_html=True)

        # Radar Chart for Model Comparison
        fig_radar = go.Figure()
        
        for model in metrics_df['Model']:
            model_metrics = metrics_df[metrics_df['Model'] == model].iloc[0]
            fig_radar.add_trace(go.Scatterpolar(
                r=[model_metrics['Accuracy'], model_metrics['Precision'], 
                   model_metrics['Recall'], model_metrics['F1 Score']],
                theta=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                fill='toself',
                name=model
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)

        # Model Performance Heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].values,
            x=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            y=metrics_df['Model'],
            colorscale='Viridis',
            text=metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].round(3).values,
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False
        ))

        fig_heatmap.update_layout(
            title='Model Performance Heatmap',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

    # Tab 3: Metrics Deep Dive
    with tabs[2]:
        st.markdown("""
            <div style='background-color: var(--card-background); padding: 1.5rem; border-radius: 10px; border: 1px solid var(--border-color); margin-bottom: 2rem;'>
                <h2 style='color: var(--text-color); margin-bottom: 1rem;'>Metrics Analysis</h2>
            </div>
        """, unsafe_allow_html=True)

        # Metrics Distribution
        fig_dist = go.Figure()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for metric in metrics:
            fig_dist.add_trace(go.Box(
                y=metrics_df[metric],
                name=metric,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))

        fig_dist.update_layout(
            title='Metrics Distribution Across Models',
            yaxis_title='Score',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )

        st.plotly_chart(fig_dist, use_container_width=True)

        # Metrics Correlation
        corr_matrix = metrics_df[metrics].corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=metrics,
            y=metrics,
            colorscale='RdBu',
            text=corr_matrix.round(3).values,
            texttemplate='%{text}',
            textfont={"size": 12},
            zmin=-1,
            zmax=1
        ))

        fig_corr.update_layout(
            title='Metrics Correlation Matrix',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )

        st.plotly_chart(fig_corr, use_container_width=True)

    # Tab 4: Feature Analysis
    with tabs[3]:
        st.markdown("""
            <div style='background-color: var(--card-background); padding: 1.5rem; border-radius: 10px; border: 1px solid var(--border-color); margin-bottom: 2rem;'>
                <h2 style='color: var(--text-color); margin-bottom: 1rem;'>Feature Analysis</h2>
            </div>
        """, unsafe_allow_html=True)

        # Feature values visualization
        feature_importance = pd.DataFrame({
            'Feature': list(input_data.keys()),
            'Value': list(input_data.values())
        })

        fig_features = go.Figure(data=[
            go.Bar(
                x=feature_importance['Feature'],
                y=feature_importance['Value'],
                marker_color='#2196F3',
                text=feature_importance['Value'],
                textposition='auto',
            )
        ])

        fig_features.update_layout(
            title='Input Feature Values',
            xaxis_title='Features',
            yaxis_title='Value',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )

        st.plotly_chart(fig_features, use_container_width=True)

        # Feature importance scores (if available in the API response)
        if 'feature_importance' in data:
            feat_imp = pd.DataFrame(data['feature_importance'])
            fig_imp = go.Figure(data=[
                go.Bar(
                    x=feat_imp['feature'],
                    y=feat_imp['importance'],
                    marker_color='#00C851',
                    text=feat_imp['importance'].round(3),
                    textposition='auto',
                )
            ])

            fig_imp.update_layout(
                title='Feature Importance Scores',
                xaxis_title='Features',
                yaxis_title='Importance Score',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )

            st.plotly_chart(fig_imp, use_container_width=True)

    # Tab 5: Trend Analysis
    with tabs[4]:
        st.markdown("""
            <div style='background-color: var(--card-background); padding: 1.5rem; border-radius: 10px; border: 1px solid var(--border-color); margin-bottom: 2rem;'>
                <h2 style='color: var(--text-color); margin-bottom: 1rem;'>Performance Trends</h2>
            </div>
        """, unsafe_allow_html=True)

        # Model Performance Trends
        fig_trends = go.Figure()
        
        for metric in metrics:
            fig_trends.add_trace(go.Scatter(
                x=metrics_df['Model'],
                y=metrics_df[metric],
                mode='lines+markers',
                name=metric,
                line=dict(width=2),
                marker=dict(size=8)
            ))

        fig_trends.update_layout(
            title='Model Performance Trends',
            xaxis_title='Models',
            yaxis_title='Score',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            yaxis=dict(range=[0, 1])
        )

        st.plotly_chart(fig_trends, use_container_width=True)

        # Parallel Coordinates Plot
        fig_parallel = go.Figure(data=
            go.Parcoords(
                line=dict(color=metrics_df['Accuracy'],
                         colorscale='Viridis'),
                dimensions=[
                    dict(range=[0, 1],
                         label='Accuracy',
                         values=metrics_df['Accuracy']),
                    dict(range=[0, 1],
                         label='Precision',
                         values=metrics_df['Precision']),
                    dict(range=[0, 1],
                         label='Recall',
                         values=metrics_df['Recall']),
                    dict(range=[0, 1],
                         label='F1 Score',
                         values=metrics_df['F1 Score'])
                ]
            )
        )

        fig_parallel.update_layout(
            title='Parallel Coordinates Plot - Model Performance',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )

        st.plotly_chart(fig_parallel, use_container_width=True)

    # Final Recommendation Section
    st.markdown("## Final Recommendation")
    
    # Get the model with highest recall
    best_recall_model = metrics_df.loc[metrics_df['Recall'].idxmax()]
    
    # Model Selection Card
    st.markdown(f"""
        <div style='background-color: rgba(33, 150, 243, 0.1); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='margin-bottom: 10px; color: white;'>🎯 Model Selection</h3>
            <p style='font-size: 16px; color: white;'>
                Based on our analysis, we recommend using the <strong>{best_recall_model['Model']}</strong> model for stroke prediction.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Metrics Display
    st.subheader("Key Performance Metrics")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🔍 Recall",
            value=f"{best_recall_model['Recall']:.3f}",
            help="Primary metric for medical diagnosis"
        )
    
    with col2:
        st.metric(
            label="📈 Precision",
            value=f"{best_recall_model['Precision']:.3f}"
        )
    
    with col3:
        st.metric(
            label="⚖️ F1 Score",
            value=f"{best_recall_model['F1 Score']:.3f}"
        )
    
    with col4:
        st.metric(
            label="🎯 Accuracy",
            value=f"{best_recall_model['Accuracy']:.3f}"
        )

    # Why Recall is Important
    st.markdown("""
        <div style='background-color: rgba(0, 200, 81, 0.1); padding: 20px; border-radius: 10px; margin-top: 20px;'>
            <h3 style='margin-bottom: 10px; color: white;'>Why Recall is Important</h3>
            <p style='color: white;'>In medical diagnosis, especially for stroke prediction, we prioritize Recall (sensitivity) over other metrics because:</p>
            <ul style='color: white; margin-left: 20px;'>
                <li>It minimizes false negatives (missed stroke risks)</li>
                <li>It's better to have some false positives than to miss actual stroke cases</li>
                <li>Early detection and prevention are crucial for patient safety</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Make Another Prediction"):
            st.session_state['page'] = 'input'
            st.rerun()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

# Main app logic
def main():
    if st.session_state['page'] == 'home':
        home_page()
    elif st.session_state['page'] == 'input':
        input_page()
    elif st.session_state['page'] == 'results':
        results_page()

if __name__ == "__main__":
    main()
