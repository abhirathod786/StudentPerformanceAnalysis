import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page config
st.set_page_config(
    page_title="Student Performance Analysis System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation-high {
        background-color: #ffe5e5;
        padding: 15px;
        border-left: 5px solid #ff4444;
        margin: 10px 0;
        border-radius: 5px;
    }
    .recommendation-medium {
        background-color: #fff4e5;
        padding: 15px;
        border-left: 5px solid #ffaa00;
        margin: 10px 0;
        border-radius: 5px;
    }
    .recommendation-low {
        background-color: #e5f5ff;
        padding: 15px;
        border-left: 5px solid #4444ff;
        margin: 10px 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model_and_data():
    try:
        # Load model files
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        with open('models/target_encoder.pkl', 'rb') as f:
            target_encoder = pickle.load(f)
        
        # Try to load feature names
        try:
            with open('models/feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
        except:
            # Use the ethical feature set (no demographic data)
            feature_names = ['parental level of education', 'lunch', 'test preparation course']
        
        # Load data - try preprocessed first, then original
        if os.path.exists('data/preprocessed_data.csv'):
            data = pd.read_csv('data/preprocessed_data.csv')
        elif os.path.exists('data/student_data.csv'):
            data = pd.read_csv('data/student_data.csv')
        else:
            raise FileNotFoundError("No data file found. Please ensure data/preprocessed_data.csv exists.")
        
        return model, label_encoders, target_encoder, feature_names, data
    
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

# Initialize
try:
    model, label_encoders, target_encoder, feature_names, data = load_model_and_data()
    
    # Ensure required columns exist
    required_cols = ['math score', 'reading score', 'writing score']
    
    for col in required_cols:
        if col not in data.columns:
            st.error(f"❌ Missing column in data: {col}")
            st.stop()
    
    # Calculate average score if not present
    if 'average_score' not in data.columns:
        data['average_score'] = data[['math score', 'reading score', 'writing score']].mean(axis=1)
    
    # Calculate status if not present
    if 'status' not in data.columns:
        data['status'] = data['average_score'].apply(lambda x: 'Pass' if x >= 50 else 'Fail')
    
    # Calculate performance_class if not present
    if 'performance_class' not in data.columns:
        data['performance_class'] = data['status']
    
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.info("Please run the complete pipeline first: `python run_complete_pipeline.py`")
    st.stop()

# Helper functions
def get_risk_level(score):
    if score >= 70:
        return "🟢 Low Risk - Performing Well", "green"
    elif score >= 50:
        return "🟡 Medium Risk - Needs Attention", "orange"
    else:
        return "🔴 High Risk - Critical Intervention Needed", "red"

def generate_recommendations(student_row):
    recommendations = []
    
    # Math recommendations
    if student_row['math score'] < 60:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Mathematics',
            'recommendation': 'Focus on fundamental math concepts and practice problem-solving',
            'expected_improvement': '+10-15 points',
            'reason': f"Current math score ({student_row['math score']}) is below average"
        })
    
    # Reading recommendations
    if student_row['reading score'] < 60:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Reading',
            'recommendation': 'Increase reading practice and comprehension exercises',
            'expected_improvement': '+8-12 points',
            'reason': f"Current reading score ({student_row['reading score']}) needs improvement"
        })
    
    # Writing recommendations
    if student_row['writing score'] < 60:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Writing',
            'recommendation': 'Practice essay writing and grammar exercises',
            'expected_improvement': '+8-12 points',
            'reason': f"Current writing score ({student_row['writing score']}) is below target"
        })
    
    # Test preparation
    if 'test preparation course' in student_row.index and student_row['test preparation course'] == 'none':
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Test Preparation',
            'recommendation': 'Enroll in test preparation course',
            'expected_improvement': '+15-20% overall',
            'reason': 'Students with test prep show significantly better performance'
        })
    
    # General study habits
    if student_row['average_score'] < 60:
        recommendations.append({
            'priority': 'CRITICAL',
            'category': 'Study Habits',
            'recommendation': 'Develop structured study schedule and seek tutoring',
            'expected_improvement': '+20-30 points',
            'reason': 'Overall performance is significantly below passing threshold'
        })
    
    return recommendations if recommendations else [{
        'priority': 'LOW',
        'category': 'Maintenance',
        'recommendation': 'Continue current study methods',
        'expected_improvement': 'Maintain current level',
        'reason': 'Performance is satisfactory'
    }]

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/student-center.png", width=100)
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "📊 Dashboard", "🔍 Individual Analysis", "📈 Batch Prediction", "ℹ️ About"]
)

# === HOME PAGE ===
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🎓 Student Performance Analysis System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the AI-Powered Student Success Platform
    
    This system uses **Machine Learning** to:
    - 🎯 Predict student performance (Pass/Fail)
    - 💡 Provide personalized improvement recommendations
    - 📊 Analyze performance trends
    - 🚀 Enable early intervention for at-risk students
    
    **Ethical AI:** This system only uses actionable factors (parental education, lunch type, test preparation) 
    and excludes demographic data (gender, race) to ensure fair predictions.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**📚 Total Students**")
        st.metric("Count", len(data))
    
    with col2:
        pass_rate = (data['status'] == 'Pass').sum() / len(data) * 100
        st.success("**✅ Pass Rate**")
        st.metric("Percentage", f"{pass_rate:.1f}%")
    
    with col3:
        avg_score = data['average_score'].mean()
        st.warning("**📊 Average Score**")
        st.metric("Score", f"{avg_score:.1f}")
    
    st.markdown("---")
    
    # Performance distribution
    st.subheader("📊 Performance Distribution")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Pass/Fail Distribution", "Score Distribution"),
        specs=[[{"type": "pie"}, {"type": "histogram"}]]
    )
    
    # Pie chart
    status_counts = data['status'].value_counts()
    fig.add_trace(
        go.Pie(labels=status_counts.index, values=status_counts.values,
               marker=dict(colors=['#51cf66', '#ff6b6b'])),
        row=1, col=1
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=data['average_score'], nbinsx=20,
                    marker=dict(color='#667eea')),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# === DASHBOARD PAGE ===
elif page == "📊 Dashboard":
    st.title("📊 Performance Dashboard")
    
    # Filters (only show if columns exist)
    st.sidebar.subheader("🔍 Filters")
    
    filters = {}
    if 'test preparation course' in data.columns:
        test_prep_filter = st.sidebar.multiselect(
            "Test Preparation",
            options=data['test preparation course'].unique(),
            default=data['test preparation course'].unique()
        )
        filters['test preparation course'] = test_prep_filter
    
    # Filter data
    filtered_data = data.copy()
    for col, values in filters.items():
        filtered_data = filtered_data[filtered_data[col].isin(values)]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(filtered_data))
    
    with col2:
        pass_count = (filtered_data['status'] == 'Pass').sum()
        st.metric("Passed", pass_count, 
                 delta=f"{pass_count/len(filtered_data)*100:.1f}%")
    
    with col3:
        fail_count = (filtered_data['status'] == 'Fail').sum()
        st.metric("Failed", fail_count,
                 delta=f"{fail_count/len(filtered_data)*100:.1f}%",
                 delta_color="inverse")
    
    with col4:
        avg_score = filtered_data['average_score'].mean()
        st.metric("Avg Score", f"{avg_score:.1f}")
    
    st.markdown("---")
    
    # Subject-wise performance
    st.subheader("📚 Subject-wise Performance")
    
    subject_cols = ['math score', 'reading score', 'writing score']
    subject_means = filtered_data[subject_cols].mean()
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Math', 'Reading', 'Writing'],
            y=subject_means.values,
            marker=dict(color=['#ff6b6b', '#51cf66', '#ffd93d']),
            text=subject_means.values.round(2),
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Average Scores by Subject",
        xaxis_title="Subject",
        yaxis_title="Average Score",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance by test preparation (if available)
    if 'test preparation course' in data.columns:
        st.subheader("🎯 Impact of Test Preparation")
        
        prep_analysis = filtered_data.groupby('test preparation course')['average_score'].agg(['mean', 'count'])
        
        fig = go.Figure(data=[
            go.Bar(
                name='Average Score',
                x=prep_analysis.index,
                y=prep_analysis['mean'],
                text=prep_analysis['mean'].round(2),
                textposition='auto',
                marker=dict(color='#667eea')
            )
        ])
        
        fig.update_layout(
            title="Average Score by Test Preparation Status",
            xaxis_title="Test Preparation",
            yaxis_title="Average Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Score Correlations")
    
    corr_data = filtered_data[['math score', 'reading score', 'writing score']].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=['Math', 'Reading', 'Writing'],
        y=['Math', 'Reading', 'Writing'],
        colorscale='RdBu',
        text=corr_data.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 14},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(title="Subject Score Correlations", height=400)
    st.plotly_chart(fig, use_container_width=True)

# === INDIVIDUAL ANALYSIS PAGE ===
elif page == "🔍 Individual Analysis":
    st.title("🔍 Individual Student Analysis")
    
    st.markdown("""
    Select a student to get:
    - Performance prediction
    - Personalized recommendations
    
    **Note:** Predictions are based only on actionable factors (parental education, lunch type, test preparation).
    """)
    
    # Student selection
    student_ids = data.index.tolist()
    selected_idx = st.selectbox("Select Student ID:", student_ids)
    
    student_row = data.loc[selected_idx]
    
    st.markdown("---")
    
    # Student Info (show available features)
    st.subheader("👤 Student Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        for feature in feature_names[:len(feature_names)//2 + 1]:
            if feature in student_row.index:
                st.write(f"**{feature.title()}:** {student_row[feature]}")
    
    with col2:
        for feature in feature_names[len(feature_names)//2 + 1:]:
            if feature in student_row.index:
                st.write(f"**{feature.title()}:** {student_row[feature]}")
    
    st.markdown("---")
    
    # Scores
    st.subheader("📊 Current Scores")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Math", student_row['math score'])
    with col2:
        st.metric("Reading", student_row['reading score'])
    with col3:
        st.metric("Writing", student_row['writing score'])
    with col4:
        st.metric("Average", f"{student_row['average_score']:.1f}")
    
    # Risk level
    risk_level, color = get_risk_level(student_row['average_score'])
    
    if color == 'red':
        st.error(f"### {risk_level}")
    elif color == 'orange':
        st.warning(f"### {risk_level}")
    else:
        st.success(f"### {risk_level}")
    
    st.markdown("---")
    
    # Prediction
    st.subheader("🎯 Performance Prediction")
    
    # Prepare input - use only the features the model was trained on
    X_input = pd.DataFrame()
    for col in feature_names:
        if col in student_row.index:
            X_input[col] = [student_row[col]]
    
    # Check if we have all required features
    if len(X_input.columns) != len(feature_names):
        st.warning("⚠️ Some required features are missing. Prediction may not be accurate.")
    
    # Encode categorical features
    for col in X_input.columns:
        if col in label_encoders:
            try:
                X_input[col] = label_encoders[col].transform(X_input[col])
            except ValueError as e:
                st.error(f"Error encoding {col}: {e}")
                st.stop()
    
    # Predict
    prediction = model.predict(X_input)[0]
    prediction_proba = model.predict_proba(X_input)[0]
    
    predicted_status = target_encoder.inverse_transform([prediction])[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if predicted_status == 'Pass':
            st.success(f"### ✅ Predicted: {predicted_status}")
        else:
            st.error(f"### ❌ Predicted: {predicted_status}")
    
    with col2:
        confidence = prediction_proba[prediction] * 100
        st.info(f"### 📊 Confidence: {confidence:.1f}%")
    
    # Probability gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_proba[1] * 100,
        title={'text': "Pass Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "#ff6b6b"},
                {'range': [50, 70], 'color': "#ffd93d"},
                {'range': [70, 100], 'color': "#51cf66"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("💡 Personalized Recommendations")
    
    recommendations = generate_recommendations(student_row)
    
    for rec in recommendations:
        priority = rec['priority']
        
        if priority == 'CRITICAL' or priority == 'HIGH':
            css_class = 'recommendation-high'
        elif priority == 'MEDIUM':
            css_class = 'recommendation-medium'
        else:
            css_class = 'recommendation-low'
        
        st.markdown(f"""
        <div class="{css_class}">
            <strong>Priority: {priority}</strong> | <strong>{rec['category']}</strong><br>
            📌 <strong>Action:</strong> {rec['recommendation']}<br>
            📈 <strong>Expected Impact:</strong> {rec['expected_improvement']}<br>
            💬 <strong>Why:</strong> {rec['reason']}
        </div>
        """, unsafe_allow_html=True)
    
    # Peer comparison
    st.markdown("---")
    st.subheader("👥 Score Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Your Score", f"{student_row['average_score']:.1f}")
    
    with col2:
        avg = data['average_score'].mean()
        gap = student_row['average_score'] - avg
        st.metric("Class Average", 
                 f"{avg:.1f}",
                 delta=f"{gap:.1f}")
    
    with col3:
        top = data['average_score'].quantile(0.9)
        gap_top = student_row['average_score'] - top
        st.metric("Top 10%", 
                 f"{top:.1f}",
                 delta=f"{gap_top:.1f}")

# === BATCH PREDICTION PAGE ===
elif page == "📈 Batch Prediction":
    st.title("📈 Batch Prediction")
    
    st.markdown(f"""
    Upload a CSV file with student data to get predictions for multiple students.
    
    **Required columns:** {', '.join(feature_names)}
    """)
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(batch_data)} students")
            
            st.subheader("📋 Data Preview")
            st.dataframe(batch_data.head())
            
            # Check for required columns
            missing_cols = [col for col in feature_names if col not in batch_data.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                if st.button("🚀 Run Predictions"):
                    with st.spinner("Processing..."):
                        # Prepare features
                        X_batch = batch_data[feature_names].copy()
                        
                        # Encode
                        for col in X_batch.columns:
                            if col in label_encoders:
                                X_batch[col] = label_encoders[col].transform(X_batch[col])
                        
                        # Predict
                        predictions = model.predict(X_batch)
                        predictions_proba = model.predict_proba(X_batch)
                        
                        # Add results
                        batch_data['Predicted_Status'] = target_encoder.inverse_transform(predictions)
                        batch_data['Pass_Probability'] = predictions_proba[:, 1] * 100
                        batch_data['Risk_Level'] = batch_data['Pass_Probability'].apply(
                            lambda x: 'Low Risk 🟢' if x >= 70 else ('Medium Risk 🟡' if x >= 50 else 'High Risk 🔴')
                        )
                        
                        st.success("✅ Predictions Complete!")
                        
                        # Summary
                        st.subheader("📊 Summary")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            pass_count = (batch_data['Predicted_Status'] == 'Pass').sum()
                            st.metric("Predicted Pass", pass_count)
                        
                        with col2:
                            fail_count = (batch_data['Predicted_Status'] == 'Fail').sum()
                            st.metric("Predicted Fail", fail_count)
                        
                        with col3:
                            high_risk = (batch_data['Risk_Level'] == 'High Risk 🔴').sum()
                            st.metric("High Risk Students", high_risk)
                        
                        # Results table
                        st.subheader("📋 Detailed Results")
                        st.dataframe(batch_data)
                        
                        # Download
                        csv = batch_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"❌ Error: {e}")

# === ABOUT PAGE ===
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ## 🎓 Student Performance Analysis System
    
    ### Overview
    This system uses Machine Learning to predict student performance
    and provide actionable recommendations for improvement.
    
    ### Key Features
    - 🎯 **Performance Prediction**: Binary classification (Pass/Fail)
    - 💡 **Personalized Recommendations**: Subject-specific improvement suggestions
    - 📊 **Interactive Dashboard**: Visual analytics and insights
    - 📈 **Batch Processing**: Analyze multiple students at once
    - ⚖️ **Ethical AI**: Only uses actionable factors, excludes demographic data
    
    ### Ethical AI Approach
    This system is designed with fairness in mind:
    - **No demographic bias**: Predictions are based solely on actionable factors
    - **Transparent**: Clear explanations for all predictions
    - **Actionable**: All recommendations can be acted upon by students
    
    ### Features Used for Prediction
    - Parental Level of Education
    - Lunch Type (indicator of socioeconomic support)
    - Test Preparation Course
    
    **Not Used:** Gender, Race/Ethnicity (to prevent discrimination)
    
    ### Technology Stack
    - **Python**: Core programming language
    - **Scikit-learn**: Machine learning models
    - **Streamlit**: Web application framework
    - **Plotly**: Interactive visualizations
    
    ### Models Used
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
    - Gradient Boosting
    
    ### Dataset
    The system uses the "Students Performance in Exams" dataset containing:
    - Academic background (parental education)
    - Behavioral features (test preparation, lunch type)
    - Performance metrics (math, reading, writing scores)
    
    ### Project By
    *ABHISHEK* (3VY22UE002)
    *SAI KIRAN* (3VY22UE046)
    *GURUGOVIND* (3VY23UE400)
    
    VTU's CPGS, Kalaburagi
    
    Department of Electronics and Communication Engineering
    
    Under the guidance of **Prof. Shrinivas.G**
    
    ---
    

    """)
    
    st.balloons()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>© 2024-2025 Student Performance Analysis System | VTU's CPGS Kalaburagi</p>
    <p style='font-size: 0.9em;'>Built with Ethical AI Principles - Fair, Transparent, Actionable</p>
</div>
""", unsafe_allow_html=True)
