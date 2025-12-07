# 🎓 Student Performance Analysis System

An AI-powered platform for predicting student performance and providing personalized recommendations using Machine Learning and Ethical AI principles.

## 🌟 Features

- **Performance Prediction**: Binary classification (Pass/Fail) with confidence scores
- **Personalized Recommendations**: Subject-specific improvement suggestions
- **Interactive Dashboard**: Visual analytics and performance insights
- **Batch Processing**: Analyze multiple students simultaneously
- **Ethical AI**: Uses only actionable factors, excludes demographic data

## 🛡️ Ethical AI Approach

This system is designed with fairness and transparency in mind:

- ✅ **No Demographic Bias**: Predictions based solely on actionable factors
- ✅ **Transparent**: Clear explanations for all predictions
- ✅ **Actionable**: All recommendations can be implemented by students

### Features Used for Prediction:
- Parental Level of Education
- Lunch Type (socioeconomic indicator)
- Test Preparation Course

### Not Used:
- Gender, Race/Ethnicity (to prevent discrimination)

## 🚀 Live Demo

[Click here to try the app](your-streamlit-url-here)

## 💻 Technology Stack

- **Python**: Core programming language
- **Scikit-learn**: Machine learning models
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas & NumPy**: Data processing

## 📊 Models Implemented

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting

## 🏗️ Project Structure

```
StudentPerformanceAnalysis/
├── app.py                          # Streamlit web application
├── run_complete_pipeline.py        # Master pipeline script
├── 1_data_collection.py            # Phase 1: Data loading
├── 2_data_preprocessing.py         # Phase 2: Data cleaning
├── 3_exploratory_analysis.py       # Phase 3: EDA
├── 4_feature_selection.py          # Phase 4: Feature engineering
├── 5_model_building.py             # Phase 5: Model training
├── 6_evaluation_insights.py        # Phase 6: Evaluation
├── train_model.py                  # Standalone training script
├── check_models.py                 # Model diagnostics
├── requirements.txt                # Dependencies
├── data/
│   ├── student_data.csv           # Original dataset
│   └── preprocessed_data.csv      # Processed data
├── models/
│   ├── best_model.pkl             # Trained model
│   ├── label_encoders.pkl         # Feature encoders
│   ├── target_encoder.pkl         # Target encoder
│   └── feature_names.pkl          # Feature names
├── eda_plots/                      # Visualization outputs
└── reports/                        # Analysis reports
```

## 🔧 Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/student-performance-analysis.git
cd student-performance-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the complete pipeline:
```bash
python run_complete_pipeline.py
```

4. Launch the Streamlit app:
```bash
streamlit run app.py
```

## 📈 Usage

### Individual Analysis
1. Navigate to "Individual Analysis" page
2. Select a student ID
3. View predictions, recommendations, and comparisons

### Batch Prediction
1. Navigate to "Batch Prediction" page
2. Upload a CSV file with required columns
3. Download results with predictions

### Dashboard
- View overall performance statistics
- Analyze subject-wise performance
- Explore test preparation impact

## 📝 Dataset

The system uses the "Students Performance in Exams" dataset containing:
- Academic background (parental education)
- Behavioral features (test preparation, lunch type)
- Performance metrics (math, reading, writing scores)

## 👨‍🎓 Project Information

**Author**: ABHISHEK (3VY22UE002)

**Institution**: VTU's CPGS, Kalaburagi

**Department**: Electronics and Communication Engineering

**Guide**: Prof. Shrinivas.G

**Year**: 2024-2025

## 📄 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Contact

For questions or feedback, please contact the project team.

---

**Built with ❤️ using Ethical AI Principles - Fair, Transparent, Actionable**
