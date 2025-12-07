import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the student performance dataset
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Create a copy
    data = df.copy()
    
    # Feature Engineering
    # 1. Average Score
    data['average_score'] = (data['math score'] + data['reading score'] + data['writing score']) / 3
    
    # 2. Performance Category
    data['performance'] = pd.cut(data['average_score'], 
                                  bins=[0, 50, 70, 100],
                                  labels=['Fail', 'Average', 'Excellent'])
    
    # 3. Total Score
    data['total_score'] = data['math score'] + data['reading score'] + data['writing score']
    
    # 4. Pass/Fail Status (50 is passing threshold)
    data['status'] = data['average_score'].apply(lambda x: 'Pass' if x >= 50 else 'Fail')
    
    # 5. Weak Subject Identification
    def identify_weak_subject(row):
        scores = {
            'math': row['math score'],
            'reading': row['reading score'],
            'writing': row['writing score']
        }
        return min(scores, key=scores.get)
    
    data['weak_subject'] = data.apply(identify_weak_subject, axis=1)
    
    return data

def prepare_features(data):
    """
    Prepare features for model training
    Note: Excludes sensitive demographic features (gender, race/ethnicity) 
    to ensure ethical AI and prevent discrimination
    """
    # Select features for training - ONLY actionable factors
    feature_columns = ['parental level of education', 'lunch', 'test preparation course']
    
    target = 'status'
    
    # Create feature dataframe
    X = data[feature_columns].copy()
    y = data[target].copy()
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Encode target variable
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    
    return X, y, label_encoders, le_target

def get_feature_names():
    """
    Return user-friendly feature names
    """
    return {
        'parental level of education': 'Parental Education',
        'lunch': 'Lunch Type',
        'test preparation course': 'Test Preparation'
    }
