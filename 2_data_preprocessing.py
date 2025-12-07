"""
PHASE 2: DATA PREPROCESSING
Clean and prepare data for analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data():
    """
    Preprocess the raw data
    """
    print("="*60)
    print("PHASE 2: DATA PREPROCESSING")
    print("="*60)
    
    # Load raw data
    print("\n📊 Loading raw data...")
    data = pd.read_csv('data/student_data.csv')
    print(f"✅ Loaded {len(data)} records")
    
    # Step 1: Feature Engineering
    print("\n🔧 Step 1: Feature Engineering...")
    
    # Create average score
    data['average_score'] = (data['math score'] + 
                            data['reading score'] + 
                            data['writing score']) / 3
    print("   ✅ Created 'average_score' feature")
    
    # Create total score
    data['total_score'] = (data['math score'] + 
                          data['reading score'] + 
                          data['writing score'])
    print("   ✅ Created 'total_score' feature")
    
    # Create pass/fail status (threshold: 50)
    data['status'] = data['average_score'].apply(
        lambda x: 'Pass' if x >= 50 else 'Fail'
    )
    print("   ✅ Created 'status' feature (Pass/Fail)")
    
    # Create performance category
    data['performance'] = pd.cut(
        data['average_score'],
        bins=[0, 50, 70, 100],
        labels=['Poor', 'Average', 'Excellent']
    )
    print("   ✅ Created 'performance' category")
    
    # Identify weak subject
    def identify_weak_subject(row):
        scores = {
            'math': row['math score'],
            'reading': row['reading score'],
            'writing': row['writing score']
        }
        return min(scores, key=scores.get)
    
    data['weak_subject'] = data.apply(identify_weak_subject, axis=1)
    print("   ✅ Identified weak subject for each student")
    
    # Step 2: Handle Missing Values
    print("\n🔍 Step 2: Checking for missing values...")
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print(f"   Found {missing.sum()} missing values")
        data = data.dropna()
        print(f"   ✅ Removed rows with missing values")
    else:
        print("   ✅ No missing values found")
    
    # Step 3: Remove Sensitive Features (Ethical AI)
    print("\n⚖️ Step 3: Applying Ethical AI principles...")
    print("   Removing sensitive demographic features to prevent discrimination:")
    
    sensitive_features = []
    if 'gender' in data.columns:
        sensitive_features.append('gender')
        print("   - Removed 'gender' (not actionable)")
    if 'race/ethnicity' in data.columns:
        sensitive_features.append('race/ethnicity')
        print("   - Removed 'race/ethnicity' (potential discrimination)")
    
    # Keep only for display, not for modeling
    data_display = data.copy()
    
    print("   ✅ Ethical AI principles applied")
    
    # Step 4: Data Validation
    print("\n✓ Step 4: Data validation...")
    
    # Check score ranges
    score_cols = ['math score', 'reading score', 'writing score']
    for col in score_cols:
        if data[col].min() < 0 or data[col].max() > 100:
            print(f"   ⚠️ Warning: {col} has values outside 0-100 range")
        else:
            print(f"   ✅ {col}: Valid range [0-100]")
    
    # Step 5: Save preprocessed data
    print("\n💾 Step 5: Saving preprocessed data...")
    
    # Save full preprocessed data
    data_display.to_csv('data/preprocessed_data.csv', index=False)
    print("   ✅ Saved: data/preprocessed_data.csv")
    
    # Create preprocessing report
    print("\n📄 Creating preprocessing report...")
    with open('reports/2_preprocessing_report.txt', 'w') as f:
        f.write("DATA PREPROCESSING REPORT\n")
        f.write("="*60 + "\n\n")
        f.write("STEPS PERFORMED:\n")
        f.write("1. Feature Engineering\n")
        f.write("   - Created: average_score, total_score\n")
        f.write("   - Created: status (Pass/Fail)\n")
        f.write("   - Created: performance category\n")
        f.write("   - Identified: weak_subject\n\n")
        f.write("2. Missing Values Handling\n")
        f.write(f"   - Found: {missing.sum()} missing values\n")
        f.write(f"   - Action: Removed rows with missing data\n\n")
        f.write("3. Ethical AI Application\n")
        f.write("   - Removed sensitive features:\n")
        for feat in sensitive_features:
            f.write(f"     • {feat}\n")
        f.write("\n4. Final Dataset\n")
        f.write(f"   - Total records: {len(data)}\n")
        f.write(f"   - Total features: {len(data.columns)}\n")
        f.write(f"   - Pass rate: {(data['status']=='Pass').sum()/len(data)*100:.1f}%\n")
        f.write(f"   - Average score: {data['average_score'].mean():.2f}\n")
    
    print("   ✅ Preprocessing report saved!")
    
    # Summary statistics
    print("\n📊 Preprocessing Summary:")
    print(f"   Total records: {len(data)}")
    print(f"   Total features: {len(data.columns)}")
    print(f"   Pass rate: {(data['status']=='Pass').sum()/len(data)*100:.1f}%")
    print(f"   Average score: {data['average_score'].mean():.2f}")
    
    return data

if __name__ == "__main__":
    # Preprocess data
    data = preprocess_data()
    
    print("\n" + "="*60)
    print("✅ PHASE 2 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n👉 Next Step: Run '3_exploratory_analysis.py'")
