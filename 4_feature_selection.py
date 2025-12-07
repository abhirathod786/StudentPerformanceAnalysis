"""
PHASE 4: FEATURE SELECTION
Select relevant features for modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def feature_selection():
    """
    Perform feature selection and importance analysis
    """
    print("="*60)
    print("PHASE 4: FEATURE SELECTION")
    print("="*60)
    
    # Load preprocessed data
    print("\n📊 Loading preprocessed data...")
    data = pd.read_csv('data/preprocessed_data.csv')
    print(f"✅ Loaded {len(data)} records")
    
    # Define features based on ethical AI principles
    print("\n🎯 Selecting features based on:")
    print("   1. Actionability (students/schools can change)")
    print("   2. Non-discrimination (no sensitive demographics)")
    print("   3. Educational relevance")
    
    # Selected features (ETHICAL & ACTIONABLE)
    selected_features = [
        'parental level of education',
        'lunch',
        'test preparation course'
    ]
    
    print("\n✅ Selected Features:")
    for i, feature in enumerate(selected_features, 1):
        print(f"   {i}. {feature}")
    
    # Excluded features
    excluded_features = []
    if 'gender' in data.columns:
        excluded_features.append('gender')
    if 'race/ethnicity' in data.columns:
        excluded_features.append('race/ethnicity')
    
    if excluded_features:
        print("\n❌ Excluded Features (Ethical AI):")
        for feature in excluded_features:
            print(f"   - {feature} (non-actionable/discriminatory)")
    
    # Prepare features for modeling
    print("\n🔧 Preparing features for modeling...")
    
    X = data[selected_features].copy()
    y = data['status'].copy()
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    print(f"✅ Encoded {len(selected_features)} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n📦 Dataset Split:")
    print(f"   Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Testing: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Feature Importance Analysis using Random Forest
    print("\n🌲 Analyzing feature importance using Random Forest...")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n📊 Feature Importance Ranking:")
    for idx, row in feature_importance.iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.4f}")
    
    # Visualize feature importance
    print("\n📈 Creating feature importance visualization...")
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(feature_importance['Feature'], 
                    feature_importance['Importance'],
                    color='#667eea', edgecolor='black', linewidth=1.5)
    plt.xlabel('Importance Score', fontweight='bold')
    plt.title('Feature Importance Analysis', fontweight='bold', fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, feature_importance['Importance'])):
        plt.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eda_plots/8_feature_importance.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: eda_plots/8_feature_importance.png")
    plt.close()
    
    # Statistical analysis of features
    print("\n📊 Statistical Analysis of Selected Features:")
    
    for feature in selected_features:
        print(f"\n   {feature}:")
        value_counts = data[feature].value_counts()
        for val, count in value_counts.items():
            percentage = count / len(data) * 100
            print(f"      - {val}: {count} ({percentage:.1f}%)")
    
    # Impact analysis
    print("\n💡 Feature Impact Analysis:")
    
    for feature in selected_features:
        # Calculate pass rate by feature value
        impact = data.groupby(feature)['status'].apply(
            lambda x: (x == 'Pass').sum() / len(x) * 100
        )
        print(f"\n   {feature} - Pass Rate:")
        for val, rate in impact.items():
            print(f"      {val}: {rate:.1f}%")
    
    # Create feature selection report
    print("\n📄 Creating feature selection report...")
    
    with open('reports/4_feature_selection_report.txt', 'w') as f:
        f.write("FEATURE SELECTION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("SELECTION CRITERIA:\n")
        f.write("1. Actionability: Students/schools can change these factors\n")
        f.write("2. Non-discrimination: No sensitive demographics\n")
        f.write("3. Educational relevance: Proven impact on performance\n\n")
        
        f.write("SELECTED FEATURES:\n")
        for i, feature in enumerate(selected_features, 1):
            f.write(f"{i}. {feature}\n")
        
        if excluded_features:
            f.write("\nEXCLUDED FEATURES (Ethical AI):\n")
            for feature in excluded_features:
                f.write(f"- {feature}\n")
        
        f.write("\nFEATURE IMPORTANCE RANKING:\n")
        for idx, row in feature_importance.iterrows():
            f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
        
        f.write("\nFEATURE STATISTICS:\n")
        for feature in selected_features:
            f.write(f"\n{feature}:\n")
            value_counts = data[feature].value_counts()
            for val, count in value_counts.items():
                percentage = count / len(data) * 100
                f.write(f"  {val}: {count} ({percentage:.1f}%)\n")
        
        f.write("\nDATA SPLIT:\n")
        f.write(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)\n")
        f.write(f"Testing set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)\n")
    
    print("   ✅ Feature selection report saved!")
    
    # Save processed features
    print("\n💾 Saving processed features...")
    
    import joblib
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(le_target, 'models/target_encoder.pkl')
    
    # Save feature list
    with open('models/selected_features.txt', 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    
    print("   ✅ Saved encoders and feature list")
    
    print("\n📊 Feature Selection Summary:")
    print(f"   Selected features: {len(selected_features)}")
    print(f"   Most important: {feature_importance.iloc[0]['Feature']}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, selected_features

if __name__ == "__main__":
    # Create models directory
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
    
    X_train, X_test, y_train, y_test, features = feature_selection()
    
    print("\n" + "="*60)
    print("✅ PHASE 4 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n👉 Next Step: Run '5_model_building.py'")
