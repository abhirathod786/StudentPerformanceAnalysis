"""
PHASE 5: MODEL BUILDING
Train and compare multiple ML algorithms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def build_models():
    """
    Build and train multiple ML models
    """
    print("="*60)
    print("PHASE 5: MODEL BUILDING")
    print("="*60)
    
    # Load data
    print("\n📊 Loading data...")
    data = pd.read_csv('data/preprocessed_data.csv')
    
    # Load selected features
    with open('models/selected_features.txt', 'r') as f:
        selected_features = [line.strip() for line in f.readlines()]
    
    print(f"✅ Using {len(selected_features)} features")
    
    # Prepare data
    X = data[selected_features].copy()
    y = data['status'].copy()
    
    # Load encoders
    label_encoders = joblib.load('models/label_encoders.pkl')
    le_target = joblib.load('models/target_encoder.pkl')
    
    # Encode features
    for col in X.columns:
        X[col] = label_encoders[col].transform(X[col])
    
    # Encode target
    y_encoded = le_target.transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n📦 Dataset Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Define models
    print("\n🤖 Defining ML models...")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    print(f"✅ Configured {len(models)} models")
    
    # Train and evaluate models
    print("\n" + "="*60)
    print("TRAINING & EVALUATION")
    print("="*60 + "\n")
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"🔄 Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred
        }
        
        trained_models[name] = model
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   ✅ CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print()
    
    # Select best model (prefer tree-based for SHAP)
    print("="*60)
    print("MODEL SELECTION")
    print("="*60 + "\n")
    
    tree_models = ['Decision Tree', 'Random Forest', 'Gradient Boosting']
    tree_results = {k: v for k, v in results.items() if k in tree_models}
    
    if tree_results:
        best_tree = max(tree_results, key=lambda x: tree_results[x]['accuracy'])
        best_overall = max(results, key=lambda x: results[x]['accuracy'])
        
        # If tree model within 2% of best, use tree for SHAP compatibility
        if tree_results[best_tree]['accuracy'] >= results[best_overall]['accuracy'] - 0.02:
            best_model_name = best_tree
            print(f"✨ Selected: {best_model_name}")
            print(f"   Reason: SHAP compatibility (instant explanations)")
            print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
            
            acc_diff = (results[best_overall]['accuracy'] - 
                       results[best_model_name]['accuracy']) * 100
            if acc_diff > 0:
                print(f"   Trade-off: -{acc_diff:.1f}% for better explainability")
        else:
            best_model_name = best_overall
            print(f"🏆 Selected: {best_model_name}")
            print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
    else:
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        print(f"🏆 Selected: {best_model_name}")
        print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    best_model = results[best_model_name]['model']
    y_pred_best = results[best_model_name]['y_pred']
    
    # Visualizations
    print("\n📊 Creating visualizations...\n")
    
    # 1. Model Comparison
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars = plt.bar(model_names, accuracies, color=colors, 
                   edgecolor='black', linewidth=1.5)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    plt.xlabel('Models', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('eda_plots/9_model_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: eda_plots/9_model_comparison.png")
    plt.close()
    
    # 2. Confusion Matrix (Best Model)
    cm = confusion_matrix(y_test, y_pred_best)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fail', 'Pass'],
                yticklabels=['Fail', 'Pass'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {best_model_name}', 
             fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'eda_plots/10_confusion_matrix_{best_model_name.replace(" ", "_")}.png', 
               dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: eda_plots/10_confusion_matrix_{best_model_name.replace(' ', '_')}.png")
    plt.close()
    
    # 3. Cross-validation scores
    cv_means = [results[m]['cv_mean'] for m in model_names]
    cv_stds = [results[m]['cv_std'] for m in model_names]
    
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(model_names))
    plt.bar(x_pos, cv_means, yerr=cv_stds, capsize=5,
           color='#667eea', edgecolor='black', linewidth=1.5)
    plt.xlabel('Models', fontsize=12, fontweight='bold')
    plt.ylabel('Cross-Validation Score', fontsize=12, fontweight='bold')
    plt.title('Cross-Validation Performance', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, model_names, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('eda_plots/11_cross_validation_scores.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: eda_plots/11_cross_validation_scores.png")
    plt.close()
    
    # Classification Report
    print("\n📋 Classification Report (Best Model):")
    print("="*60)
    report = classification_report(y_test, y_pred_best, 
                                   target_names=['Fail', 'Pass'],
                                   zero_division=0)
    print(report)
    
    # Save models
    print("\n💾 Saving models...")
    
    # Save best model
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"   ✅ Saved: models/best_model.pkl ({best_model_name})")
    
    # Save all models
    joblib.dump(trained_models, 'models/all_models.pkl')
    print("   ✅ Saved: models/all_models.pkl")
    
    # Create model building report
    print("\n📄 Creating model building report...")
    
    with open('reports/5_model_building_report.txt', 'w') as f:
        f.write("MODEL BUILDING REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("MODELS TRAINED:\n")
        for i, name in enumerate(models.keys(), 1):
            f.write(f"{i}. {name}\n")
        
        f.write("\nPERFORMANCE SUMMARY:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Model':<25} {'Accuracy':<12} {'CV Score':<15}\n")
        f.write("-"*60 + "\n")
        for name in model_names:
            acc = results[name]['accuracy']
            cv = results[name]['cv_mean']
            cv_std = results[name]['cv_std']
            f.write(f"{name:<25} {acc:.4f}       {cv:.4f} (±{cv_std:.4f})\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write(f"SELECTED MODEL: {best_model_name}\n")
        f.write(f"Accuracy: {results[best_model_name]['accuracy']:.4f}\n")
        f.write(f"CV Score: {results[best_model_name]['cv_mean']:.4f}\n")
        f.write("="*60 + "\n\n")
        
        f.write("CLASSIFICATION REPORT:\n")
        f.write("-"*60 + "\n")
        f.write(report)
        
        f.write("\nCONFUSION MATRIX:\n")
        f.write(f"{cm}\n\n")
        
        f.write("VISUALIZATIONS CREATED:\n")
        f.write("- 9_model_comparison.png\n")
        f.write(f"- 10_confusion_matrix_{best_model_name.replace(' ', '_')}.png\n")
        f.write("- 11_cross_validation_scores.png\n")
    
    print("   ✅ Model building report saved!")
    
    print("\n📊 Model Building Summary:")
    print(f"   Models trained: {len(models)}")
    print(f"   Best model: {best_model_name}")
    print(f"   Best accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"   CV score: {results[best_model_name]['cv_mean']:.4f}")
    
    return results, best_model, best_model_name

if __name__ == "__main__":
    results, best_model, best_model_name = build_models()
    
    print("\n" + "="*60)
    print("✅ PHASE 5 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n👉 Next Step: Run '6_evaluation_insights.py'")
