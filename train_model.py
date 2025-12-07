import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('utils')
from preprocessing import load_and_preprocess_data, prepare_features

def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train multiple ML models and compare performance
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    print("\n" + "="*60)
    print("MODEL TRAINING & EVALUATION")
    print("="*60 + "\n")
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        trained_models[name] = model
        
        print(f"{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print()
    
    
    # Find best model - prefer tree models for SHAP compatibility
    tree_models = ['Decision Tree', 'Random Forest', 'Gradient Boosting']
    tree_results = {k: v for k, v in results.items() if k in tree_models}

    if tree_results:
        best_tree_model = max(tree_results, key=lambda x: tree_results[x]['accuracy'])
        best_overall = max(results, key=lambda x: results[x]['accuracy'])
        
        if tree_results[best_tree_model]['accuracy'] >= results[best_overall]['accuracy'] - 0.02:
            best_model_name = best_tree_model
            print(f"\n✨ Selected {best_model_name} for SHAP compatibility")
            print(f"   Accuracy: {tree_results[best_tree_model]['accuracy']:.4f}")
            print(f"   (Trade-off: -{(results[best_overall]['accuracy'] - tree_results[best_tree_model]['accuracy'])*100:.1f}% for instant explainability)")
        else:
            best_model_name = best_overall
    else:
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    
    best_model = trained_models[best_model_name]

    return trained_models, results, best_model, best_model_name

def plot_model_comparison(results):
    """
    Plot model comparison
    """
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Models', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Model comparison plot saved: model_comparison.png\n")

def plot_confusion_matrix(y_test, y_pred, model_name):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fail', 'Pass'],
                yticklabels=['Fail', 'Pass'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=300)
    print(f"✅ Confusion matrix saved: confusion_matrix_{model_name.replace(' ', '_')}.png\n")

def save_model(model, filepath='models/trained_model.pkl'):
    """
    Save trained model
    """
    joblib.dump(model, filepath)
    print(f"✅ Model saved: {filepath}\n")

def main():
    # Load and preprocess data
    print("📊 Loading data...")
    data = load_and_preprocess_data('data/student_data.csv')
    print(f"✅ Data loaded: {data.shape[0]} students\n")
    
    # Prepare features
    print("🔧 Preparing features...")
    X, y, label_encoders, le_target = prepare_features(data)
    print(f"✅ Features prepared: {X.shape[1]} features\n")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📦 Dataset split:")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Testing set: {X_test.shape[0]} samples\n")
    
    # Train all models
    trained_models, results, best_model, best_model_name = train_all_models(
        X_train, X_test, y_train, y_test
    )
    
    # Plot comparison
    print("📊 Creating visualizations...")
    plot_model_comparison(results)
    
    # Plot confusion matrix for best model
    y_pred_best = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_best, best_model_name)
    
    # Classification report
    print("📋 Classification Report (Best Model):")
    print("="*60)
    print(classification_report(y_test, y_pred_best, 
                                target_names=['Fail', 'Pass'],
                                zero_division=0))
    
    # Save best model
    save_model(best_model, 'models/best_model.pkl')
    
    # Save label encoders
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(le_target, 'models/target_encoder.pkl')
    print("✅ All artifacts saved successfully!\n")
    
    print("="*60)
    print("🎉 MODEL TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
