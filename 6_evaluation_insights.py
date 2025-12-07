"""
PHASE 6: EVALUATION AND INSIGHTS
Generate comprehensive evaluation and actionable insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.preprocessing import LabelEncoder

def generate_insights():
    """
    Generate comprehensive evaluation and insights
    """
    print("="*60)
    print("PHASE 6: EVALUATION AND INSIGHTS")
    print("="*60)
    
    # Load data and model
    print("\n Loading data and models...")
    data = pd.read_csv('data/preprocessed_data.csv')
    best_model = joblib.load('models/best_model.pkl')
    all_models = joblib.load('models/all_models.pkl')
    
    # Load encoders
    label_encoders = joblib.load('models/label_encoders.pkl')
    le_target = joblib.load('models/target_encoder.pkl')
    
    # Load features
    with open('models/selected_features.txt', 'r') as f:
        selected_features = [line.strip() for line in f.readlines()]
    
    print(f"Loaded data and {len(all_models)} models")
    
    # Prepare test data
    from sklearn.model_selection import train_test_split
    
    X = data[selected_features].copy()
    y = data['status'].copy()
    
    for col in X.columns:
        X[col] = label_encoders[col].transform(X[col])
    
    y_encoded = le_target.transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Get predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    print(f"\nEvaluating best model on {len(X_test)} test samples...")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("\nPerformance Metrics:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # ROC Curve
    print("\nGenerating ROC curve...")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='#667eea', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('eda_plots/12_roc_curve.png', dpi=300, bbox_inches='tight')
    print("   Saved: eda_plots/12_roc_curve.png")
    plt.close()
    
    # Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        print("\nAnalyzing feature importance...")
        
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(feature_importance['Feature'], 
                       feature_importance['Importance'],
                       color='#51cf66', edgecolor='black', linewidth=1.5)
        plt.xlabel('Importance Score', fontweight='bold')
        plt.title('Feature Importance in Best Model', fontweight='bold', fontsize=14)
        plt.grid(axis='x', alpha=0.3)
        
        for i, (bar, value) in enumerate(zip(bars, feature_importance['Importance'])):
            plt.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.4f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('eda_plots/13_final_feature_importance.png', dpi=300, bbox_inches='tight')
        print("   Saved: eda_plots/13_final_feature_importance.png")
        plt.close()
    
    # Key Insights Generation
    print("\nGenerating Key Insights...\n")
    
    insights = []
    
    # Insight 1: Model Performance
    insights.append(f"1. MODEL PERFORMANCE")
    insights.append(f"   The model achieves {accuracy*100:.1f}% accuracy in predicting student performance.")
    insights.append(f"   With an AUC of {roc_auc:.3f}, the model demonstrates strong discriminative ability.")
    
    # Insight 2: Test Preparation Impact
    prep_impact = data.groupby('test preparation course')['average_score'].mean()
    prep_diff = 0
    if 'completed' in prep_impact.index and 'none' in prep_impact.index:
        prep_diff = prep_impact['completed'] - prep_impact['none']
        insights.append(f"\n2. TEST PREPARATION IMPACT")
        insights.append(f"   Students who completed test prep score {prep_diff:.1f} points higher on average.")
        insights.append(f"   Recommendation: Encourage all students to enroll in test preparation courses.")
    
    # Insight 3: Pass Rate Analysis
    pass_rate = (data['status'] == 'Pass').sum() / len(data) * 100
    fail_count = (data['status'] == 'Fail').sum()
    insights.append(f"\n3. OVERALL PERFORMANCE")
    insights.append(f"   Current pass rate: {pass_rate:.1f}%")
    insights.append(f"   {fail_count} students are at risk of failing.")
    insights.append(f"   Early intervention could help these students improve.")
    
    # Insight 4: Parental Education Impact
    parent_impact = data.groupby('parental level of education')['average_score'].mean().sort_values()
    if len(parent_impact) > 0:
        highest_ed = parent_impact.index[-1]
        lowest_ed = parent_impact.index[0]
        score_diff = parent_impact.iloc[-1] - parent_impact.iloc[0]
        insights.append(f"\n4. SOCIOECONOMIC FACTORS")
        insights.append(f"   Students with parents having '{highest_ed}' score {score_diff:.1f} points higher")
        insights.append(f"   than those with parents having '{lowest_ed}'.")
        insights.append(f"   Recommendation: Provide additional support to students from lower-education backgrounds.")
    
    # Insight 5: Subject-wise Performance
    weakest_subject = data[['math score', 'reading score', 'writing score']].mean().idxmin()
    weakest_score = data[['math score', 'reading score', 'writing score']].mean().min()
    insights.append(f"\n5. SUBJECT ANALYSIS")
    insights.append(f"   Weakest subject: {weakest_subject.replace(' score', '').title()} (avg: {weakest_score:.1f})")
    insights.append(f"   Recommendation: Focus additional resources on {weakest_subject.replace(' score', '').title()} instruction.")
    
    # Actionable Recommendations (NO EMOJIS)
    print("\n" + "="*60)
    print("ACTIONABLE RECOMMENDATIONS FOR EDUCATORS")
    print("="*60)
    
    recommendations = [
        "\nIMMEDIATE ACTIONS:",
        "1. Identify at-risk students using the prediction model",
        "2. Enroll all students in test preparation courses",
        "3. Provide extra tutoring for the weakest subject areas",
        "",
        "DATA-DRIVEN STRATEGIES:",
        "4. Track attendance and engagement metrics closely",
        "5. Monitor students from lower socioeconomic backgrounds",
        "6. Implement early warning systems (Week 4-5 of semester)",
        "",
        "SUPPORT PROGRAMS:",
        "7. Create peer mentoring programs",
        "8. Offer additional support for economically disadvantaged students",
        "9. Engage parents in the learning process",
        "",
        "CONTINUOUS IMPROVEMENT:",
        "10. Regularly update the model with new data",
        "11. Measure effectiveness of interventions",
        "12. Adjust strategies based on results"
    ]
    
    for rec in recommendations:
        print(rec)
    
    # Print insights
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    for insight in insights:
        print(insight)
    
    # Potential Impact Analysis
    print("\n" + "="*60)
    print("POTENTIAL IMPACT ANALYSIS")
    print("="*60)
    
    at_risk_students = data[data['status'] == 'Fail']
    if len(at_risk_students) > 0 and prep_diff > 0:
        no_prep_failed = at_risk_students[at_risk_students['test preparation course'] == 'none']
        
        print(f"\nIf all at-risk students complete test preparation:")
        print(f"   Currently failing: {len(at_risk_students)} students")
        print(f"   Without test prep: {len(no_prep_failed)} students")
        print(f"   Potential improvement: +{prep_diff:.1f} points average")
        
        potential_saves = len(no_prep_failed[no_prep_failed['average_score'] > (50 - prep_diff)])
        print(f"   Estimated students who could pass: {potential_saves}")
        print(f"   New pass rate potential: {((len(data)-len(at_risk_students)+potential_saves)/len(data)*100):.1f}%")
    
    # Create comprehensive evaluation report (with UTF-8 encoding)
    print("\nCreating comprehensive evaluation report...")
    
    with open('reports/6_evaluation_insights_report.txt', 'w', encoding='utf-8') as f:
        f.write("EVALUATION & INSIGHTS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-"*60 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"ROC AUC:   {roc_auc:.4f}\n")
        f.write("-"*60 + "\n\n")
        
        f.write("KEY INSIGHTS:\n")
        f.write("="*60 + "\n")
        for insight in insights:
            f.write(f"{insight}\n")
        
        f.write("\n\nACTIONABLE RECOMMENDATIONS:\n")
        f.write("="*60 + "\n")
        for rec in recommendations:
            f.write(f"{rec}\n")
        
        f.write("\n\nVISUALIZATIONS CREATED:\n")
        f.write("-"*60 + "\n")
        f.write("- 12_roc_curve.png\n")
        f.write("- 13_final_feature_importance.png\n")
        
        f.write("\n\nCLASSIFICATION REPORT:\n")
        f.write("-"*60 + "\n")
        report = classification_report(y_test, y_pred, 
                                      target_names=['Fail', 'Pass'],
                                      zero_division=0)
        f.write(report)
    
    print("   Comprehensive evaluation report saved!")
    
    # Create summary dashboard data
    print("\nCreating summary dashboard data...")
    
    summary_data = {
        'total_students': len(data),
        'pass_rate': pass_rate,
        'fail_count': fail_count,
        'model_accuracy': accuracy * 100,
        'model_auc': roc_auc,
        'avg_math_score': data['math score'].mean(),
        'avg_reading_score': data['reading score'].mean(),
        'avg_writing_score': data['writing score'].mean(),
        'test_prep_impact': prep_diff,
        'at_risk_students': len(at_risk_students)
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv('reports/summary_statistics.csv', index=False)
    print("   Saved: reports/summary_statistics.csv")
    
    # Final summary
    print("\nEvaluation Summary:")
    print(f"   Model Accuracy: {accuracy*100:.1f}%")
    print(f"   ROC AUC: {roc_auc:.3f}")
    print(f"   Total Insights: {len(insights)}")
    print(f"   Recommendations: {len([r for r in recommendations if r.strip() and not r.startswith('\n')])}")
    
    return insights, recommendations, summary_data

if __name__ == "__main__":
    insights, recommendations, summary = generate_insights()
    
    print("\n" + "="*60)
    print("PHASE 6 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nALL PHASES COMPLETED!")
    print("\nNext Step: Run 'streamlit run app.py' to view dashboard")
