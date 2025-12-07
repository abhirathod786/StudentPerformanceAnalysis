"""
PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)
Analyze and visualize data patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def exploratory_analysis():
    """
    Perform comprehensive EDA
    """
    print("="*60)
    print("PHASE 3: EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Load preprocessed data
    print("\n📊 Loading preprocessed data...")
    data = pd.read_csv('data/preprocessed_data.csv')
    print(f"✅ Loaded {len(data)} records")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # Create EDA directory
    import os
    if not os.path.exists('eda_plots'):
        os.makedirs('eda_plots')
    
    print("\n📈 Generating visualizations...\n")
    
    # 1. Distribution Analysis
    print("1️⃣ Score Distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(data['math score'], bins=20, color='#ff6b6b', edgecolor='black')
    axes[0, 0].set_title('Math Score Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Score')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].hist(data['reading score'], bins=20, color='#51cf66', edgecolor='black')
    axes[0, 1].set_title('Reading Score Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Score')
    axes[0, 1].set_ylabel('Frequency')
    
    axes[1, 0].hist(data['writing score'], bins=20, color='#ffd93d', edgecolor='black')
    axes[1, 0].set_title('Writing Score Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(data['average_score'], bins=20, color='#667eea', edgecolor='black')
    axes[1, 1].set_title('Average Score Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('eda_plots/1_score_distributions.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: eda_plots/1_score_distributions.png")
    plt.close()
    
    # 2. Pass/Fail Analysis
    print("2️⃣ Pass/Fail Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    status_counts = data['status'].value_counts()
    axes[0].bar(status_counts.index, status_counts.values, 
               color=['#51cf66', '#ff6b6b'], edgecolor='black', linewidth=1.5)
    axes[0].set_title('Pass vs Fail Distribution', fontweight='bold', fontsize=14)
    axes[0].set_ylabel('Number of Students')
    for i, v in enumerate(status_counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    axes[1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
               colors=['#51cf66', '#ff6b6b'], startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Pass Rate', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('eda_plots/2_pass_fail_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: eda_plots/2_pass_fail_analysis.png")
    plt.close()
    
    # 3. Subject Performance Comparison
    print("3️⃣ Subject Performance Comparison...")
    subject_means = data[['math score', 'reading score', 'writing score']].mean()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(['Math', 'Reading', 'Writing'], subject_means.values,
                   color=['#ff6b6b', '#51cf66', '#ffd93d'], 
                   edgecolor='black', linewidth=1.5)
    plt.title('Average Scores by Subject', fontweight='bold', fontsize=14)
    plt.ylabel('Average Score')
    plt.ylim(0, 100)
    for bar, value in zip(bars, subject_means.values):
        plt.text(bar.get_x() + bar.get_width()/2, value + 2, 
                f'{value:.1f}', ha='center', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('eda_plots/3_subject_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: eda_plots/3_subject_comparison.png")
    plt.close()
    
    # 4. Test Preparation Impact
    print("4️⃣ Test Preparation Impact...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    prep_scores = data.groupby('test preparation course')['average_score'].mean()
    axes[0].bar(prep_scores.index, prep_scores.values, 
               color=['#ff6b6b', '#51cf66'], edgecolor='black', linewidth=1.5)
    axes[0].set_title('Average Score by Test Prep Status', fontweight='bold')
    axes[0].set_ylabel('Average Score')
    for i, v in enumerate(prep_scores.values):
        axes[0].text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')
    
    prep_pass_rate = data.groupby('test preparation course')['status'].apply(
        lambda x: (x == 'Pass').sum() / len(x) * 100
    )
    axes[1].bar(prep_pass_rate.index, prep_pass_rate.values,
               color=['#ff6b6b', '#51cf66'], edgecolor='black', linewidth=1.5)
    axes[1].set_title('Pass Rate by Test Prep Status', fontweight='bold')
    axes[1].set_ylabel('Pass Rate (%)')
    axes[1].set_ylim(0, 100)
    for i, v in enumerate(prep_pass_rate.values):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eda_plots/4_test_prep_impact.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: eda_plots/4_test_prep_impact.png")
    plt.close()
    
    # 5. Correlation Analysis
    print("5️⃣ Correlation Analysis...")
    
    # Encode categorical variables for correlation
    data_encoded = data.copy()
    for col in ['parental level of education', 'lunch', 'test preparation course', 'status']:
        if col in data_encoded.columns:
            data_encoded[col] = LabelEncoder().fit_transform(data_encoded[col])
    
    corr_features = ['math score', 'reading score', 'writing score', 
                    'average_score', 'parental level of education',
                    'lunch', 'test preparation course', 'status']
    
    corr_matrix = data_encoded[corr_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
               center=0, square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
    plt.title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig('eda_plots/5_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: eda_plots/5_correlation_matrix.png")
    plt.close()
    
    # 6. Box Plots by Status
    print("6️⃣ Score Distribution by Pass/Fail Status...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    data.boxplot(column='math score', by='status', ax=axes[0])
    axes[0].set_title('Math Scores by Status')
    axes[0].set_xlabel('Status')
    axes[0].set_ylabel('Math Score')
    
    data.boxplot(column='reading score', by='status', ax=axes[1])
    axes[1].set_title('Reading Scores by Status')
    axes[1].set_xlabel('Status')
    axes[1].set_ylabel('Reading Score')
    
    data.boxplot(column='writing score', by='status', ax=axes[2])
    axes[2].set_title('Writing Scores by Status')
    axes[2].set_xlabel('Status')
    axes[2].set_ylabel('Writing Score')
    
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig('eda_plots/6_scores_by_status.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: eda_plots/6_scores_by_status.png")
    plt.close()
    
    # 7. Parental Education Impact
    print("7️⃣ Parental Education Impact...")
    parent_analysis = data.groupby('parental level of education').agg({
        'average_score': 'mean',
        'status': lambda x: (x == 'Pass').sum() / len(x) * 100
    }).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].barh(parent_analysis['parental level of education'], 
                parent_analysis['average_score'],
                color='#667eea', edgecolor='black')
    axes[0].set_xlabel('Average Score')
    axes[0].set_title('Average Score by Parental Education', fontweight='bold')
    
    axes[1].barh(parent_analysis['parental level of education'],
                parent_analysis['status'],
                color='#51cf66', edgecolor='black')
    axes[1].set_xlabel('Pass Rate (%)')
    axes[1].set_title('Pass Rate by Parental Education', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eda_plots/7_parental_education_impact.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: eda_plots/7_parental_education_impact.png")
    plt.close()
    
    # Create EDA Report
    print("\n📄 Creating EDA report...")
    with open('reports/3_eda_report.txt', 'w') as f:
        f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("KEY FINDINGS:\n\n")
        
        f.write("1. SCORE DISTRIBUTIONS:\n")
        f.write(f"   - Math: Mean={data['math score'].mean():.2f}, Std={data['math score'].std():.2f}\n")
        f.write(f"   - Reading: Mean={data['reading score'].mean():.2f}, Std={data['reading score'].std():.2f}\n")
        f.write(f"   - Writing: Mean={data['writing score'].mean():.2f}, Std={data['writing score'].std():.2f}\n\n")
        
        f.write("2. PASS/FAIL ANALYSIS:\n")
        pass_rate = (data['status'] == 'Pass').sum() / len(data) * 100
        f.write(f"   - Pass Rate: {pass_rate:.1f}%\n")
        f.write(f"   - Fail Rate: {100-pass_rate:.1f}%\n\n")
        
        f.write("3. TEST PREPARATION IMPACT:\n")
        prep_diff = prep_scores['completed'] - prep_scores['none']
        f.write(f"   - Score difference: +{prep_diff:.2f} points\n")
        f.write(f"   - Pass rate difference: +{prep_pass_rate['completed'] - prep_pass_rate['none']:.1f}%\n\n")
        
        f.write("4. CORRELATIONS:\n")
        f.write("   Top correlations with status:\n")
        status_corr = corr_matrix['status'].sort_values(ascending=False)
        for feat, val in list(status_corr.items())[1:4]:
            f.write(f"   - {feat}: {val:.3f}\n")
        
        f.write("\n5. VISUALIZATIONS CREATED:\n")
        f.write("   - 1_score_distributions.png\n")
        f.write("   - 2_pass_fail_analysis.png\n")
        f.write("   - 3_subject_comparison.png\n")
        f.write("   - 4_test_prep_impact.png\n")
        f.write("   - 5_correlation_matrix.png\n")
        f.write("   - 6_scores_by_status.png\n")
        f.write("   - 7_parental_education_impact.png\n")
    
    print("   ✅ EDA report saved!")
    
    print("\n📊 EDA Summary:")
    print(f"   Total visualizations: 7")
    print(f"   Pass rate: {pass_rate:.1f}%")
    print(f"   Test prep impact: +{prep_diff:.2f} points")
    
    return data

if __name__ == "__main__":
    data = exploratory_analysis()
    
    print("\n" + "="*60)
    print("✅ PHASE 3 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n👉 Next Step: Run '4_feature_selection.py'")
