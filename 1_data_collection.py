"""
PHASE 1: DATA COLLECTION
Collect student performance data from various sources
"""

import pandas as pd
import os

def collect_data():
    """
    Collect data from Kaggle dataset
    """
    print("="*60)
    print("PHASE 1: DATA COLLECTION")
    print("="*60)
    
    # Check if data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Load the dataset
    print("\n📊 Loading student performance dataset...")
    
    try:
        data = pd.read_csv('data/student_data.csv')
        print(f"✅ Data loaded successfully!")
        print(f"   Total records: {len(data)}")
        print(f"   Total features: {len(data.columns)}")
        
        # Display basic information
        print("\n📋 Dataset Overview:")
        print(data.info())
        
        print("\n📊 Sample Data (First 5 rows):")
        print(data.head())
        
        print("\n📈 Basic Statistics:")
        print(data.describe())
        
        # Check for missing values
        print("\n🔍 Missing Values Check:")
        missing = data.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")
        
        # Save raw data summary
        print("\n💾 Saving data collection report...")
        with open('reports/1_data_collection_report.txt', 'w') as f:
            f.write("DATA COLLECTION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total Records: {len(data)}\n")
            f.write(f"Total Features: {len(data.columns)}\n")
            f.write(f"\nColumns:\n{data.columns.tolist()}\n")
            f.write(f"\nData Types:\n{data.dtypes}\n")
            f.write(f"\nMissing Values:\n{missing}\n")
        
        print("✅ Data collection report saved!")
        
        return data
        
    except FileNotFoundError:
        print("❌ Error: 'data/student_data.csv' not found!")
        print("   Please download the dataset from Kaggle first.")
        return None

if __name__ == "__main__":
    # Create reports directory
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    # Collect data
    data = collect_data()
    
    if data is not None:
        print("\n" + "="*60)
        print("✅ PHASE 1 COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n👉 Next Step: Run '2_data_preprocessing.py'")
