"""
MASTER PIPELINE SCRIPT
Run all 6 phases in sequence
"""
import sys
import os
import subprocess

def run_pipeline():
    """
    Execute complete ML pipeline
    """
    print("\n" + "="*70)
    print(" "*15 + "STUDENT PERFORMANCE ANALYSIS")
    print(" "*20 + "COMPLETE ML PIPELINE")
    print("="*70 + "\n")
    
    phases = [
        ("1_data_collection.py", "Phase 1: Data Collection"),
        ("2_data_preprocessing.py", "Phase 2: Data Preprocessing"),
        ("3_exploratory_analysis.py", "Phase 3: Exploratory Data Analysis"),
        ("4_feature_selection.py", "Phase 4: Feature Selection"),
        ("5_model_building.py", "Phase 5: Model Building"),
        ("6_evaluation_insights.py", "Phase 6: Evaluation & Insights")
    ]
    
    for i, (script, description) in enumerate(phases, 1):
        print(f"\n{'='*70}")
        print(f"  EXECUTING: {description}")
        print(f"{'='*70}\n")
        
        # Check if script exists
        if not os.path.exists(script):
            print(f"❌ Error: {script} not found!")
            print(f"   Please ensure all phase scripts are in the current directory.")
            print(f"   Current directory: {os.getcwd()}")
            return False
        
        try:
            # Run script as separate process
            result = subprocess.run(
                [sys.executable, script],
                capture_output=False,
                text=True,
                check=True
            )
            
            print(f"\n✅ {description} completed successfully!")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error in {description}:")
            print(f"   Script exited with error code {e.returncode}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error in {description}:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # Final summary
    print("\n" + "="*70)
    print(" "*20 + "🎉 PIPELINE COMPLETED! 🎉")
    print("="*70)
    
    print("\n📊 SUMMARY:")
    print("   ✅ Phase 1: Data Collection - Complete")
    print("   ✅ Phase 2: Data Preprocessing - Complete")
    print("   ✅ Phase 3: Exploratory Analysis - Complete")
    print("   ✅ Phase 4: Feature Selection - Complete")
    print("   ✅ Phase 5: Model Building - Complete")
    print("   ✅ Phase 6: Evaluation & Insights - Complete")
    
    print("\n📁 GENERATED OUTPUTS:")
    print("   📂 data/")
    print("      - preprocessed_data.csv")
    print("   📂 models/")
    print("      - best_model.pkl")
    print("      - all_models.pkl")
    print("      - label_encoders.pkl")
    print("      - target_encoder.pkl")
    print("   📂 reports/")
    print("      - Phase reports (1-6)")
    print("      - summary_statistics.csv")
    print("   📂 eda_plots/")
    print("      - 13 visualization plots")
    
    print("\n🚀 NEXT STEPS:")
    print("   1. Review reports in 'reports/' directory")
    print("   2. Check visualizations in 'eda_plots/' directory")
    print("   3. Train model: python train_model.py")
    print("   4. Run dashboard: streamlit run app.py")
    
    print("\n" + "="*70 + "\n")
    
    return True

if __name__ == "__main__":
    success = run_pipeline()
    
    if success:
        print("✅ All phases completed successfully!")
        sys.exit(0)
    else:
        print("❌ Pipeline execution failed!")
        sys.exit(1)
