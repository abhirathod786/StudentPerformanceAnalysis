import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

class ModelExplainer:
    """
    Explain model predictions using SHAP
    Handles both tree-based and linear models
    """
    
    def __init__(self, model, X_train):
        """
        Initialize SHAP explainer based on model type
        """
        self.model = model
        self.X_train = X_train
        
        # Check model type and use appropriate explainer
        if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier)):
            # Use TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(model)
            self.explainer_type = 'tree'
        else:
            # Use KernelExplainer for linear models (Logistic Regression, SVM, etc.)
            # Sample background data for faster computation (100 samples)
            background = shap.sample(X_train, min(100, len(X_train)))
            self.explainer = shap.KernelExplainer(model.predict_proba, background)
            self.explainer_type = 'kernel'
        
    def explain_prediction(self, X_sample, feature_names):
        """
        Explain a single prediction
        """
        # Get SHAP values
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X_sample)
        else:
            shap_values = self.explainer.shap_values(X_sample)
        
        # For binary classification, take positive class (Pass)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        
        # Convert to numpy array if needed
        shap_values = np.array(shap_values)
        
        # Flatten if needed and ensure we have the right shape
        if shap_values.ndim == 1:
            # Already 1D, use directly
            shap_flat = shap_values
        elif shap_values.ndim == 2:
            # 2D array, take first row
            shap_flat = shap_values[0]
        else:
            # 3D or higher, flatten to 1D
            shap_flat = shap_values.flatten()
        
        # Get feature contributions
        contributions = {}
        for i, feature in enumerate(feature_names):
            if i < len(shap_flat):
                contrib_value = float(shap_flat[i])
                contributions[feature] = contrib_value
        
        # Sort by absolute contribution
        sorted_contributions = dict(sorted(contributions.items(), 
                                          key=lambda x: abs(x[1]), 
                                          reverse=True))
        
        return sorted_contributions
    
    def plot_force_plot(self, X_sample, feature_names, save_path='force_plot.png'):
        """
        Create SHAP force plot
        """
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X_sample)
        else:
            shap_values = self.explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        
        # Get expected value
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            expected_value = self.explainer.expected_value[1] if len(self.explainer.expected_value) > 1 else self.explainer.expected_value[0]
        else:
            expected_value = self.explainer.expected_value
        
        # Create force plot
        shap.force_plot(
            expected_value,
            shap_values[0] if len(shap_values.shape) > 1 else shap_values,
            X_sample[0] if len(X_sample.shape) > 1 else X_sample,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    
    def plot_waterfall(self, X_sample, feature_names):
        """
        Create waterfall plot showing feature contributions
        """
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X_sample)
        else:
            shap_values = self.explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        
        # Convert to numpy array if needed
        shap_values = np.array(shap_values)
        
        # Flatten if needed and ensure we have the right shape
        if shap_values.ndim == 1:
            shap_flat = shap_values
        elif shap_values.ndim == 2:
            shap_flat = shap_values[0]
        else:
            shap_flat = shap_values.flatten()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get contributions
        contributions = {}
        for i, feature in enumerate(feature_names):
            if i < len(shap_flat):
                contrib_value = float(shap_flat[i])
                contributions[feature] = contrib_value
        
        # Sort by absolute value
        sorted_items = sorted(contributions.items(), 
                            key=lambda x: abs(x[1]), 
                            reverse=True)
        
        features = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # Create colors (red for negative, green for positive)
        colors = ['#ff6b6b' if v < 0 else '#51cf66' for v in values]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11, fontweight='bold')
        ax.set_title('Feature Contribution to Prediction', fontsize=13, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v, i, f' {v:.3f}', 
                   va='center', 
                   ha='left' if v > 0 else 'right',
                   fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def get_explanation_text(self, contributions, prediction, feature_values, feature_names_map):
        """
        Generate human-readable explanation
        """
        # Get top 3 contributing features
        top_features = list(contributions.items())[:3]
        
        explanation = []
        
        if prediction == 1:  # Pass
            explanation.append("✅ **Prediction: PASS**\n")
            explanation.append("**Key factors supporting success:**\n")
        else:  # Fail
            explanation.append("❌ **Prediction: FAIL**\n")
            explanation.append("**Key factors indicating risk:**\n")
        
        for feature, contribution in top_features:
            # Get readable name
            readable_name = feature_names_map.get(feature, feature)
            
            # Create explanation
            impact = "positively" if contribution > 0 else "negatively"
            strength = "strongly" if abs(contribution) > 0.3 else "moderately"
            
            explanation.append(
                f"- **{readable_name}**: {strength} {impact} impacts prediction "
                f"(contribution: {contribution:.3f})"
            )
        
        return "\n".join(explanation)
    
    def plot_summary_plot(self, X_test, feature_names):
        """
        Create SHAP summary plot for all predictions
        """
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X_test)
        else:
            # For kernel explainer, use smaller sample for speed
            sample_size = min(50, len(X_test))
            X_sample = X_test[:sample_size]
            shap_values = self.explainer.shap_values(X_sample)
            X_test = X_sample
        
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, 
                         feature_names=feature_names,
                         show=False)
        plt.tight_layout()
        return plt.gcf()
