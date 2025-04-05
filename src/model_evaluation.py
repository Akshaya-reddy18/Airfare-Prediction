import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

class ModelEvaluator:
    def __init__(self, results_dir='models/evaluation_results'):
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate various evaluation metrics"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        return metrics
    
    def plot_prediction_vs_actual(self, y_true, y_pred, model_name):
        """Plot predicted vs actual values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predicted vs Actual Values - {model_name}')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_prediction_vs_actual.png'))
        plt.close()
    
    def plot_residuals(self, y_true, y_pred, model_name):
        """Plot residuals"""
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot - {model_name}')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_residuals.png'))
        plt.close()
    
    def plot_feature_importance(self, model, feature_names, model_name):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            plt.figure(figsize=(12, 6))
            sns.barplot(x=importance, y=feature_names)
            plt.title(f'Feature Importance - {model_name}')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(self.results_dir, f'{model_name}_feature_importance.png'))
            plt.close()
    
    def generate_evaluation_report(self, models_results):
        """Generate comprehensive evaluation report"""
        report = pd.DataFrame(models_results).T
        report.to_csv(os.path.join(self.results_dir, 'evaluation_report.csv'))
        
        # Create summary plot
        plt.figure(figsize=(12, 6))
        report['r2'].plot(kind='bar')
        plt.title('R² Score Comparison Across Models')
        plt.xlabel('Model')
        plt.ylabel('R² Score')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_comparison.png'))
        plt.close()
    
    def evaluate_models(self, models, X_test, y_test, feature_names):
        """Evaluate multiple models and generate comprehensive report"""
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred)
            results[name] = metrics
            
            # Generate visualizations
            self.plot_prediction_vs_actual(y_test, y_pred, name)
            self.plot_residuals(y_test, y_pred, name)
            self.plot_feature_importance(model, feature_names, name)
        
        # Generate evaluation report
        self.generate_evaluation_report(results)
        
        return results 