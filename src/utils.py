import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config, config_path):
    """Save configuration to JSON file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def create_experiment_directory(base_dir='experiments'):
    """Create a new experiment directory with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f'experiment_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def save_model_metadata(model, model_name, experiment_dir):
    """Save model metadata to experiment directory"""
    metadata = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'parameters': model.get_params() if hasattr(model, 'get_params') else {},
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(experiment_dir, f'{model_name}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def format_currency(amount):
    """Format amount as Indian Rupees with thousands separator"""
    return f"₹{amount:,.2f}"

def calculate_percentage_change(old_value, new_value):
    """Calculate percentage change between two values"""
    return ((new_value - old_value) / old_value) * 100

def get_feature_importance(model, feature_names):
    """Get feature importance from model"""
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        return dict(zip(feature_names, model.coef_))
    return None

def create_correlation_matrix(df, output_path=None):
    """Create and optionally save correlation matrix"""
    corr_matrix = df.corr()
    
    if output_path:
        corr_matrix.to_csv(output_path)
    
    return corr_matrix

def generate_summary_statistics(df, output_path=None):
    """Generate and optionally save summary statistics"""
    summary = {
        'basic_stats': df.describe().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict()
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=4)
    
    return summary 