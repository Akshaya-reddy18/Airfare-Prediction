import os
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from utils import create_experiment_directory, save_model_metadata
from sklearn.model_selection import train_test_split

def main():
    # Create experiment directory
    experiment_dir = create_experiment_directory()
    print(f"Created experiment directory: {experiment_dir}")
    
    # Initialize components
    preprocessor = DataPreprocessor()
    model_trainer = ModelTrainer()
    model_evaluator = ModelEvaluator()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = preprocessor.prepare_data('data/Clean_Dataset.csv', target_column='price')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if X_train is None:
        print("Error: Failed to preprocess data")
        return
    
    print("Data preprocessing completed successfully")
    
    # Save preprocessor state
    preprocessor.save_preprocessor('models/trained_models/preprocessor.joblib')
    print(f"Saved preprocessor state to models/trained_models/preprocessor.joblib")
    
    # Train models
    print("Training models...")
    results = model_trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Save only the best model
    best_model_name = max(results, key=lambda k: results[k]['r2'])
    best_model = model_trainer.models[best_model_name]
    model_trainer.save_model(best_model, 'best_model')
    
    print(f"\nTraining completed. Best model: {best_model_name}")
    print(f"Best model R² score: {results[best_model_name]['r2']:.4f}")
    
    # Save model metadata
    for model_name, model in model_trainer.models.items():
        save_model_metadata(model, model_name, experiment_dir)
    
    print("Model training completed successfully")
    
    # Evaluate models
    print("Evaluating models...")
    evaluation_results = model_evaluator.evaluate_models(
        model_trainer.models,
        X_test,
        y_test,
        X_train.columns.tolist()
    )
    
    print("Model evaluation completed successfully")
    
    # Print results
    print("\nModel Performance Results:")
    for model_name, metrics in evaluation_results.items():
        print(f"\n{model_name}:")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")

if __name__ == "__main__":
    main() 