import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import joblib
import os
import pandas as pd

class ModelTrainer:
    def __init__(self, models_dir='models/trained_models'):
        self.models_dir = models_dir
        self.models = {}
        self.metrics = {}
        self.best_model = None
        self.best_model_name = None
        
        # Create models directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
    
    def train_linear_regression(self, X_train, y_train):
        """Train Linear Regression model with non-negative constraint"""
        model = LinearRegression(positive=True)  # Ensure non-negative coefficients
        model.fit(X_train, y_train)
        self.models['linear_regression'] = model
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model with optimized parameters for smaller size"""
        param_grid = {
            'n_estimators': [50, 100],  # Reduced number of trees
            'max_depth': [8, 10],       # Reduced max depth
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt', 'log2']  # Added feature selection
        }
        
        base_model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True
        )
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        self.models['random_forest'] = best_model
        return best_model
    
    def train_gradient_boosting(self, X_train, y_train):
        """Train Gradient Boosting model"""
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['gradient_boosting'] = model
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model with hyperparameter tuning"""
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42
        )
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        self.models['xgboost'] = best_model
        return best_model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance with non-negative predictions"""
        y_pred = model.predict(X_test)
        
        # Ensure predictions are non-negative and realistic
        y_pred = np.maximum(y_pred, 1000)  # Minimum price of ₹1000
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate percentage of predictions within 20% of actual price
        within_20_percent = np.mean(np.abs((y_test - y_pred) / y_test) <= 0.2) * 100
        
        self.metrics[model_name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'within_20_percent': within_20_percent
        }
        
        print(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, Within 20%: {within_20_percent:.2f}%")
        
        return mse, r2
    
    def save_model(self, model, model_name):
        """Save trained model to disk"""
        file_path = os.path.join(self.models_dir, f'{model_name}.joblib')
        joblib.dump(model, file_path)
        print(f"Saved model to {file_path}")
    
    @staticmethod
    def load_model(model_path):
        """Load trained model from disk"""
        return joblib.load(model_path)
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate all models and select the best one"""
        results = {}
        
        # Train and evaluate Linear Regression
        lr_model = self.train_linear_regression(X_train, y_train)
        lr_mse, lr_r2 = self.evaluate_model(lr_model, X_test, y_test, 'linear_regression')
        self.save_model(lr_model, 'linear_regression')
        results['linear_regression'] = {'mse': lr_mse, 'r2': lr_r2}
        
        # Train and evaluate Random Forest
        rf_model = self.train_random_forest(X_train, y_train)
        rf_mse, rf_r2 = self.evaluate_model(rf_model, X_test, y_test, 'random_forest')
        self.save_model(rf_model, 'random_forest')
        results['random_forest'] = {'mse': rf_mse, 'r2': rf_r2}
        
        # Train and evaluate Gradient Boosting
        gb_model = self.train_gradient_boosting(X_train, y_train)
        gb_mse, gb_r2 = self.evaluate_model(gb_model, X_test, y_test, 'gradient_boosting')
        self.save_model(gb_model, 'gradient_boosting')
        results['gradient_boosting'] = {'mse': gb_mse, 'r2': gb_r2}
        
        # Train and evaluate XGBoost
        xgb_model = self.train_xgboost(X_train, y_train)
        xgb_mse, xgb_r2 = self.evaluate_model(xgb_model, X_test, y_test, 'xgboost')
        self.save_model(xgb_model, 'xgboost')
        results['xgboost'] = {'mse': xgb_mse, 'r2': xgb_r2}
        
        # Select the best model based on R² score
        best_model_name = max(results, key=lambda k: results[k]['r2'])
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        print(f"\nBest model: {best_model_name} with R² score: {results[best_model_name]['r2']:.4f}")
        
        # Save the best model as the default model
        self.save_model(self.best_model, 'best_model')
        
        return results
    
    def predict_with_ensemble(self, X):
        """Make predictions using an ensemble of models with non-negative constraint"""
        predictions = []
        
        for model_name, model in self.models.items():
            pred = model.predict(X)
            pred = np.maximum(pred, 1000)  # Ensure non-negative predictions
            predictions.append(pred)
        
        # Average the predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred