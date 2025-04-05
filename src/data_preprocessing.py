import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_categories = {}
        self.feature_names = None
        
    def load_data(self, file_path):
        """Load the dataset from CSV file"""
        try:
            df = pd.read_csv(file_path)
            # Drop the index column if it exists
            if 'Unnamed: 0' in df.columns:
                df = df.drop('Unnamed: 0', axis=1)
            
            # Create flight identifier if it doesn't exist
            if 'flight' not in df.columns:
                df['flight'] = df['airline'] + '-' + df['source_city'] + '-' + df['destination_city']
            
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Fill numeric columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Fill categorical columns with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
        
        return df
    
    def encode_categorical_features(self, df, categorical_columns):
        """Encode categorical features using Label Encoding"""
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                self.categorical_categories[column] = df[column].unique().tolist()
                df[column] = self.label_encoders[column].fit_transform(df[column])
            else:
                # Handle unseen labels by adding them to the encoder
                unseen_labels = set(df[column].unique()) - set(self.categorical_categories[column])
                if unseen_labels:
                    # Add unseen labels to the categories
                    self.categorical_categories[column].extend(list(unseen_labels))
                    # Retrain the encoder with all categories
                    self.label_encoders[column].fit(self.categorical_categories[column])
                df[column] = self.label_encoders[column].transform(df[column])
        return df
    
    def scale_numeric_features(self, df, numeric_columns):
        """Scale numeric features using StandardScaler"""
        df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        return df
    
    def prepare_data(self, file_path, target_column='price'):
        """Prepare data for model training"""
        # Load data
        df = self.load_data(file_path)
        if df is None:
            return None, None
        
        # Drop the target column to get features
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Identify numeric and categorical columns
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        # Encode categorical features
        X = self.encode_categorical_features(X, categorical_columns)
        
        # Scale numeric features
        if len(numeric_columns) > 0:
            X[numeric_columns] = self.scaler.fit_transform(X[numeric_columns])
        
        print("Feature names used for training:", self.feature_names)
        return X, y
    
    def preprocess_data(self, file_path, target_column):
        """Main preprocessing pipeline"""
        # Load data
        df = self.load_data(file_path)
        if df is None:
            return None
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Identify numeric and categorical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Remove target column from numeric columns if present
        if target_column in numeric_columns:
            numeric_columns = numeric_columns.drop(target_column)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, categorical_columns)
        
        # Scale numeric features
        df = self.scale_numeric_features(df, numeric_columns)
        
        # Prepare data for training
        X_train, y_train = self.prepare_data(file_path, target_column)
        
        return X_train, y_train
    
    def save_preprocessor(self, file_path):
        """Save preprocessor state to disk"""
        preprocessor_state = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'categorical_categories': self.categorical_categories,
            'feature_names': self.feature_names
        }
        joblib.dump(preprocessor_state, file_path)
    
    def load_preprocessor(self, file_path):
        """Load preprocessor state from disk"""
        if os.path.exists(file_path):
            preprocessor_state = joblib.load(file_path)
            self.scaler = preprocessor_state['scaler']
            self.label_encoders = preprocessor_state['label_encoders']
            self.categorical_categories = preprocessor_state['categorical_categories']
            self.feature_names = preprocessor_state['feature_names']
            return True
        return False

    def handle_unseen_labels(self, df, categorical_columns):
        """Handle unseen labels in categorical features during prediction"""
        for column in categorical_columns:
            if column in self.label_encoders:
                unseen_labels = set(df[column].unique()) - set(self.categorical_categories[column])
                if unseen_labels:
                    # Add unseen labels to the categories
                    self.categorical_categories[column].extend(list(unseen_labels))
                    # Retrain the encoder with all categories
                    self.label_encoders[column].fit(self.categorical_categories[column])
                df[column] = self.label_encoders[column].transform(df[column])
        return df 