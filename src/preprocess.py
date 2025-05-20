import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

def load_data(file_path):
    """
    Load the telco churn dataset
    """
    df = pd.read_csv("/home/kcm5750/Music/ChurnPrediction/data/telco_churn.csv")
    print(f"Loaded data with shape: {df.shape}")
    return df

def clean_data(df):
    """
    Clean the data by handling missing values and converting types
    """
    # Convert 'TotalCharges' to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop rows with missing values and make an explicit copy
    df = df.dropna().copy()

    # Convert binary variables to numeric
    binary_vars = ['Churn', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for var in binary_vars:
        df[var] = df[var].map({'Yes': 1, 'No': 0})

    return df

def feature_engineering(df):
    """
    Create new features that might be helpful for prediction
    """
    # Create tenure-related features
    df['TenureYears'] = df['tenure'] / 12
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], 
                              labels=['0-1 year', '1-2 years', '2-3 years', 
                                      '3-4 years', '4-5 years', '5+ years'])
    
    # Create ARPU (Average Revenue Per User)
    df['MonthlyARPU'] = df['MonthlyCharges'] / (df['Dependents'].astype(int) + 1)
    
    # Create contract-related feature
    df['IsLongTermContract'] = df['Contract'].apply(lambda x: 1 if x in ['One year', 'Two year'] else 0)
    
    return df

def preprocess_data(df, save_pipeline=False, pipeline_path=None):
    """
    Preprocess the data for modeling
    """
    # Separate features from target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Define numeric and categorical columns
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TenureYears', 'MonthlyARPU']
    categorical_features = [col for col in X.columns if X[col].dtype == 'object' and col != 'customerID']
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop columns that are not specified
    )
    
    # Create preprocessing pipeline
    preprocessing_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    # Fit and transform the data
    X_processed = preprocessing_pipeline.fit_transform(X)
    
    # Save the preprocessing pipeline if requested
    if save_pipeline and pipeline_path:
        os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
        joblib.dump(preprocessing_pipeline, pipeline_path)
        print(f"Preprocessing pipeline saved to {pipeline_path}")
    
    return X_processed, y, preprocessing_pipeline

def process_data_for_model(data_path, output_path=None, save_pipeline=True):
    """
    End-to-end function to process data for modeling
    """
    # Load data
    df = load_data(data_path)
    
    # Clean data
    df = clean_data(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Create pipeline path
    pipeline_path = os.path.join(output_path, 'preprocessing_pipeline.joblib') if output_path else None
    
    # Preprocess data
    X_processed, y, pipeline = preprocess_data(df, save_pipeline, pipeline_path)
    
    # Save processed data if output path is provided
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        
        # Save data as numpy arrays
        np.save(os.path.join(output_path, 'X_processed.npy'), X_processed)
        np.save(os.path.join(output_path, 'y.npy'), y.values)
        
        # Save original data with customer IDs
        customer_ids = df['customerID']
        pd.DataFrame({'customerID': customer_ids, 'Churn': y}).to_csv(
            os.path.join(output_path, 'customer_ids.csv'), index=False
        )
        
        print(f"Processed data saved to {output_path}")
    
    return X_processed, y, pipeline

if __name__ == "__main__":
    # Example usage
    data_path = "data/telco_churn.csv"
    output_path = "data/processed"
    
    X_processed, y, pipeline = process_data_for_model(data_path, output_path)
    print(f"Processed X shape: {X_processed.shape}")
    print(f"Processed y shape: {y.shape}")