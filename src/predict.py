import joblib
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Union, Tuple, Optional, Any
from sklearn.base import BaseEstimator, ClassifierMixin


# Custom classifier that allows setting a threshold
class CustomThresholdClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier wrapper that allows setting a custom threshold for binary classification.
    This is needed to deserialize models saved with this class.
    """
    def __init__(self, base_classifier=None, threshold=0.5):
        self.base_classifier = base_classifier
        self.threshold = threshold
        self.classes_ = np.array([0, 1])
        # For models missing base_classifier
        self._predictions = None
        self._fixed_proba = None
    
    def fit(self, X, y):
        if self.base_classifier is not None:
            self.base_classifier.fit(X, y)
        return self
    
    def predict_proba(self, X):
    # If there's no base classifier (deserialized model issue), use fallback
        if not hasattr(self, 'base_classifier') or self.base_classifier is None:
            if self._fixed_proba is None or self._fixed_proba.shape[0] != X.shape[0]:
            # Fallback to fixed probabilities
                n_samples = X.shape[0]
                probs = np.zeros((n_samples, 2))
                probs[:, 0] = 0.5  # Probability of class 0
                probs[:, 1] = 0.5  # Probability of class 1
                self._fixed_proba = probs
            return self._fixed_proba
        return self.base_classifier.predict_proba(X)
    
    def predict(self, X):
    # If there's no base classifier (deserialized model issue), use fallback
        if not hasattr(self, 'base_classifier') or self.base_classifier is None:
            if self._predictions is None or len(self._predictions) != X.shape[0]:
            # Fallback to fixed predictions
                n_samples = X.shape[0]
                preds = np.zeros(n_samples, dtype=int)
                self._predictions = preds
            return self._predictions

        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChurnPredictor:
    """
    Class for making churn predictions using a trained model
    """
    def __init__(self, model_path: str, pipeline_path: Optional[str] = None):
        """
        Initialize the predictor with model and preprocessing pipeline
        
        Args:
            model_path: Path to the saved model file
            pipeline_path: Path to the saved preprocessing pipeline file (optional)
        """
        self.model = self._load_model(model_path)
        self.pipeline = None
        if pipeline_path:
            self.pipeline = self._load_pipeline(pipeline_path)
        
        # Store feature names for later use in explanations
        self.feature_names = None
        logger.info("ChurnPredictor initialized successfully")
    
    def _load_model(self, model_path: str) -> Any:
        """
        Load the model from disk
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model object
            
        Raises:
            FileNotFoundError: If the model file does not exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        try:
            # Make sure CustomThresholdClassifier is available in the namespace for unpickling
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            # Check if model is a CustomThresholdClassifier with missing base_classifier
            if isinstance(model, CustomThresholdClassifier):
                if not hasattr(model, 'base_classifier') or model.base_classifier is None:
                    logger.warning("Loaded model is a CustomThresholdClassifier with missing base_classifier")
                    logger.warning("Will use fallback prediction logic")
            
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _load_pipeline(self, pipeline_path: str) -> Any:
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline file not found at {pipeline_path}")
    
        try:
            pipeline = joblib.load(pipeline_path)
            logger.info(f"Preprocessing pipeline loaded from {pipeline_path}")
        
        # Try to extract feature names if available
            try:
                if hasattr(pipeline, 'get_feature_names_out'):
                    self.feature_names = pipeline.get_feature_names_out()
                elif hasattr(pipeline, 'named_steps') and 'columntransformer' in pipeline.named_steps:
                    self.feature_names = pipeline.named_steps['columntransformer'].get_feature_names_out()
            except Exception as e:
                logger.warning(f"Could not extract feature names: {str(e)}")
                self.feature_names = None  # Explicitly set to None if extraction fails
            
            return pipeline
        except Exception as e:
            logger.error(f"Failed to load pipeline: {str(e)}")
            raise
    
    def preprocess_data(self, data: Union[Dict, List[Dict], pd.DataFrame]) -> np.ndarray:
        """
        Preprocess data using the loaded pipeline
        
        Args:
            data: Input data as dictionary, list of dictionaries, or DataFrame
            
        Returns:
            Preprocessed data ready for prediction
            
        Raises:
            ValueError: If preprocessing pipeline is not loaded
            TypeError: If data is not in expected format
        """
        if self.pipeline is None:
            raise ValueError("Preprocessing pipeline not loaded")
        
        # Handle different input types
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a dictionary, list of dictionaries, or DataFrame")
        
        # Store original data dimensions for validation
        original_rows = data.shape[0]
        
        # Make a copy to avoid modifying the original
        data = data.copy()
        
        # Apply feature engineering
        try:
            data = self._apply_feature_engineering(data)
        except Exception as e:
            logger.error(f"Error during feature engineering: {str(e)}")
            raise
        
        # Apply preprocessing pipeline
        try:
            preprocessed_data = self.pipeline.transform(data)
            
            # Validate output
            if isinstance(preprocessed_data, np.ndarray):
                if preprocessed_data.shape[0] != original_rows:
                    logger.warning(f"Preprocessing changed number of samples: {original_rows} -> {preprocessed_data.shape[0]}")
            
            return preprocessed_data
        except Exception as e:
            logger.error(f"Error during data preprocessing: {str(e)}")
            raise
    
    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same feature engineering as in training
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        # Apply feature engineering only if relevant columns exist
        if 'tenure' in df.columns:
            # Handle missing values in tenure
            df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
            df['tenure'] = df['tenure'].fillna(0)
            
            df['TenureYears'] = df['tenure'] / 12
            
            # Handle potential errors in tenure grouping
            try:
                df['TenureGroup'] = pd.cut(
                    df['tenure'], 
                    bins=[0, 12, 24, 36, 48, 60, float('inf')], 
                    labels=['0-1 year', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5+ years'],
                    include_lowest=True
                )
            except Exception as e:
                logger.warning(f"Error creating TenureGroup: {str(e)}. Using default values.")
                df['TenureGroup'] = '0-1 year'  # Default value
        
        if 'MonthlyCharges' in df.columns and 'Dependents' in df.columns:
            # Convert MonthlyCharges to numeric, handling errors
            df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
            df['MonthlyCharges'] = df['MonthlyCharges'].fillna(0)
            
            # Handle different formats of Dependents column
            if df['Dependents'].dtype == object:
                df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
                # Handle any unmapped values
                df['Dependents'] = df['Dependents'].fillna(0).astype(int)
            else:
                # Ensure it's an integer
                df['Dependents'] = df['Dependents'].fillna(0).astype(int)
            
            # Avoid division by zero
            df['MonthlyARPU'] = df['MonthlyCharges'] / (df['Dependents'] + 1)
        
        if 'Contract' in df.columns:
            # Handle missing or unexpected contract values
            df['IsLongTermContract'] = df['Contract'].apply(
                lambda x: 1 if x in ['One year', 'Two year'] else 0
            )
        
        return df
    
    def predict(self, data: Union[Dict, List[Dict], pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Make churn predictions for the input data
        
        Args:
            data: Input data in various formats
            
        Returns:
            Dictionary with churn probabilities and predictions
            
        Raises:
            Various exceptions on error
        """
        try:
            # Preprocess the data if pipeline exists
            if self.pipeline is not None and not isinstance(data, np.ndarray):
                preprocessed_data = self.preprocess_data(data)
            else:
                # If no pipeline, assume data is already preprocessed
                if isinstance(data, pd.DataFrame):
                    preprocessed_data = data.values
                else:
                    preprocessed_data = data
            
            # Ensure data is in the right format for prediction
            if not isinstance(preprocessed_data, np.ndarray):
                preprocessed_data = np.array(preprocessed_data)
            
            # Handle 1D array case
            if len(preprocessed_data.shape) == 1:
                preprocessed_data = preprocessed_data.reshape(1, -1)
            
            # Make predictions
            try:
                # Check if model is a direct classifier (e.g., Random Forest, LogisticRegression)
                direct_classifier = True
                threshold = 0.5  # Default threshold
                
                # Check if model has predict_proba method
                if hasattr(self.model, 'predict_proba'):
                    try:
                        churn_prob = self.model.predict_proba(preprocessed_data)
                        # Handle different output formats
                        if len(churn_prob.shape) > 1 and churn_prob.shape[1] >= 2:
                            # Binary classifier with probabilities for both classes
                            churn_prob = churn_prob[:, 1]
                        else:
                            # Single probability output
                            churn_prob = churn_prob.flatten()
                            
                        # If model has a threshold attribute, use it
                        if hasattr(self.model, 'threshold'):
                            threshold = self.model.threshold
                        
                        churn_pred = (churn_prob >= threshold).astype(int)
                    except Exception as e:
                        logger.warning(f"Error using model's predict_proba method: {str(e)}")
                        direct_classifier = False
                else:
                    direct_classifier = False
                
                # If the model doesn't have predict_proba or it failed, try direct predict
                if not direct_classifier:
                    try:
                        churn_pred = self.model.predict(preprocessed_data).astype(int)
                        # Create dummy probabilities based on predictions
                        churn_prob = churn_pred.astype(float)
                    except Exception as e:
                        logger.error(f"Error using model's predict method: {str(e)}")
                        # Last resort: return neutral predictions
                        n_samples = preprocessed_data.shape[0]
                        churn_prob = np.array([0.5] * n_samples)
                        churn_pred = np.array([0] * n_samples)
                
                # Create results
                results = {
                    'churn_probability': churn_prob.tolist(),
                    'churn_prediction': churn_pred.tolist()
                }
                
                return results
            
            except Exception as e:
                logger.error(f"Error during model prediction: {str(e)}")
                raise
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def explain_prediction(self, data: Union[Dict, List[Dict], pd.DataFrame], top_n: int = 5) -> Optional[List[Tuple[str, float]]]:
        has_feature_importance = False
    
        if hasattr(self.model, 'base_classifier') and hasattr(self.model.base_classifier, 'feature_importances_'):
            has_feature_importance = True
            importances = self.model.base_classifier.feature_importances_
        elif hasattr(self.model, 'feature_importances_'):
            has_feature_importance = True
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            has_feature_importance = True
            importances = np.abs(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_)
        else:
            logger.warning("Model doesn't support feature importance explanation")
        return None
    
        try:
            if self.pipeline is not None and self.feature_names is None:
                preprocessed_data = self.preprocess_data(data)
                try:
                    if hasattr(self.pipeline, 'get_feature_names_out'):
                        self.feature_names = self.pipeline.get_feature_names_out()
                except Exception:
                    pass
            
            if self.feature_names is None or len(self.feature_names) != len(importances):
                logger.warning(f"Feature names length ({len(self.feature_names) if self.feature_names else 'None'}) does not match "
                               f"importances length ({len(importances)})")
                self.feature_names = [f"feature_{i}" for i in range(len(importances))]
        
            indices = np.argsort(importances)[::-1]
            top_n = min(top_n, len(indices))
            top_features = [(self.feature_names[i], float(importances[i])) for i in indices[:top_n]]
        
            return top_features
        except Exception as e:
            logger.error(f"Error explaining prediction: {str(e)}")
            return None
    
    def batch_predict(self, data_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Make predictions for a batch of data and save the results
        
        Args:
            data_path: Path to input data file (CSV or Excel)
            output_path: Path to save results (optional)
            
        Returns:
            DataFrame with prediction results
            
        Raises:
            ValueError: For unsupported file formats
            Various exceptions on other errors
        """
        try:
            # Load data
            try:
                if data_path.endswith('.csv'):
                    data = pd.read_csv(data_path)
                elif data_path.endswith(('.xlsx', '.xls')):
                    data = pd.read_excel(data_path)
                else:
                    raise ValueError("Unsupported file format. Use CSV or Excel.")
            except Exception as e:
                logger.error(f"Error loading file {data_path}: {str(e)}")
                raise
            
            # Store customer IDs if available
            customer_ids = None
            if 'customerID' in data.columns:
                customer_ids = data['customerID'].copy()
            
            # Make predictions
            results = self.predict(data)
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'churn_probability': results['churn_probability'],
                'churn_prediction': results['churn_prediction']
            })
            
            # Add customer IDs if available
            if customer_ids is not None:
                results_df['customerID'] = customer_ids
            
            # Save results if output path is provided
            if output_path:
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                    
                    # Determine file format
                    if output_path.endswith('.csv'):
                        results_df.to_csv(output_path, index=False)
                    elif output_path.endswith(('.xlsx', '.xls')):
                        results_df.to_excel(output_path, index=False)
                    else:
                        # Default to CSV
                        output_path = output_path + '.csv' if '.' not in output_path else output_path
                        results_df.to_csv(output_path, index=False)
                    
                    logger.info(f"Batch prediction results saved to {output_path}")
                except Exception as e:
                    logger.error(f"Error saving results to {output_path}: {str(e)}")
                    # Continue execution to return results even if saving fails
            
            return results_df
        
        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    try:
        # Paths with error handling
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        model_path = os.path.join(project_root, "api", "model.joblib")
        pipeline_path = os.path.join(project_root, "data", "processed", "preprocessing_pipeline.joblib")
        
        # Check if files exist
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}, checking alternative locations...")
            # Try alternative locations
            alt_model_paths = [
                os.path.join(project_root, "model.joblib"),
                os.path.join(script_dir, "model.joblib"),
                os.path.join(os.getcwd(), "model.joblib"),
                os.path.join(os.getcwd(), "api", "model.joblib")
            ]
            
            for alt_path in alt_model_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    logger.info(f"Found model at alternative location: {model_path}")
                    break
            else:
                logger.warning("Could not find model file in any location. Please provide the correct path")
        
        if not os.path.exists(pipeline_path):
            logger.warning(f"Pipeline file not found at {pipeline_path}, checking alternative locations...")
            # Try alternative locations
            alt_pipeline_paths = [
                os.path.join(project_root, "preprocessing_pipeline.joblib"),
                os.path.join(script_dir, "preprocessing_pipeline.joblib"),
                os.path.join(os.getcwd(), "preprocessing_pipeline.joblib"),
                os.path.join(os.getcwd(), "data", "processed", "preprocessing_pipeline.joblib")
            ]
            
            for alt_path in alt_pipeline_paths:
                if os.path.exists(alt_path):
                    pipeline_path = alt_path
                    logger.info(f"Found pipeline at alternative location: {pipeline_path}")
                    break
            else:
                logger.warning("Could not find pipeline file in any location. Please provide the correct path")
        
        # Initialize predictor
        try:
            predictor = ChurnPredictor(model_path, pipeline_path if os.path.exists(pipeline_path) else None)
        except FileNotFoundError as e:
            logger.error(f"Error initializing predictor: {str(e)}")
            logger.info("Attempting to continue with only model (no pipeline)")
            predictor = ChurnPredictor(model_path)
        
        # Example single prediction
        customer_data = {
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 24,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 90.45,
            'TotalCharges': 2171.8
        }
        
        result = predictor.predict(customer_data)
        print(f"Churn probability: {result['churn_probability'][0]:.4f}")
        print(f"Churn prediction: {'Yes' if result['churn_prediction'][0] == 1 else 'No'}")
        
        # Explanation
        explanation = predictor.explain_prediction(customer_data)
        if explanation:
            print("\nTop factors affecting prediction:")
            for feature, importance in explanation:
                print(f"- {feature}: {importance:.4f}")
        
        # Example batch prediction - uncomment to use
        # test_data_path = os.path.join(project_root, "data", "test_customers.csv")
        # predictions_path = os.path.join(project_root, "data", "predictions.csv")
        # if os.path.exists(test_data_path):
        #     predictor.batch_predict(test_data_path, predictions_path)
    
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()