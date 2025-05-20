import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class CustomThresholdClassifier:
    def __init__(self, base_classifier=None, threshold=0.5):
        self.base_classifier = base_classifier
        self.threshold = threshold
        self._fixed_proba = None
        self._predictions = None

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_fixed_proba' not in state:
            self._fixed_proba = None
        if '_predictions' not in state:
            self._predictions = None         
    def predict_proba(self, X):
        return self.base_classifier.predict_proba(X)
                    
    def predict(self, X):
        probas = self.base_classifier.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)
                    
    def fit(self, X, y):
        self.base_classifier.fit(X, y)
        return self
                    
                # For model saving compatibility
    def __getattr__(self, name):
        return getattr(self.base_classifier, name)##
            
#model = CustomThresholdClassifier(original_model, threshold)
#logger.info(f"Applied optimal threshold of {threshold:.2f} to the model")

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
    logger.info(f"Training set class distribution: {np.bincount(y_train)}")
    logger.info(f"Testing set class distribution: {np.bincount(y_test)}")
    return X_train, X_test, y_train, y_test

def handle_class_imbalance(X_train, y_train, method='smote', random_state=42):
    """
    Handle class imbalance in the training data
    
    Parameters:
    method: str, one of 'smote', 'undersampling', 'smoteenn', 'smotetomek', 'class_weight', None
    """
    logger.info(f"Handling class imbalance using method: {method}")
    
    if method == 'smote':
        # Oversample the minority class
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE - Class distribution: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled
    
    elif method == 'undersampling':
        # Undersample the majority class
        undersampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
        logger.info(f"After undersampling - Class distribution: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled
    
    elif method == 'smoteenn':
        # Combination of SMOTE and Edited Nearest Neighbors
        smoteenn = SMOTEENN(random_state=random_state)
        X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)
        logger.info(f"After SMOTEENN - Class distribution: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled
    
    elif method == 'smotetomek':
        # Combination of SMOTE and Tomek links
        smotetomek = SMOTETomek(random_state=random_state)
        X_resampled, y_resampled = smotetomek.fit_resample(X_train, y_train)
        logger.info(f"After SMOTETomek - Class distribution: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled
    
    elif method == 'class_weight':
        # Return original data, class weights will be set in the model
        logger.info("Using class_weight parameter in the model")
        return X_train, y_train
    
    else:
        # No handling of class imbalance
        logger.info("No class imbalance handling applied")
        return X_train, y_train

def train_model(X_train, y_train, model_type='random_forest', param_grid=None, cv=5, 
                class_imbalance_method=None, threshold=0.5):
    """
    Train a model with hyperparameter tuning and class imbalance handling
    """
    logger.info(f"Training {model_type} model")
    
    # Define the model based on type
    if model_type == 'random_forest':
        if class_imbalance_method == 'class_weight':
            model = RandomForestClassifier(random_state=42, class_weight='balanced')
        else:
            model = RandomForestClassifier(random_state=42)
            
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
    
    elif model_type == 'gradient_boosting':
        if class_imbalance_method == 'class_weight':
            # Note: GradientBoostingClassifier doesn't have class_weight param
            # Using scale_pos_weight instead
            # Calculate class weight
            n_samples = len(y_train)
            n_classes = np.bincount(y_train)
            scale_pos_weight = n_classes[0] / n_classes[1]
            model = GradientBoostingClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
        else:
            model = GradientBoostingClassifier(random_state=42)
            
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
    
    elif model_type == 'logistic_regression':
        if class_imbalance_method == 'class_weight':
            model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        else:
            model = LogisticRegression(random_state=42, max_iter=1000)
            
        if param_grid is None:
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2']
            }
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Use stratified k-fold for cross-validation with imbalanced data
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Perform grid search for hyperparameter tuning
    # Use F1 score or AUC as the scoring metric for imbalanced data
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=cv_strategy, 
        scoring='f1', 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # If threshold is different from 0.5, create a custom threshold model
    if threshold != 0.5:
        original_model = best_model
        
        # Create a custom threshold model
        class CustomThresholdClassifier:
            def __init__(self, base_estimator, threshold):
                self.base_estimator = base_estimator
                self.threshold = threshold
                
            def predict_proba(self, X):
                return self.base_estimator.predict_proba(X)
                
            def predict(self, X):
                probas = self.base_estimator.predict_proba(X)[:, 1]
                return (probas >= self.threshold).astype(int)
                
            def fit(self, X, y):
                self.base_estimator.fit(X, y)
                return self
                
            # For model saving compatibility
            def __getattr__(self, name):
                return getattr(self.base_estimator, name)
        
        best_model = CustomThresholdClassifier(original_model, threshold)
        logger.info(f"Applied custom threshold of {threshold} to the model")
    
    return best_model, grid_search.best_params_

def find_optimal_threshold(y_true, y_prob):
    """
    Find the optimal threshold for classification based on F1 score
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        report = classification_report(y_true, y_pred, output_dict=True)
        f1 = report['1']['f1-score']  # F1 score for the positive class
        f1_scores.append(f1)
    
    # Find threshold with highest F1 score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    logger.info(f"Optimal threshold: {best_threshold:.2f} with F1 score: {best_f1:.4f}")
    
    # Plot F1 score vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, 'o-')
    plt.axvline(x=best_threshold, color='r', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score (Positive Class)')
    plt.title('Finding Optimal Threshold')
    plt.grid(True)
    
    return best_threshold

def feature_importance(model, X, preprocessing_pipeline=None, top_n=20):
    """
    Calculate and plot feature importance
    """
    # Check if the model or its base_classifier supports feature importance
    if hasattr(model, 'base_classifier') and hasattr(model.base_classifier, 'feature_importances_'):
        importances = model.base_classifier.feature_importances_
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        logger.warning("Model doesn't support feature importance explanation")
        return None

    # Try to get feature names from the preprocessing pipeline
    feature_names = None
    if preprocessing_pipeline and hasattr(preprocessing_pipeline, 'transformers_'):
        feature_names = []

        # Handle numeric features
        try:
            num_cols = preprocessing_pipeline.transformers_[0][2]
            feature_names.extend(num_cols)
        except:
            logger.warning("Could not extract numeric feature names")

        # Handle categorical features
        try:
            categorical_features = preprocessing_pipeline.named_transformers_.get('cat', None)
            if categorical_features and hasattr(categorical_features, 'named_steps'):
                cat_cols = preprocessing_pipeline.transformers_[1][2]
                onehot = categorical_features.named_steps.get('onehot')
                if hasattr(onehot, 'get_feature_names_out'):
                    cat_feature_names = onehot.get_feature_names_out(cat_cols)
                    feature_names.extend(cat_feature_names)
        except:
            logger.warning("Could not extract categorical feature names")

    # Create DataFrame and plot
    if feature_names is not None and len(feature_names) == len(importances):
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    else:
        if feature_names is not None:
            logger.warning("Feature names length doesn't match importances length. Using generic names.")
        indices = np.argsort(importances)[::-1][:top_n]
        feature_importance_df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in indices],
            'importance': importances[indices]
        })

    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(top_n)

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()

    return feature_importance_df

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"Average Precision: {avg_precision:.4f}")
    
    return {
        'classification_report': report,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision
    }

def train_and_save_model(X_train, y_train, X_test, y_test, model_path, model_type='random_forest', 
                         preprocessing_pipeline=None, class_imbalance_method='smote', optimize_threshold=True):
    """
    Train model, evaluate it, and save it to disk
    """
    # Handle class imbalance if specified
    if class_imbalance_method not in [None, 'class_weight']:
        X_train_resampled, y_train_resampled = handle_class_imbalance(
            X_train, y_train, method=class_imbalance_method
        )
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Train model
    model, best_params = train_model(
        X_train_resampled, y_train_resampled, 
        model_type=model_type,
        class_imbalance_method=class_imbalance_method
    )
    
    # Find optimal threshold if requested
    threshold = 0.5
    if optimize_threshold:
        y_prob = model.predict_proba(X_test)[:, 1]
        threshold = find_optimal_threshold(y_test, y_prob)
        
        # Recreate model with optimal threshold
        if threshold != 0.5:
            original_model = model
            model = CustomThresholdClassifier(base_classifier=original_model,threshold= threshold)
            model._fixed_proba = None  # Initialize fallback probabilities
            model._predictions = None  # Initialize fallback predictions
            logger.info(f"Applied optimal threshold of {threshold:.2f} to the model")
            
            # Create a custom threshold model
            
        logger.info(f"Applied optimal threshold of {threshold:.2f} to the model") 
    
    # Evaluate model
    evaluation_results = evaluate_model(model, X_test, y_test)
    
    # Calculate feature importance if applicable
    if hasattr(model, 'feature_importances_'):
        feature_imp = feature_importance(model, X_train, preprocessing_pipeline)
    
    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save metadata with the model
    metadata = {
        'model_type': model_type,
        'class_imbalance_method': class_imbalance_method,
        'best_params': best_params,
        'threshold': threshold,
        'evaluation_metrics': {
            'accuracy': evaluation_results['classification_report']['accuracy'],
            'positive_class_f1': evaluation_results['classification_report']['1']['f1-score'],
            'roc_auc': evaluation_results['roc_auc'],
            'avg_precision': evaluation_results['avg_precision']
        }
    }
    
    metadata_path = os.path.join(os.path.dirname(model_path), 'model_metadata.joblib')
    joblib.dump(metadata, metadata_path)
    logger.info(f"Model metadata saved to {metadata_path}")
    
    # Save evaluation results
    metrics_path = os.path.join(os.path.dirname(model_path), 'model_metrics.joblib')
    joblib.dump(evaluation_results, metrics_path)
    logger.info(f"Evaluation metrics saved to {metrics_path}")
    
    return model, evaluation_results, metadata

def main(data_path='data/processed', preprocessing_pipeline_path='data/processed/preprocessing_pipeline.joblib',
         model_path='api/model.joblib', model_type='random_forest', class_imbalance_method='smote',
         optimize_threshold=True):
    """
    Main function to train and save the model
    """
    # Load data
    X = np.load(os.path.join(data_path, 'X_processed.npy'))
    y = np.load(os.path.join(data_path, 'y.npy'))
    
    # Check class distribution
    class_distribution = np.bincount(y)
    logger.info(f"Class distribution: {class_distribution}")
    class_ratio = class_distribution[0] / class_distribution[1]
    logger.info(f"Class imbalance ratio (majority:minority): {class_ratio:.2f}:1")
    
    # Load preprocessing pipeline if exists
    preprocessing_pipeline = None
    if os.path.exists(preprocessing_pipeline_path):
        preprocessing_pipeline = joblib.load(preprocessing_pipeline_path)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train and save model
    model, evaluation_results, metadata = train_and_save_model(
        X_train, y_train, X_test, y_test, model_path, 
        model_type=model_type, 
        preprocessing_pipeline=preprocessing_pipeline,
        class_imbalance_method=class_imbalance_method,
        optimize_threshold=optimize_threshold
    )
    
    return model, evaluation_results, metadata

if __name__ == "__main__":
    main(class_imbalance_method='smote', optimize_threshold=True)