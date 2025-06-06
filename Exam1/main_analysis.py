import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
import shap
from scipy.stats import ttest_rel
import os
import warnings

warnings.filterwarnings('ignore') # Suppress warnings globally

# --- Global Constants ---
DATA_PATH = 'Exam1/data/ObesityDataSet_raw_and_data_sinthetic.csv'
OUTPUT_DIR = 'outputs' # Relative to script location (Exam1/outputs)
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.2
CV_FOLDS = 5
FEATURE_IMPORTANCE_THRESHOLD = 0.01
SHAP_SAMPLE_SIZE = 200 # For SHAP analysis to manage computation time
# RandomizedSearchCV iterations
N_ITER_RF = 8
N_ITER_XGB = 4 # Reduced iterations for faster demo; can be increased
N_ITER_LGBM = 4 # Reduced iterations for faster demo; can be increased
KNN_K_RANGE = range(3, 16) # Range of k values for KNN tuning

# --- Function Definitions ---

def load_and_preprocess_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Loads data from CSV, performs Label Encoding for the target variable,
    and One-Hot Encoding for features.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        tuple[pd.DataFrame, pd.Series, LabelEncoder]:
            X_df (DataFrame of features),
            y_series (Series for target),
            le (fitted LabelEncoder for target).
    """
    data = pd.read_csv(csv_path)
    print("Original columns:", data.columns.tolist())

    le = LabelEncoder()
    data['NObeyesdad'] = le.fit_transform(data['NObeyesdad'])

    X_df = pd.get_dummies(data.drop('NObeyesdad', axis=1), prefix_sep='_')
    y_series = data['NObeyesdad']

    print(f"Data shape after OHE: X_df: {X_df.shape}, y_series: {y_series.shape}")
    return X_df, y_series, le

def plot_correlation_heatmap(df: pd.DataFrame, output_path: str) -> None:
    """
    Plots and saves a correlation heatmap for the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame for which to plot correlation.
        output_path (str): Path to save the heatmap image.
    """
    plt.figure(figsize=(20, 18))
    sns.heatmap(df.corr(), cmap='coolwarm', vmin=-1, vmax=1, annot=False) # Annot=False for large matrices
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig(output_path)
    plt.close() # Close the figure to free memory
    print(f"Correlation heatmap saved to {output_path}")

def plot_class_distribution(y: pd.Series, title: str, output_path: str) -> None:
    """
    Plots and saves the class distribution for the target variable.

    Args:
        y (pd.Series): Target variable Series.
        title (str): Title for the plot.
        output_path (str): Path to save the distribution plot.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y, palette='viridis')
    plt.title(title)
    plt.xlabel('Class Label') # More descriptive label
    plt.ylabel('Frequency')  # More descriptive label
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Class distribution plot '{title}' saved to {output_path}")

def apply_smote(X_df: pd.DataFrame, y_series: pd.Series, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Applies SMOTE to handle class imbalance in the dataset.

    Args:
        X_df (pd.DataFrame): DataFrame of features.
        y_series (pd.Series): Series for target variable.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Resampled X_df and y_series.
    """
    print(f"Shape before SMOTE: X: {X_df.shape}, y: {y_series.shape}")
    smote = SMOTE(random_state=random_state)
    X_resampled_df, y_resampled_series = smote.fit_resample(X_df, y_series)
    print(f"Shape after SMOTE: X: {X_resampled_df.shape}, y: {y_resampled_series.shape}")
    return X_resampled_df, y_resampled_series

# --- Main script flow starts here, to be wrapped in main() later ---

# (Original code for converting to NumPy arrays, plotting post-SMOTE distribution,
# splitting, scaling, feature importance, model training etc. will be moved incrementally)

# Convert to NumPy arrays (Example, will be part of a function later)
# X_smote_np = X_smote_df.to_numpy()
# y_smote_np = y_smote_series.to_numpy() # This was already present from previous refactoring step
# Store feature names from after SMOTE (if columns changed, though SMOTE usually doesn't change feature names/count)
# In this case, X_smote_df columns are same as original_feature_names
# smote_feature_names = X_smote_df.columns.tolist() # This was already present

def split_and_scale_data(X_np: np.ndarray, y_np: np.ndarray, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Splits data into training and testing sets, then scales the features.
    The scaler is fit only on the training data.

    Args:
        X_np (np.ndarray): NumPy array of features.
        y_np (np.ndarray): NumPy array of target variable.
        test_size (float): Proportion of the dataset to allocate to the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
            Scaled training features, scaled test features, training target, test target, fitted scaler.
    """
    X_train_all_feats_np, X_test_all_feats_np, y_train_np, y_test_np = train_test_split(
        X_np, y_np, test_size=test_size, stratify=y_np, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled_all_feats_np = scaler.fit_transform(X_train_all_feats_np)
    X_test_scaled_all_feats_np = scaler.transform(X_test_all_feats_np)
    print("Data split and scaled. Scaler fit on training data.")
    return X_train_scaled_all_feats_np, X_test_scaled_all_feats_np, y_train_np, y_test_np, scaler

def get_feature_importance_and_shap(
    X_train_scaled_all_feats_np: np.ndarray,
    y_train_np: np.ndarray,
    all_feature_names: list,
    output_dir_local: str,
    random_state_local: int,
    importance_threshold: float = FEATURE_IMPORTANCE_THRESHOLD,
    shap_sample_size: int = SHAP_SAMPLE_SIZE
) -> tuple[np.ndarray, list]:
    """
    Calculates Random Forest feature importance, performs SHAP analysis, saves plots,
    and returns original indices and names of selected features based on the threshold.

    Args:
        X_train_scaled_all_feats_np (np.ndarray): Scaled training features (all features).
        y_train_np (np.ndarray): Training target variable.
        all_feature_names (list): List of names for all features, order matches columns in X_train_scaled_all_feats_np.
        output_dir_local (str): Directory to save plots.
        random_state_local (int): Random state for reproducibility.
        importance_threshold (float): Threshold for selecting important features.
        shap_sample_size (int): Sample size for SHAP analysis.

    Returns:
        tuple[np.ndarray, list]: NumPy array of original column indices for selected features,
                                 list of names for selected features.
    """
    # Train a temporary Random Forest to get feature importances
    rf_temp = RandomForestClassifier(random_state=random_state_local)
    rf_temp.fit(X_train_scaled_all_feats_np, y_train_np)

    importances = rf_temp.feature_importances_
    # Create DataFrame with features and their importances, using original feature names
    # The index of this DataFrame (0 to N-1) corresponds to the original column indices
    feature_importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot top N feature importances (all features)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
    plt.title('Top 20 Feature Importances (All Features, Pre-selection)')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir_local, 'feature_importance_all_features.png')); plt.close()

    # SHAP analysis on a sample of the data for efficiency
    X_train_shap_sample_np = X_train_scaled_all_feats_np[:shap_sample_size]
    explainer = shap.TreeExplainer(rf_temp)
    shap_values = explainer.shap_values(X_train_shap_sample_np)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_train_shap_sample_np, feature_names=all_feature_names, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance (Sample of {shap_sample_size} instances, All Features)')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir_local, 'shap_feature_importance_all_features.png')); plt.close()

    # Select features based on importance threshold
    important_features_mask = feature_importance_df['Importance'] >= importance_threshold
    selected_feature_names = feature_importance_df.loc[important_features_mask, 'Feature'].tolist()

    # Get the original indices of these selected features from the *unsorted* feature_importance_df's index
    # which corresponds to the original column indices
    original_indices_of_selected_features = feature_importance_df[important_features_mask].index.to_numpy()

    print(f"Number of features before selection: {len(all_feature_names)}")
    print(f"Number of features after selection (threshold {importance_threshold}): {len(selected_feature_names)}")

    # Plot importances of only the selected features
    if selected_feature_names:
        selected_features_plotting_df = feature_importance_df[important_features_mask]
        fig_height_selected = max(6, len(selected_feature_names) * 0.4)
        plt.figure(figsize=(10, fig_height_selected))
        sns.barplot(x='Importance', y='Feature', data=selected_features_plotting_df)
        plt.title(f'Selected Feature Importances (Importance >= {importance_threshold})')
        plt.tight_layout(); plt.savefig(os.path.join(output_dir_local, 'feature_importance_selected_features.png')); plt.close()
    else:
        print("No features met the importance threshold for selection.")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No features selected based on importance threshold.', ha='center', va='center')
        plt.title('Selected Feature Importances'); plt.xlabel('Importance'); plt.ylabel('Feature')
        plt.tight_layout(); plt.savefig(os.path.join(output_dir_local, 'feature_importance_selected_features.png')); plt.close()

    print("Feature importance and SHAP analysis complete.")
    # Return the original column indices for slicing the numpy array
    return original_indices_of_selected_features, selected_feature_names


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        print("Seaborn style 'seaborn-v0_8-darkgrid' not found, falling back to ggplot.")
        plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 120})

    X_df, y_series, le = load_and_preprocess_data(DATA_PATH) # Use constant
    original_feature_names = X_df.columns.tolist()

    plot_correlation_heatmap(X_df, os.path.join(OUTPUT_DIR, 'feature_correlation.png'))
    plot_class_distribution(y_series, 'Class Distribution Before SMOTE', os.path.join(OUTPUT_DIR, 'class_distribution_before_smote.png'))

    X_smote_df, y_smote_series = apply_smote(X_df, y_series, RANDOM_STATE)

    plot_class_distribution(y_smote_series, 'Class Distribution After SMOTE', os.path.join(OUTPUT_DIR, 'class_distribution_after_smote.png'))

    X_smote_np = X_smote_df.to_numpy()
    y_smote_np = y_smote_series.to_numpy()
    smote_feature_names = X_smote_df.columns.tolist() # These are the names for the columns in X_smote_np

    X_train_all_feats_np, X_test_all_feats_np, y_train_np, y_test_np, _ = split_and_scale_data(
        X_smote_np, y_smote_np, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE # Use constants
    )
    # scaler object is not used further in main, so assigned to _

    important_indices, selected_feature_names = get_feature_importance_and_shap(
        X_train_all_feats_np, y_train_np, smote_feature_names, OUTPUT_DIR, RANDOM_STATE,
        importance_threshold=FEATURE_IMPORTANCE_THRESHOLD, shap_sample_size=SHAP_SAMPLE_SIZE # Pass constants
    )

    X_train_full = X_train_all_feats_np[:, important_indices]
    X_test = X_test_all_feats_np[:, important_indices]
    y_train_full = y_train_np # Renaming for consistency with the rest of the original script
    y_test = y_test_np     # Renaming for consistency

    # Store counts for the report
    num_original_features = X_train_all_feats_np.shape[1]
    num_selected_features = X_train_full.shape[1]

    # (The rest of the script: model definitions, tuning, evaluation, report generation will be added here)
    print(f"Number of original features: {num_original_features}")
    print(f"Number of selected features: {num_selected_features}")
    print("Preprocessing, scaling, and feature selection complete.")
    # Keep a reference to le for use in main() scope, to pass to other functions
    main_le = le

    # Define and tune models
    tuned_models = define_and_tune_models(X_train_full, y_train_full, RANDOM_STATE)

    # Train and evaluate final models
    results_df, conf_matrices_test, roc_data_test = train_and_evaluate_final_models(
        tuned_models, X_train_full, y_train_full, X_test, y_test, main_le, OUTPUT_DIR, RANDOM_STATE
    )

    # Perform statistical tests (using selected scaled training data)
    if not results_df.empty:
        t_test_results = perform_statistical_tests(tuned_models, X_train_full, y_train_full, RANDOM_STATE)

        # Save comprehensive results and plots
        save_final_results_and_plots(results_df, conf_matrices_test, roc_data_test, main_le, OUTPUT_DIR)

        # Generate summary report
        generate_summary_report(
            results_df, num_original_features, num_selected_features, main_le.classes_,
            list(tuned_models.keys()), t_test_results, OUTPUT_DIR
        )
    else:
        print("Warning: Results DataFrame is empty, skipping statistical tests, saving plots, and report generation.")

    print(f"Analysis complete. Outputs are in '{OUTPUT_DIR}'.")


def define_and_tune_models(X_train_np, y_train_np, random_state_local):
    """Defines base models and tunes hyperparameters for RF, XGBoost, LightGBM, KNN."""
    base_models = {
        'Logistic Regression': LogisticRegression(max_iter=5000, class_weight='balanced', random_state=random_state_local),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=random_state_local),
        'Random Forest': RandomForestClassifier(random_state=random_state_local),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True, random_state=random_state_local),
        'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=random_state_local, use_label_encoder=False),
        'LightGBM': LGBMClassifier(random_state=random_state_local)
    }
    tuned_models = base_models.copy()
    cv_folds = 5

    rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
    grid_rf = RandomizedSearchCV(RandomForestClassifier(random_state=random_state_local), rf_params, n_iter=8, cv=cv_folds, scoring='accuracy', n_jobs=-1, random_state=random_state_local)
    grid_rf.fit(X_train_np, y_train_np)
    tuned_models['Random Forest'] = grid_rf.best_estimator_
    print(f"Best Random Forest Params: {grid_rf.best_params_}")

    xgb_params = {'max_depth': [3, 6], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
    grid_xgb = RandomizedSearchCV(XGBClassifier(eval_metric='mlogloss', random_state=random_state_local, use_label_encoder=False), xgb_params, n_iter=4, cv=cv_folds, scoring='accuracy', n_jobs=-1, random_state=random_state_local)
    grid_xgb.fit(X_train_np, y_train_np)
    tuned_models['XGBoost'] = grid_xgb.best_estimator_
    print(f"Best XGBoost Params: {grid_xgb.best_params_}")

    lgbm_params = {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1], 'num_leaves': [31, 50]}
    grid_lgbm = RandomizedSearchCV(LGBMClassifier(random_state=random_state_local), lgbm_params, n_iter=4, cv=cv_folds, scoring='accuracy', n_jobs=-1, random_state=random_state_local)
    grid_lgbm.fit(X_train_np, y_train_np)
    tuned_models['LightGBM'] = grid_lgbm.best_estimator_
    print(f"Best LightGBM Params: {grid_lgbm.best_params_}")

    k_values = range(3, 16)
    cv_scores_knn = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train_np, y_train_np, cv=cv_folds, scoring='accuracy').mean() for k in k_values]
    best_k = k_values[np.argmax(cv_scores_knn)]
    tuned_models['KNN'] = KNeighborsClassifier(n_neighbors=best_k)
    print(f"Best KNN n_neighbors: {best_k}")

    tuned_models['Voting Classifier'] = VotingClassifier(
        estimators=[('rf', tuned_models['Random Forest']), ('xgb', tuned_models['XGBoost']),
                    ('lgbm', tuned_models['LightGBM']), ('dt', tuned_models['Decision Tree'])],
        voting='soft'
    )
    print("Hyperparameter tuning complete.")
    return tuned_models

def train_and_evaluate_final_models(models_dict, X_train_full_np, y_train_full_np, X_test_final_np, y_test_final_np, le_obj, output_dir_local, random_state_local):
    """Trains final models, evaluates on CV and test set, saves models."""
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_local)
    results_list = []
    conf_matrices_test_set = {} # Store confusion matrices from test set
    roc_curves_data_test_set = {} # Store ROC data from test set

    for name, model in models_dict.items():
        train_acc_cv, val_acc_cv, val_prec_cv, val_recall_cv, val_f1_cv, val_roc_auc_cv = [], [], [], [], [], []

        for train_idx, val_idx in kfold.split(X_train_full_np, y_train_full_np):
            X_cv_train, X_cv_val = X_train_full_np[train_idx], X_train_full_np[val_idx]
            y_cv_train, y_cv_val = y_train_full_np[train_idx], y_train_full_np[val_idx]

            model.fit(X_cv_train, y_cv_train)
            y_pred_train_cv = model.predict(X_cv_train)
            y_pred_val_cv = model.predict(X_cv_val)

            train_acc_cv.append(accuracy_score(y_cv_train, y_pred_train_cv))
            val_acc_cv.append(accuracy_score(y_cv_val, y_pred_val_cv))
            val_prec_cv.append(precision_score(y_cv_val, y_pred_val_cv, average='macro', zero_division=0))
            val_recall_cv.append(recall_score(y_cv_val, y_pred_val_cv, average='macro', zero_division=0))
            val_f1_cv.append(f1_score(y_cv_val, y_pred_val_cv, average='macro', zero_division=0))
            if hasattr(model, 'predict_proba'):
                y_prob_val_cv = model.predict_proba(X_cv_val)
                val_roc_auc_cv.append(roc_auc_score(y_cv_val, y_prob_val_cv, multi_class='ovr'))
            else:
                val_roc_auc_cv.append(0.0) # Default for models without predict_proba

        # Fit final model on the entire training dataset (X_train_full_np)
        model.fit(X_train_full_np, y_train_full_np)

        # Generate classification report on the full training data for reference
        if name in ['Random Forest', 'XGBoost', 'LightGBM', 'Voting Classifier']:
            y_pred_on_full_train = model.predict(X_train_full_np) # Predict on the data it was just trained on
            print(f"\nClassification Report for {name} (on full selected training data):\n{classification_report(y_train_full_np, y_pred_on_full_train, target_names=le_obj.classes_)}")

        results_list.append({
            'Model': name, 'Train Accuracy': np.mean(train_acc_cv), 'Validation Accuracy': np.mean(val_acc_cv),
            'Validation Precision': np.mean(val_prec_cv), 'Validation Recall': np.mean(val_recall_cv),
            'Validation F1': np.mean(val_f1_cv), 'Validation ROC-AUC (CV)': np.mean(val_roc_auc_cv)
        })

        # Evaluate on the actual Test Set
        y_pred_test = model.predict(X_test_final_np)
        conf_matrices_test_set[name] = confusion_matrix(y_test_final_np, y_pred_test)

        if hasattr(model, 'predict_proba'):
            y_prob_test = model.predict_proba(X_test_final_np)
            # Micro-average ROC for multi-class
            fpr, tpr, _ = roc_curve(pd.get_dummies(y_test_final_np).values.ravel(), y_prob_test.ravel())
            roc_curves_data_test_set[name] = (fpr, tpr, auc(fpr, tpr))

        joblib.dump(model, os.path.join(output_dir_local, f'{name}_model.pkl'))
    print("Model training and evaluation complete.")
    return pd.DataFrame(results_list), conf_matrices_test_set, roc_curves_data_test_set

def perform_statistical_tests(models_to_test_dict, X_train_data_np, y_train_data_np, random_state_local):
    """Performs t-tests between specified model accuracies using cross_val_score."""
    cv_folds = 5
    results = {}

    # Ensure keys exist before trying to access them
    if 'Random Forest' in models_to_test_dict and 'XGBoost' in models_to_test_dict:
        rf_acc = cross_val_score(models_to_test_dict['Random Forest'], X_train_data_np, y_train_data_np, cv=cv_folds, scoring='accuracy')
        xgb_acc = cross_val_score(models_to_test_dict['XGBoost'], X_train_data_np, y_train_data_np, cv=cv_folds, scoring='accuracy')
        t_stat_rf_xgb, p_value_rf_xgb = ttest_rel(rf_acc, xgb_acc)
        results["rf_vs_xgb"] = (t_stat_rf_xgb, p_value_rf_xgb)
        print(f"\nT-test Random Forest vs XGBoost: t-stat={t_stat_rf_xgb:.3f}, p-value={p_value_rf_xgb:.3f}")

    if 'Random Forest' in models_to_test_dict and 'LightGBM' in models_to_test_dict:
        rf_acc = cross_val_score(models_to_test_dict['Random Forest'], X_train_data_np, y_train_data_np, cv=cv_folds, scoring='accuracy') # Re-calculate if not already done
        lgbm_acc = cross_val_score(models_to_test_dict['LightGBM'], X_train_data_np, y_train_data_np, cv=cv_folds, scoring='accuracy')
        t_stat_rf_lgbm, p_value_rf_lgbm = ttest_rel(rf_acc, lgbm_acc)
        results["rf_vs_lgbm"] = (t_stat_rf_lgbm, p_value_rf_lgbm)
        print(f"T-test Random Forest vs LightGBM: t-stat={t_stat_rf_lgbm:.3f}, p-value={p_value_rf_lgbm:.3f}")

    if not results:
        print("Statistical tests could not be performed (e.g. required models not found).")
    return results

def save_final_results_and_plots(results_dataframe, conf_matrix_dict, roc_data_dict, le_obj, output_dir_local):
    """Saves final results to CSV/LaTeX and generates various plots based on Test Set performance."""
    results_dataframe.to_csv(os.path.join(output_dir_local, 'model_evaluation_results.csv'), index=False)
    try:
        results_dataframe.to_latex(os.path.join(output_dir_local, 'model_evaluation_results.tex'), index=False, float_format="%.3f")
    except ImportError:
        print("Warning: Jinja2 or other LaTeX dependencies not installed. Skipping .tex report for results.")

    # Plot aggregated CV accuracy (from results_dataframe which contains CV means)
    plt.figure(figsize=(12, 6))
    for metric_col in ['Train Accuracy', 'Validation Accuracy', 'Validation ROC-AUC (CV)']:
        if metric_col in results_dataframe.columns:
             plt.plot(results_dataframe['Model'], results_dataframe[metric_col], marker='o', label=metric_col)
    plt.title('Model Performance Metrics (Cross-Validation Averages)')
    plt.xlabel('Model'); plt.ylabel('Score'); plt.xticks(rotation=45, ha='right')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir_local, 'model_performance_cv_summary.png')); plt.close()

    # Plot ROC curves from Test Set
    plt.figure(figsize=(10, 8))
    for model_name, (fpr, tpr, roc_auc_value) in roc_data_dict.items():
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_value:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title('ROC Curves (Test Set)'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right"); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir_local, 'roc_curves_test_set.png')); plt.close()

    # Plot Confusion Matrices from Test Set
    for model_name, cm_array in conf_matrix_dict.items():
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm_array, annot=True, fmt='.0f', cmap='Blues',
                    xticklabels=le_obj.classes_, yticklabels=le_obj.classes_)
        plt.title(f'Confusion Matrix - {model_name} (Test Set)'); plt.xlabel('Predicted'); plt.ylabel('True')
        plt.tight_layout(); plt.savefig(os.path.join(output_dir_local, f'confusion_matrix_test_set_{model_name}.png')); plt.close()
    print("Final results and plots saved.")

def generate_summary_report(results_dataframe, num_original_feats, num_selected_feats, le_classes_list, model_keys_list, t_test_results_dict, output_dir_local):
    """Generates a markdown summary report of the analysis."""
    report_path = os.path.join(output_dir_local, 'analysis_summary_report.md')

    stat_summary_text = ""
    if t_test_results_dict:
        for test_name, (t_stat, p_val) in t_test_results_dict.items():
            stat_summary_text += f"- {test_name.replace('_', ' ').title()}: t-statistic={t_stat:.3f}, p-value={p_val:.3f}\n"
    else:
        stat_summary_text = "No t-tests performed or results unavailable.\n"

    try:
        results_md = results_dataframe.to_markdown(index=False)
    except ImportError:
        results_md = "Results table could not be generated (tabulate package missing).\n" + str(results_dataframe)
        print("Warning: 'tabulate' package not found for markdown report generation.")

    report_content = f"""# Obesity Classification Analysis Report

## Dataset Overview
- Number of original features (after OHE): {num_original_feats}
- Number of features after selection (importance >= 0.01): {num_selected_feats}
- Target classes: {list(le_classes_list)}

## Methodology
- **Preprocessing**: Data loaded, categorical features One-Hot Encoded, target Label Encoded. SMOTE applied for class imbalance. Features then scaled using StandardScaler (fit on training data only).
- **Feature Selection**: Based on Random Forest feature importance (from all features on training data), followed by SHAP analysis for insights.
- **Models Trained**: {', '.join(model_keys_list)}.
- **Hyperparameter Tuning**: RandomizedSearchCV used for Random Forest, XGBoost, LightGBM. KNN tuned via grid search over k.
- **Evaluation**: Models evaluated using 5-Fold Stratified Cross-Validation on the training set. Final performance metrics (including ROC AUC and confusion matrices) are reported on the held-out test set.

## Model Performance Summary (Mean CV Scores & Test Set Metrics)
{results_md}
*Note: 'Train Accuracy' and 'Validation Accuracy/ROC-AUC (CV)' are averages from cross-validation on the training dataset. Other plots (ROC, Confusion Matrix) reflect performance on the final test set.*

## Statistical Analysis (t-tests on CV accuracies from training data)
{stat_summary_text}

## Visualizations
All generated plots are saved in the '{output_dir_local}' directory. Key plots include:
- `feature_correlation.png`: Heatmap of feature correlations.
- `class_distribution_before_smote.png` & `class_distribution_after_smote.png`: Class balance.
- `feature_importance_all_features.png`: Importance scores for all features from RF.
- `shap_feature_importance_all_features.png`: SHAP summary plot.
- `feature_importance_selected_features.png`: Importance scores for selected features.
- `model_performance_cv_summary.png`: Comparison of models based on CV metrics.
- `roc_curves_test_set.png`: ROC curves for models on the test set.
- `confusion_matrix_test_set_<model_name>.png`: Confusion matrices for models on the test set.

## Conclusion
The analysis pipeline successfully processed the data, selected relevant features, tuned multiple classification models, and evaluated their performance. The best performing model can be chosen based on the detailed results and plots generated. Models and results are saved for further inspection.
"""
    with open(report_path, 'w') as f:
        f.write(report_content)
    print(f"Summary report generated: {report_path}")

# (The existing models dictionary and other code below this line will be removed by the diff)
# تعریف مدل‌ها # This part is still outside main(), to be moved next.
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42),