# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# import lightgbm as lgb
# from imblearn.over_sampling import SMOTE
# import joblib
#
# # Load data
# data = pd.read_csv('C:\\Users\\dd\\Desktop\\ارشد code\\python\\deeplearning\\pythonProject\\ObesityDataSet_raw_and_data_sinthetic.csv')
# print("Columns:", data.columns)
#
# # Encode target
# le = LabelEncoder()
# data['NObeyesdad'] = le.fit_transform(data['NObeyesdad'])
#
# # One-hot encode categorical features
# X = data.drop('NObeyesdad', axis=1)
# X = pd.get_dummies(X)
# y = data['NObeyesdad']
#
# # Check class distribution
# print("\nClass Distribution:")
# print(data['NObeyesdad'].value_counts(normalize=True))
#
# # Apply SMOTE to handle class imbalance
# smote = SMOTE(random_state=42)
# X, y = smote.fit_resample(X, y)
#
# # Feature scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # Feature importance analysis using Random Forest
# rf_temp = RandomForestClassifier(random_state=42)
# rf_temp.fit(X_scaled, y)
# feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_temp.feature_importances_})
# print("\nFeature Importance (Random Forest):")
# print(feature_importance.sort_values(by='Importance', ascending=False))
#
# # Remove low-importance features (threshold: 0.01)
# important_features = feature_importance[feature_importance['Importance'] >= 0.01]['Feature'].values
# X = X[important_features]
# X_scaled = scaler.fit_transform(X)
#
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Importance', y='Feature', data=feature_importance.sort_values(by='Importance', ascending=False))
# plt.title('Feature Importance')
# plt.savefig('feature_importance.png')
# plt.close()
#
# # Define models
# models = {
#     'Logistic Regression': LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42),
#     'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
#     'Random Forest': RandomForestClassifier(random_state=42),
#     'KNN': KNeighborsClassifier(),
#     'SVM': SVC(probability=True, random_state=42),
#     'XGBoost': XGBClassifier(eval_metric='mlogloss', max_depth=6, learning_rate=0.1, random_state=42),
#     'LightGBM': lgb.LGBMClassifier(random_state=42)
# }
#
# # Tune Random Forest
# rf_params = {
#     'n_estimators': [100, 200],
#     'max_depth': [10, 20, None],
#     'min_samples_split': [2, 5]
# }
# grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy', n_jobs=-1)
# grid_rf.fit(X_scaled, y)
# models['Random Forest'] = grid_rf.best_estimator_
# print("\nBest Random Forest Params:", grid_rf.best_params_)
#
# # Tune XGBoost
# xgb_params = {
#     'max_depth': [3, 6, 9],
#     'learning_rate': [0.1, 0.01],
#     'n_estimators': [100, 200]
# }
# grid_xgb = GridSearchCV(XGBClassifier(eval_metric='mlogloss', random_state=42), xgb_params, cv=5, scoring='accuracy', n_jobs=-1)
# grid_xgb.fit(X_scaled, y)
# models['XGBoost'] = grid_xgb.best_estimator_
# print("\nBest XGBoost Params:", grid_xgb.best_params_)
#
# # Tune LightGBM
# lgbm_params = {
#     'max_depth': [3, 6, 9],
#     'learning_rate': [0.1, 0.01],
#     'n_estimators': [100, 200]
# }
# grid_lgbm = GridSearchCV(lgb.LGBMClassifier(random_state=42), lgbm_params, cv=5, scoring='accuracy', n_jobs=-1)
# grid_lgbm.fit(X_scaled, y)
# models['LightGBM'] = grid_lgbm.best_estimator_
# print("\nBest LightGBM Params:", grid_lgbm.best_params_)
#
# # Tune KNN
# k_values = range(3, 21)  # Start from 3 to avoid overfitting
# cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_scaled, y, cv=5, scoring='accuracy').mean() for k in k_values]
# best_k = k_values[np.argmax(cv_scores)]
# models['KNN'] = KNeighborsClassifier(n_neighbors=best_k)
# print(f"\nBest KNN n_neighbors: {best_k}")
#
# # Add Voting Classifier
# voting_clf = VotingClassifier(
#     estimators=[
#         ('rf', models['Random Forest']),
#         ('xgb', models['XGBoost']),
#         ('lgbm', models['LightGBM']),
#         ('dt', models['Decision Tree'])
#     ],
#     voting='soft'
# )
# models['Voting Classifier'] = voting_clf
#
# # KFold setup
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
# # Store results
# results = []
# conf_matrices = {}
#
# for name, model in models.items():
#     train_acc, val_acc, val_precision, val_recall, val_f1 = [], [], [], [], []
#     fold_conf_matrices = []
#
#     for train_idx, val_idx in kfold.split(X_scaled, y):
#         X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]
#
#         model.fit(X_train, y_train)
#         y_pred_train = model.predict(X_train)
#         y_pred_val = model.predict(X_val)
#
#         train_acc.append(accuracy_score(y_train, y_pred_train))
#         val_acc.append(accuracy_score(y_val, y_pred_val))
#         val_precision.append(precision_score(y_val, y_pred_val, average='macro'))
#         val_recall.append(recall_score(y_val, y_pred_val, average='macro'))
#         val_f1.append(f1_score(y_val, y_pred_val, average='macro'))
#
#         fold_conf_matrices.append(confusion_matrix(y_val, y_pred_val))
#
#     # Classification report for top models (only for the last fold)
#     if name in ['Random Forest', 'XGBoost', 'LightGBM', 'Voting Classifier']:
#         print(f"\nClassification Report for {name} (Last Fold):")
#         print(classification_report(y_val, y_pred_val, target_names=le.classes_))
#
#     results.append({
#         'Model': name,
#         'Train Accuracy': np.mean(train_acc),
#         'Validation Accuracy': np.mean(val_acc),
#         'Validation Precision': np.mean(val_precision),
#         'Validation Recall': np.mean(val_recall),
#         'Validation F1': np.mean(val_f1)
#     })
#
#     conf_matrices[name] = np.mean(fold_conf_matrices, axis=0)
#
#     # Save model
#     joblib.dump(model, f'{name}_model.pkl')
#
# # Results
# results_df = pd.DataFrame(results)
# print("\nCross-Validation Results:")
# print(results_df)
#
# # Plotting accuracy
# plt.figure(figsize=(12, 6))
# for metric in ['Train Accuracy', 'Validation Accuracy']:
#     plt.plot(results_df['Model'], results_df[metric], marker='o', label=metric)
# plt.title('Train vs Validation Accuracy')
# plt.xlabel('Model')
# plt.ylabel('Accuracy')
# plt.xticks(rotation=45)
#
# plt.legend()
# plt.grid(True)
#
# plt.tight_layout()
# plt.savefig('train_vs_validation_accuracy.png')
# plt.close()
#
# # Plot Confusion Matrices
# best_models = ['Random Forest', 'XGBoost', 'LightGBM', 'Voting Classifier']
# for model_name in best_models:
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(conf_matrices[model_name], annot=True, fmt='.0f', cmap='Blues')
#     plt.title(f'Confusion Matrix - {model_name}')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.tight_layout()
#     plt.savefig(f'confusion_matrix_{model_name}.png')
#     plt.close()


# new version


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
warnings.filterwarnings('ignore')

# ایجاد پوشه خروجی
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# تنظیم استایل با بررسی وجود seaborn
try:
    plt.style.use('seaborn')
except OSError:
    print("Seaborn style not found, falling back to ggplot")
    plt.style.use('ggplot')

plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

# بارگذاری داده‌ها
data = pd.read_csv('C:\\Users\\dd\\Desktop\\ارشد code\\python\\deeplearning\\pythonProject\\ObesityDataSet_raw_and_data_sinthetic.csv')
print("Columns:", data.columns)

# کدگذاری هدف
le = LabelEncoder()
data['NObeyesdad'] = le.fit_transform(data['NObeyesdad'])

# کدگذاری One-Hot برای ویژگی‌های دسته‌ای
X = data.drop('NObeyesdad', axis=1)
X = pd.get_dummies(X)
y = data['NObeyesdad']

# تحلیل همبستگی
plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_correlation.png'))
plt.close()

# توزیع کلاس‌ها قبل از SMOTE
plt.figure(figsize=(8, 6))
sns.countplot(x=y, palette='viridis')
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_distribution_before_smote.png'))
plt.close()

# اعمال SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# تبدیل به آرایه‌های numpy
X = X.to_numpy()
y = y.to_numpy()

# توزیع کلاس‌ها بعد از SMOTE
plt.figure(figsize=(8, 6))
sns.countplot(x=y, palette='viridis')
plt.title('Class Distribution After SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_distribution_after_smote.png'))
plt.close()

# مقیاس‌بندی ویژگی‌ها
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تقسیم داده‌ها به آموزشی و آزمایشی
X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# تحلیل اهمیت ویژگی‌ها با Random Forest
rf_temp = RandomForestClassifier(random_state=42)
rf_temp.fit(X_train_full, y_train_full)
feature_importance = pd.DataFrame({'Feature': [f'Feature_{i}' for i in range(X.shape[1])], 'Importance': rf_temp.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# تحلیل SHAP
explainer = shap.TreeExplainer(rf_temp)
shap_values = explainer.shap_values(X_train_full[:100])  # محدود برای سرعت
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, [f'Feature_{i}' for i in range(X.shape[1])], plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_feature_importance.png'))
plt.close()

# حذف ویژگی‌های کم‌اهمیت (آستانه: 0.01)
important_features_idx = feature_importance[feature_importance['Importance'] >= 0.01].index
X = X[:, important_features_idx]
X_scaled = scaler.fit_transform(X)
X_train_full = X_train_full[:, important_features_idx]
X_test = X_test[:, important_features_idx]

# رسم اهمیت ویژگی‌ها
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.close()

# تعریف مدل‌ها
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42),
    'LightGBM': LGBMClassifier(random_state=42)
}

# تنظیم Random Forest با RandomizedSearchCV
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_params, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
grid_rf.fit(X_train_full, y_train_full)
models['Random Forest'] = grid_rf.best_estimator_
print("\nBest Random Forest Params:", grid_rf.best_params_)

# تنظیم XGBoost بدون early stopping
xgb_params = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300]
}
grid_xgb = RandomizedSearchCV(XGBClassifier(eval_metric='mlogloss', random_state=42), xgb_params, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
grid_xgb.fit(X_train_full, y_train_full)
models['XGBoost'] = grid_xgb.best_estimator_
print("\nBest XGBoost Params:", grid_xgb.best_params_)

# تنظیم LightGBM با RandomizedSearchCV
lgbm_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'num_leaves': [31, 50, 70]
}
grid_lgbm = RandomizedSearchCV(LGBMClassifier(random_state=42), lgbm_params, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
grid_lgbm.fit(X_train_full, y_train_full)
models['LightGBM'] = grid_lgbm.best_estimator_
print("\nBest LightGBM Params:", grid_lgbm.best_params_)

# تنظیم KNN
k_values = range(3, 21)
cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train_full, y_train_full, cv=5, scoring='accuracy').mean() for k in k_values]
best_k = k_values[np.argmax(cv_scores)]
models['KNN'] = KNeighborsClassifier(n_neighbors=best_k)
print(f"\nBest KNN n_neighbors: {best_k}")

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', models['Random Forest']),
        ('xgb', models['XGBoost']),
        ('lgbm', models['LightGBM']),
        ('dt', models['Decision Tree'])
    ],
    voting='soft'
)
models['Voting Classifier'] = voting_clf

# تنظیم KFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ذخیره نتایج
results = []
conf_matrices = {}
roc_auc_scores = {}
roc_curves = {}

for name, model in models.items():
    train_acc, val_acc, val_precision, val_recall, val_f1, val_roc_auc = [], [], [], [], [], []
    fold_conf_matrices = []

    for train_idx, val_idx in kfold.split(X_train_full, y_train_full):
        X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

        # آموزش مدل
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_prob_val = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else np.zeros_like(y_val)

        train_acc.append(accuracy_score(y_train, y_pred_train))
        val_acc.append(accuracy_score(y_val, y_pred_val))
        val_precision.append(precision_score(y_val, y_pred_val, average='macro'))
        val_recall.append(recall_score(y_val, y_pred_val, average='macro'))
        val_f1.append(f1_score(y_val, y_pred_val, average='macro'))
        val_roc_auc.append(roc_auc_score(y_val, y_prob_val, multi_class='ovr') if y_prob_val.any() else 0)

        fold_conf_matrices.append(confusion_matrix(y_val, y_pred_val))

    # گزارش طبقه‌بندی برای مدل‌های برتر
    if name in ['Random Forest', 'XGBoost', 'LightGBM', 'Voting Classifier']:
        print(f"\nClassification Report for {name} (Last Fold):")
        print(classification_report(y_val, y_pred_val, target_names=le.classes_))

    results.append({
        'Model': name,
        'Train Accuracy': np.mean(train_acc),
        'Validation Accuracy': np.mean(val_acc),
        'Validation Precision': np.mean(val_precision),
        'Validation Recall': np.mean(val_recall),
        'Validation F1': np.mean(val_f1),
        'Validation ROC-AUC': np.mean(val_roc_auc)
    })

    conf_matrices[name] = np.mean(fold_conf_matrices, axis=0)
    roc_auc_scores[name] = np.mean(val_roc_auc)

    # منحنی ROC
    if hasattr(model, 'predict_proba'):
        model.fit(X_train_full, y_train_full)
        y_prob = model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(pd.get_dummies(y_test).values.ravel(), y_prob.ravel())
        roc_curves[name] = (fpr, tpr, auc(fpr, tpr))

    # ذخیره مدل
    joblib.dump(model, os.path.join(output_dir, f'{name}_model.pkl'))

# مقایسه آماری (t-test)
rf_acc = cross_val_score(models['Random Forest'], X_train_full, y_train_full, cv=5, scoring='accuracy')
xgb_acc = cross_val_score(models['XGBoost'], X_train_full, y_train_full, cv=5, scoring='accuracy')
lgbm_acc = cross_val_score(models['LightGBM'], X_train_full, y_train_full, cv=5, scoring='accuracy')
t_stat_rf_xgb, p_value_rf_xgb = ttest_rel(rf_acc, xgb_acc)
t_stat_rf_lgbm, p_value_rf_lgbm = ttest_rel(rf_acc, lgbm_acc)
print(f"\nT-test between Random Forest and XGBoost: t-statistic={t_stat_rf_xgb:.3f}, p-value={p_value_rf_xgb:.3f}")
print(f"T-test between Random Forest and LightGBM: t-statistic={t_stat_rf_lgbm:.3f}, p-value={p_value_rf_lgbm:.3f}")

# نتایج
results_df = pd.DataFrame(results)
print("\nCross-Validation Results:")
print(results_df)

# ذخیره نتایج به‌صورت CSV و LaTeX
results_df.to_csv(os.path.join(output_dir, 'model_results.csv'), index=False)
try:
    results_df.to_latex(os.path.join(output_dir, 'model_results.tex'), index=False, float_format="%.3f")
    print("LaTeX file 'model_results.tex' generated successfully.")
except ImportError:
    print("Warning: Jinja2 is not installed. LaTeX file was not generated. Please install Jinja2 with 'pip install jinja2'.")

# رسم دقت
plt.figure(figsize=(12, 6))
for metric in ['Train Accuracy', 'Validation Accuracy']:
    plt.plot(results_df['Model'], results_df[metric], marker='o', label=metric)
plt.title('Train vs Validation Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'train_vs_validation_accuracy.png'))
plt.close()

# رسم منحنی‌های ROC
plt.figure(figsize=(10, 8))
for name, (fpr, tpr, roc_auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves for Test Set')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
plt.close()

# رسم ماتریس‌های درهم‌ریختگی
best_models = ['Random Forest', 'XGBoost', 'LightGBM', 'Voting Classifier']
for model_name in best_models:
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrices[model_name], annot=True, fmt='.0f', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()

# تولید گزارش
report = f"""
# Obesity Classification Analysis Report

## Dataset Overview
- Number of samples: {len(data)}
- Number of features: {X.shape[1]}
- Classes: {le.classes_}

## Methodology
- **Preprocessing**: One-Hot Encoding, SMOTE for class imbalance, StandardScaler for scaling.
- **Feature Selection**: Random Forest and SHAP for importance analysis.
- **Models**: {', '.join(models.keys())}.
- **Evaluation**: 5-Fold Cross-Validation, metrics include Accuracy, Precision, Recall, F1-Score, ROC-AUC.

## Results
"""
try:
    report += results_df.to_markdown(index=False)
except ImportError:
    report += "Results table could not be generated due to missing 'tabulate' package. Please install tabulate with 'pip install tabulate'.\n"
    report += str(results_df)

report += f"""

## Statistical Analysis
- T-test between Random Forest and XGBoost: t-statistic={t_stat_rf_xgb:.3f}, p-value={p_value_rf_xgb:.3f}
- T-test between Random Forest and LightGBM: t-statistic={t_stat_rf_lgbm:.3f}, p-value={p_value_rf_lgbm:.3f}

## Visualizations
- Feature importance and SHAP analysis saved in 'outputs/feature_importance.png' and 'outputs/shap_feature_importance.png'.
- ROC Curves saved in 'outputs/roc_curves.png'.
- Confusion matrices for top models saved in 'outputs/confusion_matrix_<model>.png'.

## Conclusion
The Random Forest, XGBoost, and LightGBM models performed best, with the Voting Classifier providing robust results. Further analysis with larger datasets or additional features could enhance performance.
"""

with open(os.path.join(output_dir, 'analysis_report.md'), 'w') as f:
    f.write(report)