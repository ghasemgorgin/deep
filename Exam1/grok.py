import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load data
data = pd.read_csv('C:\\Users\\dd\\Desktop\\ارشد code\\python\\deeplearning\\pythonProject\\ObesityDataSet_raw_and_data_sinthetic.csv')
print("Columns:", data.columns)

# Encode target
le = LabelEncoder()
data['NObeyesdad'] = le.fit_transform(data['NObeyesdad'])

# One-hot encode categorical features
X = data.drop('NObeyesdad', axis=1)
X = pd.get_dummies(X)
y = data['NObeyesdad']

# Check class distribution
print("\nClass Distribution:")
print(data['NObeyesdad'].value_counts(normalize=True))

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature importance analysis using Random Forest
rf_temp = RandomForestClassifier(random_state=42)
rf_temp.fit(X_scaled, y)
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_temp.feature_importances_})
print("\nFeature Importance (Random Forest):")
print(feature_importance.sort_values(by='Importance', ascending=False))

# Remove low-importance features (threshold: 0.01)
important_features = feature_importance[feature_importance['Importance'] >= 0.01]['Feature'].values
X = X[important_features]
X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.sort_values(by='Importance', ascending=False))
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
plt.close()

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='mlogloss', max_depth=6, learning_rate=0.1, random_state=42)
}

# Tune Random Forest
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_scaled, y)
models['Random Forest'] = grid_rf.best_estimator_
print("\nBest Random Forest Params:", grid_rf.best_params_)

# Tune XGBoost
xgb_params = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.1, 0.01],
    'n_estimators': [100, 200]
}
grid_xgb = GridSearchCV(XGBClassifier(eval_metric='mlogloss', random_state=42), xgb_params, cv=5, scoring='accuracy', n_jobs=-1)
grid_xgb.fit(X_scaled, y)
models['XGBoost'] = grid_xgb.best_estimator_
print("\nBest XGBoost Params:", grid_xgb.best_params_)

# Tune KNN
k_values = range(3, 21)  # Start from 3 to avoid overfitting
cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_scaled, y, cv=5, scoring='accuracy').mean() for k in k_values]
best_k = k_values[np.argmax(cv_scores)]
models['KNN'] = KNeighborsClassifier(n_neighbors=best_k)
print(f"\nBest KNN n_neighbors: {best_k}")

# Add Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', models['Random Forest']),
        ('xgb', models['XGBoost']),
        ('dt', models['Decision Tree'])
    ],
    voting='soft'
)
models['Voting Classifier'] = voting_clf

# KFold setup
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store results
results = []
conf_matrices = {}

for name, model in models.items():
    train_acc, val_acc, val_precision, val_recall, val_f1 = [], [], [], [], []
    fold_conf_matrices = []

    for train_idx, val_idx in kfold.split(X_scaled, y):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        train_acc.append(accuracy_score(y_train, y_pred_train))
        val_acc.append(accuracy_score(y_val, y_pred_val))
        val_precision.append(precision_score(y_val, y_pred_val, average='macro'))
        val_recall.append(recall_score(y_val, y_pred_val, average='macro'))
        val_f1.append(f1_score(y_val, y_pred_val, average='macro'))

        fold_conf_matrices.append(confusion_matrix(y_val, y_pred_val))

    # Classification report for top models (only for the last fold)
    if name in ['Random Forest', 'XGBoost', 'Voting Classifier']:
        print(f"\nClassification Report for {name} (Last Fold):")
        print(classification_report(y_val, y_pred_val, target_names=le.classes_))

    results.append({
        'Model': name,
        'Train Accuracy': np.mean(train_acc),
        'Validation Accuracy': np.mean(val_acc),
        'Validation Precision': np.mean(val_precision),
        'Validation Recall': np.mean(val_recall),
        'Validation F1': np.mean(val_f1)
    })

    conf_matrices[name] = np.mean(fold_conf_matrices, axis=0)

    # Save model
    joblib.dump(model, f'{name}_model.pkl')

# Results
results_df = pd.DataFrame(results)
print("\nCross-Validation Results:")
print(results_df)

# Plotting accuracy
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
plt.savefig('train_vs_validation_accuracy.png')
plt.close()

# Plot Confusion Matrices
best_models = ['Random Forest', 'XGBoost', 'Voting Classifier']
for model_name in best_models:
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrices[model_name], annot=True, fmt='.0f', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()