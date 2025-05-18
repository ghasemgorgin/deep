import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Load data
data = pd.read_csv('C:\\Users\\dd\\Desktop\\ارشد code\\python\\deeplearning\\pythonProject\\ObesityDataSet_raw_and_data_sinthetic.csv')

print(data.columns)  # چک کن اسم ستون‌ها چیه

# فرض کنیم اسم درست ستون هدف مثلا 'NObeyesdad' باشه

le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

X = data.drop('NObeyesdad', axis=1)

y = data['NObeyesdad']
# 👇 اضافه کن
# X = pd.get_dummies(X)
# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# KFold setup
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store results
results = []
conf_matrices = {}

for name, model in models.items():
    train_acc, val_acc, val_precision, val_recall, val_f1 = [], [], [], [], []
    fold_conf_matrices = []

    for train_idx, val_idx in kfold.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        train_acc.append(accuracy_score(y_train, y_pred_train))
        val_acc.append(accuracy_score(y_val, y_pred_val))
        val_precision.append(precision_score(y_val, y_pred_val, average='macro'))
        val_recall.append(recall_score(y_val, y_pred_val, average='macro'))
        val_f1.append(f1_score(y_val, y_pred_val, average='macro'))

        fold_conf_matrices.append(confusion_matrix(y_val, y_pred_val))

    results.append({
        'Model': name,
        'Train Accuracy': np.mean(train_acc),
        'Validation Accuracy': np.mean(val_acc),
        'Validation Precision': np.mean(val_precision),
        'Validation Recall': np.mean(val_recall),
        'Validation F1': np.mean(val_f1)
    })

    conf_matrices[name] = np.mean(fold_conf_matrices, axis=0)

# Make results DataFrame
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

plt.savefig('train_vs_validation_accuracy.png')  # Save plot
plt.close()  # مهم! ببند تا حافظه خالی شه

# Plot Confusion Matrices
best_models = ['SVM', 'XGBoost']
for model_name in best_models:
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrices[model_name], annot=True, fmt='.0f', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    plt.savefig(f'confusion_matrix_{model_name}.png')  # Save confusion matrix
    plt.close()
