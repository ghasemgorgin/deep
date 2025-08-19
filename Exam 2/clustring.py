
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from hpelm import ELM
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import time
import seaborn as sns
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ELM Wrapper
class ELMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neurons=1000, func='sigm', kernel=None, rbf_width=None):
        self.n_neurons = n_neurons
        self.func = func
        self.kernel = kernel
        self.rbf_width = rbf_width
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        self.classes_ = np.unique(y)
        y_one_hot = pd.get_dummies(y).values
        if self.kernel == 'rbf':
            self.model = ELM(X.shape[1], y_one_hot.shape[1], classification="wc", batch=1000, w=self.rbf_width)
        else:
            self.model = ELM(X.shape[1], y_one_hot.shape[1], classification="c", batch=1000)
        self.model.add_neurons(self.n_neurons, self.func)
        self.model.train(X, y_one_hot, "c")
        return self

    def predict(self, X):
        X = np.asarray(X)
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        X = np.asarray(X)
        proba = self.model.predict(X)
        proba = np.clip(proba, 0, None)
        proba_sum = proba.sum(axis=1, keepdims=True)
        proba_sum = np.where(proba_sum == 0, 1, proba_sum)
        return proba / proba_sum

    def get_params(self, deep=True):
        return {"n_neurons": self.n_neurons, "func": self.func, "kernel": self.kernel, "rbf_width": self.rbf_width}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


# Load dataset
data = pd.read_csv("Dataset.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Step 1: Clustering with K-means
kmeans = KMeans(n_clusters=6, random_state=42)
train_clusters = kmeans.fit_predict(X_train_scaled)
test_clusters = kmeans.predict(X_test_scaled)

# Step 2: Visualize clustering with t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_scaled)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train_tsne[:, 0], y=X_train_tsne[:, 1], hue=y_train, palette='deep', legend='full')
plt.title('t-SNE Visualization of Training Data with True Labels')
plt.savefig('tsne_true_labels.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train_tsne[:, 0], y=X_train_tsne[:, 1], hue=train_clusters, palette='deep', legend='full')
plt.title('t-SNE Visualization of Training Data with K-means Clusters')
plt.savefig('tsne_clusters.png')
plt.close()

# Step 3: Analyze clusters
cluster_class_dist = {}
for cluster in np.unique(train_clusters):
    cluster_indices = np.where(train_clusters == cluster)[0]
    cluster_labels = y_train[cluster_indices]
    cluster_class_dist[cluster] = np.bincount(cluster_labels, minlength=len(np.unique(y)))
    print(f"Cluster {cluster} Class Distribution:\n{cluster_class_dist[cluster]}")

# Step 4: Classify based on clusters
results = []
class_labels = np.unique(y)
problem_classes = [1, 4, 5]

for cluster in np.unique(train_clusters):
    print(f"\nProcessing Cluster {cluster}")
    cluster_indices_train = np.where(train_clusters == cluster)[0]
    cluster_indices_test = np.where(test_clusters == cluster)[0]

    X_cluster_train = X_train_scaled.iloc[cluster_indices_train]
    y_cluster_train = y_train[cluster_indices_train]
    X_cluster_test = X_test_scaled.iloc[cluster_indices_test]
    y_cluster_test = y_test[cluster_indices_test]

    # Check if cluster has enough data and classes
    if len(X_cluster_train) < 2 or len(np.unique(y_cluster_train)) < 2:
        print(f"Cluster {cluster} has insufficient data or classes, skipping classification.")
        continue

    # Determine if cluster is complex (contains problem classes)
    is_complex = any(np.isin(y_cluster_train, problem_classes))

    if is_complex:
        print(f"Cluster {cluster} is complex, using Stacking Classifier with ELM")
        # Use Stacking Classifier with ELM for complex clusters
        elm = ELMWrapper(n_neurons=1000, func='sigm', kernel='rbf', rbf_width=0.08)
        estimators = [
            ('elm', elm),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('lgbm', LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42))
        ]
        stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000),
                                      n_jobs=1)
        stacking.fit(X_cluster_train, y_cluster_train)

        # Second stage for problem classes
        y_pred_cluster = stacking.predict(X_cluster_test)
        y_pred_proba = stacking.predict_proba(X_cluster_test)
        problem_indices = np.isin(y_pred_cluster, problem_classes)
        X_test_problem = X_cluster_test[problem_indices]
        y_test_problem = y_cluster_test[problem_indices]
        probas_problem = y_pred_proba[problem_indices]

        if len(np.unique(y_test_problem)) > 1:
            elm_second = ELMWrapper(n_neurons=1000, func='sigm', kernel='rbf', rbf_width=0.08)
            X_second = np.hstack([X_test_problem, probas_problem])
            X_train_second = np.hstack([X_cluster_train, stacking.predict_proba(X_cluster_train)])
            y_train_second = y_cluster_train
            train_problem_indices = np.isin(y_cluster_train, problem_classes)
            X_train_second = X_train_second[train_problem_indices]
            y_train_second = y_cluster_train[train_problem_indices]
            class_mapping = {1: 0, 4: 1, 5: 2}
            y_train_second_mapped = np.array([class_mapping[cls] for cls in y_train_second])
            elm_second.fit(X_train_second, y_train_second_mapped)
            y_pred_second_mapped = elm_second.predict(X_second)
            reverse_mapping = {0: 1, 1: 4, 2: 5}
            y_pred_second = np.array([reverse_mapping[cls] for cls in y_pred_second_mapped])
            y_pred_cluster[problem_indices] = y_pred_second
    else:
        print(f"Cluster {cluster} is simple, using Random Forest")
        # Use simpler model for non-complex clusters
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_cluster_train, y_cluster_train)
        y_pred_cluster = rf.predict(X_cluster_test)

    # Evaluate cluster
    acc = accuracy_score(y_cluster_test, y_pred_cluster)
    cm = confusion_matrix(y_cluster_test, y_pred_cluster)
    report = classification_report(y_cluster_test, y_pred_cluster, zero_division=0)
    results.append((cluster, acc, cm, report))
    print(f"Cluster {cluster} Accuracy: {acc:.4f}")
    print(f"Confusion Matrix for Cluster {cluster}:\n{cm}")
    print(f"Classification Report for Cluster {cluster}:\n{report}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix (Cluster {cluster})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_cluster_{cluster}.png')
    plt.close()

# Combine predictions for overall evaluation
y_pred_all = np.zeros_like(y_test)
for cluster in np.unique(train_clusters):
    cluster_indices_test = np.where(test_clusters == cluster)[0]
    if len(cluster_indices_test) == 0:
        continue
    X_cluster_test = X_test_scaled.iloc[cluster_indices_test]
    y_cluster_test = y_test[cluster_indices_test]

    if any(np.isin(y_cluster_test, problem_classes)):
        elm = ELMWrapper(n_neurons=1000, func='sigm', kernel='rbf', rbf_width=0.08)
        estimators = [
            ('elm', elm),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('lgbm', LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42))
        ]
        stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000),
                                      n_jobs=1)
        stacking.fit(X_train_scaled.iloc[np.where(train_clusters == cluster)[0]],
                     y_train[np.where(train_clusters == cluster)[0]])
        y_pred_cluster = stacking.predict(X_cluster_test)

        y_pred_proba = stacking.predict_proba(X_cluster_test)
        problem_indices = np.isin(y_pred_cluster, problem_classes)
        X_test_problem = X_cluster_test[problem_indices]
        y_test_problem = y_cluster_test[problem_indices]
        probas_problem = y_pred_proba[problem_indices]

        if len(np.unique(y_test_problem)) > 1:
            elm_second = ELMWrapper(n_neurons=1000, func='sigm', kernel='rbf', rbf_width=0.08)
            X_second = np.hstack([X_test_problem, probas_problem])
            X_train_second = np.hstack([X_train_scaled.iloc[np.where(train_clusters == cluster)[0]],
                                        stacking.predict_proba(
                                            X_train_scaled.iloc[np.where(train_clusters == cluster)[0]])])
            y_train_second = y_train[np.where(train_clusters == cluster)[0]]
            train_problem_indices = np.isin(y_train_second, problem_classes)
            X_train_second = X_train_second[train_problem_indices]
            y_train_second = y_train_second[train_problem_indices]
            class_mapping = {1: 0, 4: 1, 5: 2}
            y_train_second_mapped = np.array([class_mapping[cls] for cls in y_train_second])
            elm_second.fit(X_train_second, y_train_second_mapped)
            y_pred_second_mapped = elm_second.predict(X_second)
            reverse_mapping = {0: 1, 1: 4, 2: 5}
            y_pred_second = np.array([reverse_mapping[cls] for cls in y_pred_second_mapped])
            y_pred_cluster[problem_indices] = y_pred_second
    else:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled.iloc[np.where(train_clusters == cluster)[0]],
               y_train[np.where(train_clusters == cluster)[0]])
        y_pred_cluster = rf.predict(X_cluster_test)

    y_pred_all[cluster_indices_test] = y_pred_cluster

# Overall evaluation
overall_accuracy = accuracy_score(y_test, y_pred_all)
overall_cm = confusion_matrix(y_test, y_pred_all)
overall_report = classification_report(y_test, y_pred_all, zero_division=0)
print(f"\nOverall Test Accuracy: {overall_accuracy:.4f}")
print(f"Overall Confusion Matrix:\n{overall_cm}")
print(f"Overall Classification Report:\n{overall_report}")

# Plot overall confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Overall Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('overall_confusion_matrix.png')
plt.close()
