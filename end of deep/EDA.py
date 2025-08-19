import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split

# ======== بارگذاری دیتاست ========
data = pd.read_csv("../Exam 2/Dataset.csv")  # مسیر فایل را مطابق با سیستم خود تنظیم کنید
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
feature_names = X.columns

# ======== استانداردسازی ویژگی‌ها ========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#
# # ======== 1. توزیع کلاس‌ها ========
# plt.figure(figsize=(7, 4))
# sns.countplot(x=y)
# plt.title("توزیع کلاس‌ها")
# plt.xlabel("کلاس")
# plt.ylabel("تعداد نمونه")
# plt.tight_layout()
# plt.savefig("class_distribution.png")
# plt.close()
#
# # ======== 2. ماتریس همبستگی ========
# plt.figure(figsize=(10, 8))
# sns.heatmap(data.iloc[:, :-1].corr(), annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("ماتریس همبستگی ویژگی‌ها")
# plt.tight_layout()
# plt.savefig("correlation_heatmap.png")
# plt.close()
#
# # ======== 3. Pairplot (برای درک درهم‌تنیدگی کلاس‌ها) ========
# sampled = data.copy()
# sampled['Class'] = y
# if len(feature_names) > 5:
#     sampled = sampled[[*feature_names[:5], 'Class']]
# sns.pairplot(sampled, hue="Class", diag_kind="kde", palette="Set2")
# plt.savefig("pairplot_features.png")
# plt.close()
#
# # ======== 4. کاهش بُعد با PCA ========
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
# pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
# pca_df["Class"] = y.values
#
# plt.figure(figsize=(8, 6))
# sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Class", palette="Set1")
# plt.title("پراکندگی داده‌ها با PCA (2 بعدی)")
# plt.tight_layout()
# plt.savefig("pca_projection.png")
# plt.close()

# # ======== 5. کاهش بُعد با t-SNE ========
# tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
# X_tsne = tsne.fit_transform(X_scaled)
# tsne_df = pd.DataFrame(X_tsne, columns=["Dim1", "Dim2"])
# tsne_df["Class"] = y.values
#
# plt.figure(figsize=(8, 6))
# sns.scatterplot(data=tsne_df, x="Dim1", y="Dim2", hue="Class", palette="Dark2")
# plt.title("نمایش داده‌ها با t-SNE")
# plt.tight_layout()
# plt.savefig("tsne_projection.png")
# plt.close()

# ======== 6. تشخیص داده‌های پرت با Isolation Forest ========
iso = IsolationForest(contamination=0.05, random_state=42)
outlier_pred = iso.fit_predict(X_scaled)
out_df = pd.DataFrame(X_scaled[:, :2], columns=["Feat1", "Feat2"])
out_df["Outlier"] = np.where(outlier_pred == -1, "Outlier", "Inlier")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=out_df, x="Feat1", y="Feat2", hue="Outlier", palette={"Outlier": "red", "Inlier": "green"})
plt.title("تشخیص داده‌های پرت با Isolation Forest")
plt.tight_layout()
plt.savefig("outlier_detection.png")
plt.close()

# ======== 7. اهمیت ویژگی‌ها با Random Forest ========
rf = RandomForestClassifier(random_state=42)
rf.fit(X_scaled, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
plt.title("اهمیت ویژگی‌ها (با Random Forest)")
plt.xlabel("درصد اهمیت")
plt.ylabel("ویژگی")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

print("✅ نمودارها با موفقیت ذخیره شدند!")
