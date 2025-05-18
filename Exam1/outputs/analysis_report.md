
# Obesity Classification Analysis Report

## Dataset Overview
- Number of samples: 2111
- Number of features: 20
- Classes: ['Insufficient_Weight' 'Normal_Weight' 'Obesity_Type_I' 'Obesity_Type_II'
 'Obesity_Type_III' 'Overweight_Level_I' 'Overweight_Level_II']

## Methodology
- **Preprocessing**: One-Hot Encoding, SMOTE for class imbalance, StandardScaler for scaling.
- **Feature Selection**: Random Forest and SHAP for importance analysis.
- **Models**: Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost, LightGBM, Voting Classifier.
- **Evaluation**: 5-Fold Cross-Validation, metrics include Accuracy, Precision, Recall, F1-Score, ROC-AUC.

## Results
Results table could not be generated due to missing 'tabulate' package. Please install tabulate with 'pip install tabulate'.
                 Model  Train Accuracy  ...  Validation F1  Validation ROC-AUC
0  Logistic Regression        0.893257  ...       0.877079            0.984207
1        Decision Tree        0.994020  ...       0.934679            0.963987
2        Random Forest        1.000000  ...       0.950550            0.996610
3                  KNN        0.902417  ...       0.818004            0.938558
4                  SVM        0.933333  ...       0.871247            0.985669
5              XGBoost        1.000000  ...       0.970378            0.998867
6             LightGBM        1.000000  ...       0.976053            0.999015
7    Voting Classifier        1.000000  ...       0.972497            0.998978

[8 rows x 7 columns]

## Statistical Analysis
- T-test between Random Forest and XGBoost: t-statistic=-3.771, p-value=0.020
- T-test between Random Forest and LightGBM: t-statistic=-5.483, p-value=0.005

## Visualizations
- Feature importance and SHAP analysis saved in 'outputs/feature_importance.png' and 'outputs/shap_feature_importance.png'.
- ROC Curves saved in 'outputs/roc_curves.png'.
- Confusion matrices for top models saved in 'outputs/confusion_matrix_<model>.png'.

## Conclusion
The Random Forest, XGBoost, and LightGBM models performed best, with the Voting Classifier providing robust results. Further analysis with larger datasets or additional features could enhance performance.
