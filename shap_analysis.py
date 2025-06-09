import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import sklearn
path = u''
data = pd.read_csv(path)
y = data['sppn']
X = data.drop('sppn', axis=1)
cols = ['SEX-1', 'SEX-2', 'LOVE-1', 'LOVE-2', 'LOVE-3', 'COLLEGE-1', 'COLLEGE-2',
        'COLLEGE-3', 'COLLEGE-4', 'pain1', 'pain2', 'pain3', 'pain4','pain5',
        'pain6', 'YEAR', 'AGE', 'HER', 'LBC', 'AS1', 'AS2', 'AS3', 'AS4', 'AS5',
        'CTQ1', 'CTQ2', 'CTQ3', 'CTQ4', 'CTQ5']
X_selected_features = X[cols]
X_train, X_test, y_train, y_test = train_test_split(X_selected_features, y, test_size=0.2, random_state=2023, shuffle=True)
smote = SMOTE(random_state=2023)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
X_train, y_train = sklearn.utils.shuffle(X_train_res, y_train_res, random_state=2023)
xgb = XGBClassifier(learning_rate=0.02,
                   n_estimators=1000,
                   max_depth=20,
                   min_child_weight=1,
                   gamma=9,
                   subsample=0.6,
                   colsample_bytree=1,
                   scale_pos_weight=1,
                   random_state=50,
                   reg_lambda=0.7,
                   booster='gbtree')
xgb.fit(X_train, y_train)
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)
if isinstance(shap_values, list):
    shap_values = shap_values[1]
shap.summary_plot(shap_values, X_test, feature_names=cols)


y_pred = xgb.predict(X_test)
TP_indices = np.where((y_pred == 1) & (y_test == 1))[0]  # True Positive
TN_indices = np.where((y_pred == 0) & (y_test == 0))[0]  # True Negative
FP_indices = np.where((y_pred == 1) & (y_test == 0))[0]  # False Positive
FN_indices = np.where((y_pred == 0) & (y_test == 1))[0]  # False Negative


num_samples_per_category = 2
TP_sample_indices = np.random.choice(TP_indices, min(num_samples_per_category, len(TP_indices)), replace=False)
TN_sample_indices = np.random.choice(TN_indices, min(num_samples_per_category, len(TN_indices)), replace=False)
FP_sample_indices = np.random.choice(FP_indices, min(num_samples_per_category, len(FP_indices)), replace=False)
FN_sample_indices = np.random.choice(FN_indices, min(num_samples_per_category, len(FN_indices)), replace=False)



def plot_shap_waterfall(sample_indices, sample_label):
    for sample_index in sample_indices:
        sample_shap_values = shap_values[sample_index]
        explanation = shap.Explanation(values=sample_shap_values,
                                       base_values=explainer.expected_value,
                                       data=X_test.iloc[sample_index],
                                       feature_names=cols)

        plt.figure()
        shap.waterfall_plot(explanation)
        plt.title(f"Waterfall Plot for {sample_label} Sample {sample_index}")
        plt.show()


plot_shap_waterfall(TP_sample_indices, "True Positive")
plot_shap_waterfall(TN_sample_indices, "True Negative")
plot_shap_waterfall(FP_sample_indices, "False Positive")
plot_shap_waterfall(FN_sample_indices, "False Negative")