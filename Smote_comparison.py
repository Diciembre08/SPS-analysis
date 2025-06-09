# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import sklearn

path = u''
data = pd.read_csv(path)
y = data['sppn']
X = data.drop('sppn', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023, shuffle=True)
results = {}
xgb_raw = XGBClassifier(
                              learning_rate=0.02,
                              n_estimators=1000,
                              max_depth=20,
                              min_child_weight=1,
                              gamma=9,
                              subsample=0.6,
                              colsample_btree=1,
                              scale_pos_weight=1,
                              random_state=50,
                              reg_lambda=0.7,
                              booster='gbtree'
)
xgb_raw.fit(X_train, y_train)
y_prob_raw = xgb_raw.predict_proba(X_test)[:, 1]
fpr_raw, tpr_raw, _ = roc_curve(y_test, y_prob_raw)
roc_auc_raw = auc(fpr_raw, tpr_raw)
results['No SMOTE'] = (fpr_raw, tpr_raw, roc_auc_raw)


for k in [3, 5, 10]:
    smote = SMOTE(k_neighbors=k, random_state=2023)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    X_train_res, y_train_res = sklearn.utils.shuffle(X_train_res, y_train_res, random_state=1)
    if k == 5:
        model = XGBClassifier(learning_rate=0.02,
                              n_estimators=1000,
                              max_depth=20,
                              min_child_weight=1,
                              gamma=9,
                              subsample=0.6,
                              colsample_btree=1,
                              scale_pos_weight=1,
                              random_state=50,
                              reg_lambda=0.8,
                              booster='gbtree')
    else:
        model = XGBClassifier()
    model.fit(X_train_res, y_train_res)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    results[f'SMOTE(k={k})'] = (fpr, tpr, roc_auc)

plt.figure(figsize=(8, 6))
for label, (fpr, tpr, roc_auc) in results.items():
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves with Different SMOTE k_neighbors')
plt.legend(loc='lower right')
plt.grid(False)
plt.tight_layout()
plt.show()