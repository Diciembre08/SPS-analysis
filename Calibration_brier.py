# -*- coding: utf-8 -*-
import sklearn
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import numpy as np
path = u''
data = pd.read_csv(path)
y = data['sppn']
X = data.drop('sppn',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023,shuffle=True)
print("y_test:", np.unique(y_test, return_counts=True))
smote = SMOTE(random_state=2023)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

X_train_res, y_train_res = sklearn.utils.shuffle(X_train_res, y_train_res, random_state=1)

xgb = XGBClassifier(learning_rate=0.02,
                   n_estimators=1000,  
                   max_depth=20, 
                   min_child_weight=1,  
                   gamma=9,  
                   subsample=0.6, 
                   colsample_btree=1,  
                   scale_pos_weight=1,  
                   random_state=50,  
                   slient=0,
                   reg_lambda = 0.8,
                   booster = 'gbtree'
                   )


xgb.fit(X_train_res, y_train_res)
prediction = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:, 1]


prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')
print(prob_true,prob_pred)

plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='XGBoost')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.xlabel('Predicted probability')
plt.ylabel('True probability in each bin')
plt.title('Calibration Curve')
plt.legend()
plt.grid(False)
plt.tight_layout()
# plt.show()
brier = brier_score_loss(y_test, y_prob)
print(f"Brier Score: {brier:.4f}")

def net_benefit_curve(y_true, y_prob, thresholds=np.linspace(0.0, 0.99, 99)):
    net_benefit_model = []
    net_benefit_all = []

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    N = len(y_true)

    for threshold in thresholds:
        pred_positive = y_prob >= threshold
        TP = np.sum((pred_positive == 1) & (y_true == 1))
        FP = np.sum((pred_positive == 1) & (y_true == 0))

        print(f"threshold={threshold:.2f}, TP={TP}, FP={FP}, pred_positive.sum()={pred_positive.sum()}")

        nb_model = (TP / N) - (FP / N) * (threshold / (1 - threshold))
        nb_all = np.mean(y_true) - (1 - np.mean(y_true)) * (threshold / (1 - threshold))

        net_benefit_model.append(nb_model)
        net_benefit_all.append(nb_all)

    return thresholds, net_benefit_model, net_benefit_all
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(xgb, cv=5, method='isotonic')
calibrated.fit(X_train, y_train)
y_prob = calibrated.predict_proba(X_test)[:, 1]
# y_prob = xgb.predict_proba(X_test)[:, 1]

print("y_prob min:", y_prob.min())
print("y_prob max:", y_prob.max())

thresholds, nb_model, nb_all = net_benefit_curve(y_test, y_prob)
print(nb_model)
print(nb_all)

plt.figure(figsize=(10, 6))

plt.plot(thresholds, nb_model, label="XGBoost Model", color="blue", linestyle="-", linewidth=2)
plt.plot(thresholds, nb_all, label="Treat All", color="gray", linestyle="--", linewidth=2)
plt.plot(thresholds, [0]*len(thresholds), label="Treat None", color="black", linestyle=":")

diff = np.array(nb_model) - np.array(nb_all)
max_diff_idx = np.argmax(np.abs(diff))
plt.scatter(thresholds[max_diff_idx], nb_model[max_diff_idx],
           color="red", s=80, zorder=5, label=f"Max Diff: {diff[max_diff_idx]:.4f}")


plt.xlabel("Threshold Probability", fontsize=12)
plt.ylabel("Net Benefit", fontsize=12)
plt.title("Decision Curve Analysis (XGBoost)", fontsize=14)
plt.ylim(-0.05, 0.25)
# plt.grid(True, alpha=0.3)
plt.grid(False)
plt.legend(loc="upper right")
plt.tight_layout()


plt.axes([0.25, 0.5, 0.5, 0.3]) 
plt.plot(thresholds, diff, color="red", label="Model - Treat All")
plt.axhline(0, color="black", linestyle="--", linewidth=0.5)
plt.ylabel("Difference")
# plt.grid(True, alpha=0.3)
plt.grid(False)

plt.show()