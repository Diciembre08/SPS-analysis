from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2023,shuffle=True)


smote = SMOTE(random_state=2023,k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

X_train_res, y_train_res = sklearn.utils.shuffle(X_train_res, y_train_res, random_state=1)
models = {
    "XGBoost": XGBClassifier(learning_rate=0.02, n_estimators=1000, max_depth=20,
                              min_child_weight=1, gamma=9, subsample=0.6,
                              colsample_bytree=1, scale_pos_weight=1,
                              random_state=50, reg_lambda=0.8, booster='gbtree'),
    "LogisticRegression": LogisticRegression(penalty="l2", C=1, max_iter=200),
    "SVM": SVC(probability=True),
    "MLP": MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 10]),
    "RandomForest": RandomForestClassifier(random_state=10),
    "NaiveBayes": GaussianNB(),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=2024),
    "AdaBoost": AdaBoostClassifier(random_state=2024),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# 训练 + 评估 Brier Score
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_prob = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_prob)
    print(f"{name} Brier Score: {brier:.4f}")