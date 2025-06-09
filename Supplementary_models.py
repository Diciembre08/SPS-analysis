# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import sklearn.utils
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_recall_curve, auc
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from sklearn.preprocessing import label_binarize

path = u''
data = pd.read_csv(path)
y = data['sppn']
X = data.drop('sppn',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2023,shuffle=True)
smote = SMOTE(random_state=2023)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
X_train_res, y_train_res = sklearn.utils.shuffle(X_train_res, y_train_res, random_state=1)


def bootstrap_auc_ci(y_true, y_scores, n_bootstraps=1000, ci=0.95, random_seed=42):
    rng = np.random.RandomState(random_seed)
    bootstrapped_scores = []
    n_samples = len(y_true)
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    for i in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    lower = np.percentile(sorted_scores, (1 - ci) / 2 * 100)
    upper = np.percentile(sorted_scores, (1 + ci) / 2 * 100)

    return lower, upper



path = u''
data = pd.read_csv(path)
y = data['sppn']
X = data.drop('sppn', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2023, shuffle=True)


smote = SMOTE(random_state=2023)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
X_train_res, y_train_res = sklearn.utils.shuffle(X_train_res, y_train_res, random_state=1)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]


    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)


    train_auc = roc_auc_score(y_train, y_train_prob)
    train_bal_acc = balanced_accuracy_score(y_train, y_train_pred)
    precision, recall, _ = precision_recall_curve(y_train, y_train_prob)
    train_auprc = auc(recall, precision)


    test_auc = roc_auc_score(y_test, y_test_prob)
    test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
    test_auprc = auc(recall, precision)


    ci_lower, ci_upper = bootstrap_auc_ci(y_test, y_test_prob, n_bootstraps=1000, ci=0.95)

    return train_auc, test_auc, (ci_lower, ci_upper), test_bal_acc, test_auprc



models = {
    "LDA": LinearDiscriminantAnalysis(),
    "Naive Bayes": GaussianNB(),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=2024),
    "LightGBM": LGBMClassifier(num_leaves=11, random_state=2024),
    "AdaBoost": AdaBoostClassifier(random_state=2024),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []

for name, model in models.items():
    train_auc, test_auc, (ci_lower, ci_upper), test_bal_acc, test_auprc = evaluate_model(
        model, X_train_res, y_train_res, X_test, y_test
    )
    results.append({
        "Model": name,
        "Train AUC": round(train_auc, 4),
        "Test AUC": f"{test_auc:.4f} ({ci_lower:.4f}-{ci_upper:.4f})",
        "Balanced Acc": round(test_bal_acc, 4),
        "Test AUPRC": round(test_auprc, 4)
    })
print(f"{test_auc:.4f} ({ci_lower:.4f}-{ci_upper:.4f})")
results_df = pd.DataFrame(results)
print(results_df)