# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, LogisticRegressionCV


path = u''
data = pd.read_csv(path)
y = data['sppn'].values
T = data['gender'].values
X = data.drop(['sppn', 'gender'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
    X_scaled, T, y, test_size=0.2, random_state=2023
)

est = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0),
    model_t=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0),
    n_estimators=100,
    min_samples_leaf=10,
    max_depth=50,
    random_state=2025
)
est.fit(Y=y_train, T=T_train, X=X_train)
te_pred = est.effect(X_test)



plt.figure(figsize=(8,5))
plt.hist(te_pred, bins=30, color='skyblue', edgecolor='black')
plt.axvline(te_pred.mean(), color='red', linestyle='--', label=f'Mean Effect = {te_pred.mean():.3f}')
plt.title('Estimated Heterogeneous Treatment Effect (Gender)', fontsize=13)
plt.xlabel('Individual Treatment Effect (Male vs Female)', fontsize=11)
plt.ylabel('Frequency')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()
