import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

path = u''
data = pd.read_csv(path)
y = data['sppn']
X = data.drop('sppn', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = LogisticRegressionCV(
    Cs=1000,
    cv=5,
    penalty='l1',
    solver='saga',
    max_iter=1000,
    scoring='roc_auc',
    random_state=42
)
lasso.fit(X_scaled, y)


coef = lasso.coef_[0]
selected_features = X.columns[coef != 0]
dropped_features = X.columns[coef == 0]

print("✔ Remained Features:")
print(selected_features.tolist())
print("\n✘ Dropped Features:")
print(dropped_features.tolist())


nonzero_coefs = coef[coef != 0]
feature_names = selected_features.tolist()

plt.figure(figsize=(10, 6))
plt.barh(feature_names, nonzero_coefs, color='skyblue')
plt.xlabel("LASSO Coefficient")
plt.title("Feature Importance via LASSO")
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()