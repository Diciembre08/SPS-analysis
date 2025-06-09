import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


sns.set(style="whitegrid")
path = r''
data = pd.read_csv(path)
y = data['sppn']
X = data.drop('sppn', axis=1)


X_const = add_constant(X)
vif_df = pd.DataFrame()
vif_df["feature"] = X_const.columns
vif_df["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
vif_df = vif_df[vif_df["feature"] != "const"]
print(vif_df.sort_values(by='VIF', ascending=False))

plt.figure(figsize=(12, 6))
sns.barplot(x="VIF", y="feature", data=vif_df.sort_values(by="VIF", ascending=False), palette="coolwarm")
plt.title("Variance Inflation Factor (VIF) for Each Feature")
plt.xlabel("VIF Value")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 12))
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap='RdBu_r', center=0, square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Matrix of Predictor Variables")
plt.tight_layout()
plt.show()