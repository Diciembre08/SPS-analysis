import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 读取数据
path = u''
data = pd.read_csv(path)

# 分离正例和负例
positive_samples = data[data['sppn'] == 1]
negative_samples = data[data['sppn'] == 0]

# 随机采样 100 个正例和 100 个负例
positive_sampled = positive_samples.sample(n=600, random_state=42)
negative_sampled = negative_samples.sample(n=600, random_state=42)

# 合并样本
sampled_data = pd.concat([positive_sampled, negative_sampled])

# 特征和标签
y = sampled_data['sppn']
X_selected_features = sampled_data.drop(['sppn'], axis=1)

# 进行 t-SNE 降维
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X_selected_features)

# 绘制 t-SNE 图
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                      c=y.map({1: 'lightskyblue', 0: 'lightcoral'}),
                      s=20, alpha=0.6)
plt.colorbar()
plt.title("t-SNE Visualization of Selected Features (Sampled 100 Positives and 100 Negatives)")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.show()