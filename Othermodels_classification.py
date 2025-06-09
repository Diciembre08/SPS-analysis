# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py  # a tool for visualizing
# py.init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore')
# import yaml
# yaml.warnings({'YAMLLoadWarning': False})
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer
import sklearn


path = u''
data = pd.read_csv(path)
y = data['sppn']
X = data.drop('sppn',axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023,shuffle=True)
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)



smote = SMOTE(random_state=2023, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
X_train_res, y_train_res = sklearn.utils.shuffle(X_train_res, y_train_res, random_state=1)

logistic = LogisticRegression(penalty="l2",C=1,max_iter=200)
svm = SVC(probability=True)
mlp = MLPClassifier(solver='lbfgs', random_state=0,hidden_layer_sizes=[10,10])

logistic.fit(X_train_res,y_train_res)
#svm.fit(X_train_res,y_train_res)
mlp.fit(X_train_res,y_train_res)

prediction_lg = logistic.predict(X_test)
#prediction_svm = svm.predict(X_test)
prediction_mlp = mlp.predict(X_test)

feature = logistic.coef_
print(feature)
feature= pd.DataFrame(feature)
feature.to_csv('./features')


print("lg Accuracy : %.4g" % metrics.accuracy_score(y_test, prediction_lg))
print("lg balanced Accuracy : %.4g" % metrics.balanced_accuracy_score(y_test, prediction_lg))
#print("svm Accuracy : %.4g" % metrics.accuracy_score(y_test, prediction_svm))
#("svm balanced Accuracy : %.4g" % metrics.balanced_accuracy_score(y_test, prediction_svm))
print("mlp Accuracy : %.4g" % metrics.accuracy_score(y_test, prediction_mlp))
print("mlp balanced Accuracy : %.4g" % metrics.balanced_accuracy_score(y_test, prediction_mlp))


# print(logistic.score(X_test,y_test))

y_proba_logistc = logistic.predict_proba(X_test)[:,1]
auc_logistic = roc_auc_score(y_test,y_proba_logistc)

#y_svc_prob = svm.predict_proba(X_test)[:,1]
#auc_svm = roc_auc_score(y_test,y_svc_prob)

y_mlp_prob = mlp.predict_proba(X_test)[:,1]
auc_mlp = roc_auc_score(y_test,y_mlp_prob)

print('auc score of logistic is {:.4f}'.format(auc_logistic))
#print('auc score of svc is {:.4f}'.format(auc_svm))
print('auc score of mlp is {:.4f}'.format(auc_mlp))




label = pd.DataFrame(data=y_test)
prob_lg=pd.DataFrame(data=y_proba_logistc)

prob_mlp = pd.DataFrame(data = y_mlp_prob)
label.to_csv('./label.csv')
prob_lg.to_csv('./lg.csv')
prob_mlp.to_csv('./mlp.csv')
''''''
y_proba_logistc_train = logistic.predict_proba(X_train)[:,1]
#y_svc_prob_train = svm.predict_proba(X_train)[:,1]
y_mlp_prob_train = mlp.predict_proba(X_train)[:,1]
prob_lg=pd.DataFrame(data=y_proba_logistc_train)
#prob_svm = pd.DataFrame(data=y_svc_prob_train)
prob_mlp = pd.DataFrame(data = y_mlp_prob_train)
prob_lg.to_csv('./lg_train.csv')
#prob_svm.to_csv('./svm_train.csv')
prob_mlp.to_csv('./mlp_train.csv')
auc_logistic = roc_auc_score(y_train,y_proba_logistc_train)
print('auc score of logistic train is {:.4f}'.format(auc_logistic))