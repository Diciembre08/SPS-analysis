# -*- coding: utf-8 -*-
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve

path = u''
data = pd.read_csv(path)
y = data['sppn']
X = data.drop('sppn',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023,shuffle=True)


smote = SMOTE(random_state=2023,k_neighbors=5)
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
parameters = {'learning_rate': [0.3,0.1,0.03,0.02,0.01,0.001, 0.0001], 'n_estimators':[10,50,100,150,200,250,300,350,400,450,500,1000,1100,1200], 'max_depth':[0,4,5,6,7,8,9,10,16,24,32],
               'booster':['gbtree','gblinear']}

xgb = GridSearchCV(xgb,parameters,cv=5)


xgb.fit(X_train_res, y_train_res)
prediction = xgb.predict(X_test)
auc = roc_auc_score(y_test, prediction)

print('auc score is {:.4f}'.format(auc))


print("Accuracy on training set: {:.3f}".format(xgb.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(xgb.score(X_test, y_test)))
feature_importance = xgb.feature_importances_
y_proba = xgb.predict_proba(X_test)[:,1]
y_train_proba = xgb.predict_proba(X_train)[:,1]


y_test, y_pred = y_test, xgb.predict(X_test)
print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
print("balanced Accuracy : %.4g" % metrics.balanced_accuracy_score(y_test, y_pred))
print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_train_proba))




label = pd.DataFrame(data=y_train)
label.to_csv('./label_train.csv')
test=pd.DataFrame(data=y_train_proba)
test.to_csv('./xgb_train.csv')
print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_proba))

#------------- external dataset -----------------#
from sklearn.metrics import average_precision_score, precision_recall_curve

external_path = r''
external_data = pd.read_csv(external_path)
y_external = external_data['sppn']
X_external = external_data.drop('sppn', axis=1)


external_pred = xgb.predict(X_external)
external_proba = xgb.predict_proba(X_external)[:, 1]


external_auc = roc_auc_score(y_external, external_proba)
external_bal_acc = metrics.balanced_accuracy_score(y_external, external_pred)
external_auprc = average_precision_score(y_external, external_proba)

print("External AUC Score       : {:.4f}".format(external_auc))
print("External Balanced Acc    : {:.4f}".format(external_bal_acc))
print("External AUPRC Score     : {:.4f}".format(external_auprc))


# ==== ROC Curve ====
fpr, tpr, _ = roc_curve(y_external, external_proba)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.4f)' % external_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (External Dataset)')
plt.legend(loc="lower right")
plt.grid(False)

