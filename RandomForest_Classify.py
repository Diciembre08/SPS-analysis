import sklearn
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from scipy import interpolate
path = u''
data = pd.read_csv(path)
y = data['sppn']
X = data.drop('sppn',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023,shuffle=True)

smote = SMOTE(random_state=2023,k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
X_train_res, y_train_res = sklearn.utils.shuffle(X_train_res, y_train_res, random_state=1)

forest = RandomForestClassifier(random_state=10)
parameters = {'learning_rate': [0.3,0.1,0.03,0.02,0.01,0.001, 0.0001], 'n_estimators':[10,50,100,150,200,250,300,350,400,450,500,1000,1100,1200], 'max_depth':[0,4,5,6,7,8,9,10,16,24,32]}

forest = GridSearchCV(forest, parameters, cv=5)
forest.fit(X_train_res, y_train_res)

prediction = forest.predict(X_test)

y_proba = forest.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test,prediction)

print('auc score of rf is {:.4f}'.format(auc))
print("Accuracy : %.4g" % metrics.accuracy_score(y_test, prediction))
print("balanced Accuracy : %.4g" % metrics.balanced_accuracy_score(y_test, prediction))

y_train_proba = forest.predict_proba(X_train)[:,1]


label = pd.DataFrame(data=y_test)
label.to_csv('./label.csv')
test=pd.DataFrame(data=y_proba)
test.to_csv('./test.csv')

label = pd.DataFrame(data=y_train)
label.to_csv('./label_train.csv')
test=pd.DataFrame(data=y_train_proba)
test.to_csv('./forest_train.csv')

fpr, tpr, threshold = metrics.roc_curve(y_test, y_proba)
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
