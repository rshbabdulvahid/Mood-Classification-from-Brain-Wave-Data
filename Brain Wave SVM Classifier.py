#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from itertools import islice
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score

def plotLCurve(model, X, y):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
    train_sizes = [10, 50, 100, 200, 300, 500, 1000, 1279]
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=splitter, train_sizes=train_sizes, scoring='accuracy')
    train_scores = np.mean(train_scores, axis=1)
    test_scores = np.mean(test_scores, axis=1)
    pyplot.figure()
    pyplot.xlabel("Training examples")
    pyplot.ylabel("Score")
    pyplot.plot(train_sizes, train_scores, 'o-', color="r", label="Training score")
    pyplot.plot(train_sizes, test_scores, 'o-', color="g", label="Cross-validation score")
    
data = pd.read_csv('emotions.csv')
scaler = StandardScaler()
X = np.array(data.iloc[:, :-1])
y = np.array(data.iloc[:, -1])
X = scaler.fit_transform(X)

#Dimensionality reduction to retain 99% of variance. Features reduced from 1.3K to around 400
pca = PCA(n_components=0.99, svd_solver='full')
X_reduced = pca.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, random_state=1)

"""
grid_values = {'C': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 0.3, 1, 5, 10],
                       'gamma': [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.5, 1, 5, 10]}
temp = SVC(kernel='rbf')
model = GridSearchCV(estimator=temp, param_grid=grid_values, scoring='accuracy', refit=True)
"""

#Best settings found via GridSearch for SVM
model = SVC(kernel='rbf', C=3, gamma=0.001)
model.fit(X_train, y_train)

plotLCurve(model, X_train, y_train)
#Cross-validation and test-set scores
print (np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')))
print (classification_report(y_test, model.predict(X_test)))


# In[ ]:




