# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:02:20 2020

@author: Harini
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from twinsvm import twinsvmclassifier

params1 = {'Epsilon1': 0.1, 'Epsilon2': 0.1, 'C1': 1, 'C2': 1,'kernel_type':0,'kernel_param': 1}
params2 = {'Epsilon1': 0.1, 'Epsilon2': 0.1, 'C1': 1, 'C2': 1,'kernel_type':2,'kernel_param': 2}

names = ["Twin SVM","Twin SVM with RBF Kernel"]
classifiers = [
    twinsvmclassifier(**params1),
    twinsvmclassifier(**params2)]

X = [[1,2],[4,2]]
X = np.asarray(X)
y = [0,1]
y = np.asarray(y)

for name, clf in zip(names, classifiers):
    print(name)
    clf.fit(X,y)
    ypred = clf.predict(X)
    print("ypred:"+str(ypred))
    