# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:56:54 2019

@author: Binish125
"""


import numpy as np
from sklearn.svm import SVC

X=np.array([[-1,-1],[-2,-1],[1,1],[2,1]])

y= np.array([1,1,2,2])

clf=SVC(kernel='linear')

clf.fit(X,y)

print(clf.predict([[-0.8,-1],[0.8,1]]))



