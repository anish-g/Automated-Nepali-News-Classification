# -*- coding: utf-8 -*-
"""
Created on Sat May 25 20:37:29 2019

@author: Binish125
"""

import numpy as np

arr=np.empty((0,2))

arr2=np.empty((0,2))

arr2=np.append(arr2,[('zxcq','qwe')],axis=0)

print(arr2)

arr2=np.append(arr2,[('zxcq','qwe')],axis=0)
arr2=np.append(arr2,[('zxcq','qwe')],axis=0)

arr=np.append(arr,arr2,axis=0)

print(arr2.ndim)
print(arr.ndim)

print(arr)