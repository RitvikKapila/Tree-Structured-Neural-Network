# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:57:54 2020

@author: Harini
"""

import math
import numpy as np

def kernelfunction(Type, u, v, p):
    if(Type==1): #linear
        return np.dot(u,v)
    if(Type==2): #rbf
        temp = u-v
        return pow(math.e,(-np.dot(temp,temp)/(p**2)))
