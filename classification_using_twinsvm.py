# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:08:46 2020

@author: Harini
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from twinsvm import twinsvmclassifier
from sklearn.datasets import make_moons, make_circles


X,y = make_moons(n_samples = 10,noise=0.3, random_state=0)
# X,y = make_circles(n_samples = 10,noise=0.2, factor=0.5, random_state=1)
# df = pd.read_csv("nonlin_points1.csv")
# df = pd.read_csv("nonparallel.csv")
# df = pd.read_csv("xor1.csv")
# X=df.values[:,:2]
# y=df.values[:,-1]
print(X)
print(y)
N=len(X)
M=len(X[0])
print("M:"+str(M))
print("N:"+str(N))

#linear kernel
params1 = {'Epsilon1': 0, 'Epsilon2': 0, 'C1': 1, 'C2': 1,'kernel_type':0,'kernel_param': 1}
#rbf kernel
params2 = {'Epsilon1': 0.1, 'Epsilon2': 0.1, 'C1': 1, 'C2': 1,'kernel_type':2,'kernel_param': 2}

def get_C(y1,ypred1,index_list):
    C1=list()
    C2=list()
    C3=list()
    C4=list()
    for i in range(len(y1)):
        if(ypred1[i]==1):  #Actual=1
            # print("Actual:1")
            if(y1[i]==1): #Desired=1
                # print("***C2***")
                C2.append(index_list[i]) 
            if(y1[i]==0): #Desired=0
                # print("***C4***")
                C4.append(index_list[i])
        if(ypred1[i]==0): #Actual=0
            # print("Actual:0")
            if(y1[i]==1): #Desired=1
                # print("***C3***")
                C3.append(index_list[i])
            if(y1[i]==0): #Desired=0
                # print("***C1***")
                C1.append(index_list[i])
    return C1,C2,C3,C4

def get_y(C1,C2,C3,C4):
    y_a=list('d'*N)
    y_b=list('d'*N)
    # print("****************")
    # print("ya:"+str(y_a))
    # print("yb:"+str(y_b))
    # print("****************")
    for i in range(len(X)):
        if i in C1:
            y_a[i]=0
            y_b[i]='d'
        if i in C2:
            y_a[i]='d'
            y_b[i]=0           
        if i in C3:
            y_a[i]=1
            y_b[i]=0
        if i in C4:
            y_a[i]=0
            y_b[i]=1
        # if i in Cd:
        #     y_a[i]="dontcare"
        #     y_b[i]="dontcare"
    return y_a,y_b

y_tree=list()
sol_tree=list()
w_plane1 = list()
b_plane1 = list()
w_plane2 = list()
b_plane2 = list()

def loop(X,y1,dontcare,AX,BX):
    # combine X features with AX and BX
    classifier = twinsvmclassifier(**params2)
    X1 = np.hstack((X,AX,BX))
    # X1 = X
    X_list = list()
    y_list = list()
    index_list = list()
    for i in range(N):
        if i not in dontcare:
            X_list.append(X1[i])
            y_list.append(y1[i])  
            index_list.append(i)
    X1 = np.asarray(X_list)
    y1 = np.asarray(y_list)
    # scaling input features
    X1 = StandardScaler().fit_transform(X1)
    # print("::::::::data::::::::")
    # print(X1)
    # if all 1's are satisfied return 
    if(type(y1)==np.ndarray):
        y11=y1.tolist()
    else:
        y11=y1
    if(y11.count(1)==0):
        return
    # else solve
    else:
        classifier.fit(X1, y1)
        ypred1 = classifier.predict(X1) 
        print("y:"+str(y1))
        print("ypred:"+str(ypred1))
        print("index:"+str(index_list))
        print("dc:"+str(dontcare)) 
        C1,C2,C3,C4=get_C(y1,ypred1,index_list)
        print("C1:"+str(C1))
        print("C2:"+str(C2))
        print("C3:"+str(C3))
        print("C4:"+str(C4)) 
        y1_new,y2_new=get_y(C1,C2,C3,C4)
        dontcare1=list()
        dontcare2=list()
        for i in range(len(X)):
            if y1_new[i]=='d':
                dontcare1.append(i)
            if y2_new[i]=='d':
                dontcare2.append(i)
        print("y1_new:"+str(y1_new))
        print("y2_new:"+str(y2_new))
        print("dontcare1:"+str(dontcare1))
        print("dontcare2:"+str(dontcare2))
        q=input("c?")
        # y_tree.append(y1)
        # sol_tree.append(sol1)
        print("#########")
        loop(X,y1_new,dontcare1,AX,BX)
        print("#########")
        print("@@@@@@@@@")
        loop(X,y2_new,dontcare2,AX,BX)
        print("@@@@@@@@@")

y1=y
#y2=0
dontcare=list()
AX=list(N*[0])
BX=list(N*[0])
AX = np.asarray(AX).reshape(N,1)
BX = np.asarray(BX).reshape(N,1)
print("AX:"+str(AX))
print("BX:"+str(BX))
loop(X,y1,dontcare,AX,BX)






    

        
            
            

            
        




