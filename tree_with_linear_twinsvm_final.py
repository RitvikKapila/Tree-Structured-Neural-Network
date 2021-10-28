# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:57:46 2020

@author: Harini
"""
import numpy as np
from scipy.optimize import linprog 
from twinsvm import twinsvmclassifier
import random

def optimize(X,y,delta, alpha):
    (m,n) = X.shape
    K = (y == 0).sum()
    M = (y == 1).sum()
    A = np.zeros((2*n,m))
    B = np.zeros((m))
    for i in range(m):
        if y[i] == 0:     
            V = np.zeros((2,m))
            V[0][i] = 1
            V[1][i] = -1
            A = np.vstack((A,V))
            for j in range(n):
                A[j][i] = X[i][j]
                A[j+n][i] = -1*X[i][j]
                B[i] = -delta
        else:  
            V = np.zeros((2,m))
            V[0][i] = -1
            V[1][i] = 1
            A = np.vstack((A,V)) 
            for j in range(n):
                A[j][i] = X[i][j]
                A[j+n][i] = -1*X[i][j]
                B[i] = delta
    A = np.vstack((A, np.zeros((n,m))))   # dimension of A is 2n+2m+n, m
    s1 = np.zeros((3*n+2*m))
    s2 = np.zeros((3*n+2*m))
    W = np.zeros((2*n+2*m+n))
    for i in range(m):
        if y[i] == 0:
            W[2*i+2*n] = 0                  #
            W[2*i+2*n+1] = 1/K              #
            s2[2*n+2*i] = -1
        else:
            W[2*i+2*n] = 0                  #
            W[2*i+2*n+1] = 1/M              #
            s1[2*n+2*i] = -1
    for i in range(n):
        W[i+2*n+2*m] = alpha   ##########
    A = A.T
    #Ax=B the equality contraints
    C = np.zeros((2*n, 3*n+2*m))
    D = np.zeros((2*n+2))
    D[2*n+2-1] = -1
    D[2*n+2-2] = -1
    for i in range (n):
        C[i,i] = 1
        C[i+n,i] = -1
        C[i, i+n] = -1
        C[i+n, i+n] = 1
        C[i, i+2*n+2*m] = -1
        C[i+n, i+2*n+2*m] = -1
        # Cx<=D
    C = np.vstack((C, s1))
    C = np.vstack((C, s2))
    
    res = linprog(W, A_ub=C, b_ub=D, A_eq=A, b_eq=B,  bounds=(0, None), method = 'simplex') 
    wd = res.x[:n]-res.x[n:2*n]
    si1 = np.zeros((m))
    si2 = np.zeros((m))
    for i in range(m):
        si1[i] = res.x[i*2+2*n]
    for i in range(m):
        si2[i] = res.x[i*2+2*n+1]
    print("wd:"+str(wd))
    return np.matrix(wd),None

def y_hat(X, weights, delta, y,twin_svm_en,clf):
    if(twin_svm_en == 0):
        n = len(X)
        y_res = np.asarray(np.dot(weights, X.T))
        for i in range(n):
            if y_res[0][i] >= delta:
                y_res[0][i] = 1
            elif y_res[0][i] <= -delta:
                y_res[0][i] = 0
            else :
                y_res[0][i] = 1-y[i]
    else:
        Xx = X[:,1:len(X[0])]
        y_pred = clf.predict(Xx)
        y_res = np.array([y_pred])
        
    return y_res[0]


def classes(y,y_res):
    C1 = []
    C2 = []
    C3 = []
    C4 = []
    y = y.astype(int)
    y_res = y_res.astype(int)
    for i in range(len(y_res)): 
        if y[i] == 0 and y_res[i] == 0:
            C1.append(i)
        if y[i] == 1 and y_res[i] == 1:
            C2.append(i)
        if y[i] == 1 and y_res[i] == 0:
            C3.append(i)
        if y[i] == 0 and y_res[i] == 1:
            C4.append(i)
    return C1, C2, C3, C4


#plane1 < plane2 => class 0
#plane1 >= plane2 => class 1
# plane1-plane2 < 0 => class 0 
# plane1-plane2 >= 0 => class 1
#C3 => desired 1 we get 0
#so plane1 < plane2 
#   plane1-plane2 < 0
#   Wa decided by min(plane1-plane2)
def find_Wa(X, C3, weights):
    res = 0
    y=0
    for i in C3:
        if(weights.shape[1]==X.shape[1]):
            y = np.asarray(np.dot(weights, X[i].T))
            res = min(res, y[0][0])
        else:
            print("twinsvm")
            XX = X[i,1:len(X[i])]
            weights = np.asarray(weights)
            X= np.asarray(X)
            b1 = weights[0][0]
            w1 = weights[0][1:X.shape[1]]
            b2 = weights[0][X.shape[1]]
            w2 = weights[0][X.shape[1]+1:2*(X.shape[1])]
            w1mod = np.linalg.norm(w1)
            w2mod = np.linalg.norm(w2)
            y1 = np.dot(XX,w1)+ b1
            y2 = np.dot(XX,w2)+ b2
            distFromPlane1 = y1/w1mod 
            distFromPlane2 = y2/w2mod 
            dis = (distFromPlane1 - distFromPlane2)
            res = min(res, dis)
    return -res+0.01+delta

#C4 => desired 0 we get 1
#so plane1 >= plane2
#   plane1-plane2 >= 0
#   Wb decided by max(plane1-plane2)
def find_Wb(X, C4, weights):
    res = 0
    y=0
    for i in C4:
        if(weights.shape[1]==X.shape[1]):
            y = np.asarray(np.dot(weights, X[i].T))
            res = max(res, y[0][0])
        else:
            print("twinsvm")
            XX = X[i,1:len(X[i])]
            weights = np.asarray(weights)
            X= np.asarray(X)
            b1 = weights[0][0]
            w1 = weights[0][1:X.shape[1]]
            b2 = weights[0][X.shape[1]]
            w2 = weights[0][X.shape[1]+1:2*(X.shape[1])]
            w1mod = np.linalg.norm(w1)
            w2mod = np.linalg.norm(w2)
            y1 = np.dot(XX,w1)+ b1
            y2 = np.dot(XX,w2)+ b2
            distFromPlane1 = y1/w1mod 
            distFromPlane2 = y2/w2mod 
            dis = (distFromPlane1 - distFromPlane2)
            res = max(res, dis)
    return -res-0.01-delta

def quantize(a,b,twin_node,twin_classifier,XX):
    if(twin_node == 0):
        res = np.dot(a,b.T)
        print("res_before_quant:"+str(res))
        res = res.astype(int)
        for j in range(res.shape[0]):
            if(res[j][0] >= delta):
                res[j][0] = 1
            elif(res[j][0] <= -delta):
                res[j][0] = 0
            else:
                print("delta condition not met")
        print("res_after_quant:"+str(res))
    else:
        Xd = a[:,1:XX.shape[1]]
        if(a.shape[1] > XX.shape[1]):
            XAB = a[:,XX.shape[1]:a.shape[1]]
        print("twin_svm_res")
        b1 = b[0][0]
        w1 = b[0][1:XX.shape[1]].reshape(XX.shape[1]-1,1)
        b2 = b[0][XX.shape[1]]
        w2 = b[0][XX.shape[1]+1:2*(XX.shape[1])].reshape(XX.shape[1]-1,1)
        wAB = b[0][2*(XX.shape[1]):b.shape[1]].reshape(-1,1)
        w1mod = np.linalg.norm(w1)
        w2mod = np.linalg.norm(w2)
        y1 = np.dot(Xd,w1)+ b1*np.ones((XX.shape[0],1))
        y2 = np.dot(Xd,w2)+ b2*np.ones((XX.shape[0],1))
        distFromPlane1 = y1/w1mod 
        distFromPlane2 = y2/w2mod 
        res = (distFromPlane1 - distFromPlane2) + np.dot(XAB,wAB)
        print("res_twin_bef:"+str(res))
        for j in range(res.shape[0]):
            if(res[j][0] >= 0):
                res[j][0] = 1
            elif(res[j][0] < 0):
                res[j][0] = 0
        print("res_twin_aft:"+str(res))
    return res

class Neuron:
    
    def __init__(self,n,twin_node,twin_classifier):
        self.X = np.zeros((1,n))
        self.inp = n
        self.A = None
        self.B = None
        self.twin_node = twin_node
        self.twin_classifier = twin_classifier
        self.weight = 0
        
    def insert(self,neuron_type, weight):     #### Type is a restricted variable, Type was always set to A, weigth to 1
        if neuron_type == 'A':
            self.A = Neuron(self.inp,0,None)
            self.X = np.hstack((self.X, np.array([[weight]])))
        else:
            self.B = Neuron(self.inp,0,None)
            self.X = np.hstack((self.X, np.array([[weight]])))
            
    def calculate(self,XX):
        if self.A != None and self.B != None:
            print("AB")
            y_1 = self.A.calculate(XX)
            y_2 = self.B.calculate(XX)
            print("y_1:"+str(y_1))
            print("y_2:"+str(y_2))
            res = quantize(np.hstack((XX, y_1, y_2)),self.X,self.twin_node,self.twin_classifier,XX)
            return res
        elif self.A != None:
            print("A")
            y_1 = self.A.calculate(XX)
            print("y_1:"+str(y_1))
            res = quantize(np.hstack((XX, y_1)),self.X,self.twin_node,self.twin_classifier,XX)
            return res
        elif self.B != None:
            print("B")
            y_2 = self.B.calculate(XX)
            print("y_2:"+str(y_2))
            res = quantize(np.hstack((XX, y_2)),self.X,self.twin_node,self.twin_classifier,XX)
            return res
        else:
            print("no child node")
            res = quantize(XX,self.X,self.twin_node,self.twin_classifier,XX)
            return res

def get_twin_classifier(X,y):
    while(True):
        C1value = random.uniform(0,5)
        C2value = random.uniform(0,5)
        params = {'Epsilon1': 0.1, 'Epsilon2': 0.1, 'C1': C1value, 'C2': C2value,'kernel_type':0,'kernel_param': 2}
        clf=twinsvmclassifier(**params)
        clf.fit(X,y)
        y_pred = clf.predict(X)
        C1,C2,C3,C4 = classes(y, y_pred)
        if(not(len(C1)==0 or len(C2)==0)):
            print("#######")
            print(C1)
            print(C2)
            print(C3)
            print(C4)
            print("#######")
            print("wb:"+str(clf.getwb()))
            return clf    
        
def twin(X,y):
    XX = X[:,1:len(X[0])]
    yy = y
    print("XX:"+str(XX))
    print("yy:"+str(yy))
    clf = get_twin_classifier(XX, yy)
    clf.fit(XX,yy)
    w1,b1,w2,b2 = clf.getwb()
    w1 = w1.T
    w2 = w2.T
    b1 = np.array([b1])
    b2 = np.array([b2])
    wd = np.hstack((b1,w1,b2,w2))
    print("wdtwin:"+str(wd))
    return wd,clf
    
def addneuron_A(X, y, W, C1, C3, C4, delta, alpha, curr_Neuron):
    n = X.shape[1]
    print('A')
    m = len(C1) + len(C3) + len(C4)
#     print(m)
    print("C1:")
    print((C1))
    print("C3:")
    print((C3))
    print("C4:")
    print((C4))
    y_A = np.zeros((m))
    X_A = np.zeros((m, n))
    for i in range(len(C1)):
        X_A[i] = X[C1[i]]
        y_A[i] = 0

    for i in range(len(C3)):
        X_A[i + len(C1)] = X[C3[i]]
        y_A[i + len(C1)] = 1

    for i in range(len(C4)):
        X_A[i + len(C1)+ len(C3)] = X[C4[i]]
        y_A[i + len(C1)+ len(C3)] = 0
    print("X_A:")
    print(X_A)
    print("y_A:")
    print(y_A)
    # print("X:"+str(X))
    # print("C3:"+str(C3))
    print("W:"+str(W))
    Wa = find_Wa(X, C3, W)
    print("Wa:"+str(Wa))
#         print(Wa)     
    W_A,clf = optimize(X_A, y_A, delta, alpha)
    y_res = y_hat(X_A, W_A, delta, y_A,0,clf)
    Cc1, Cc2, Cc3, Cc4 = classes(y_A, y_res)
    twin_svm_enable = int(len(Cc1)==0 or len(Cc2)==0)
    print("twin_svm_enable:"+str(twin_svm_enable))
    if(twin_svm_enable):
        W_A,clf = twin(X_A,y_A)
    curr_Neuron.insert('A', Wa)
    curr_Neuron.A.X = W_A
    curr_Neuron.A.twin_node = twin_svm_enable
    curr_Neuron.A.twin_classifier = clf
    return X_A, y_A, W_A,twin_svm_enable,clf


def addneuron_B(X, y, W, C2, C3, C4, delta, alpha, curr_Neuron):
    n = X.shape[1]
    print('B')
    m = len(C2) + len(C3) + len(C4)
    # print(m)
    y_B = np.zeros((m))
    X_B = np.zeros((m, n))
#         y_B = []
    print("C2:")
    print((C2))
    print("C3:")
    print((C3))
    print("C4:")
    print((C4))
    for i in range(len(C2)):
        X_B[i] = X[C2[i]]
        y_B[i] = 0

    for i in range(len(C3)):
        X_B[i + len(C2)] = X[C3[i]]
        y_B[i + len(C2)] = 0

    for i in range(len(C4)):
        X_B[i + len(C2)+ len(C3)] = X[C4[i]]
        y_B[i + len(C2)+ len(C3)] = 1

#     print(X_B)
    print("X_B:")
    print(X_B)
    print("y_B:")
    print(y_B)
    # print("X:"+str(X))
    # print("C4:"+str(C4))
    print("W:"+str(W))
    Wb = find_Wb(X, C4, W)
    print("Wb:"+str(Wb))
    W_B,clf = optimize(X_B, y_B, delta, alpha)
    y_res = y_hat(X_B, W_B, delta, y_B,0,clf)
    Cc1, Cc2, Cc3, Cc4 = classes(y_B, y_res)
    twin_svm_enable = int(len(Cc1)==0 or len(Cc2)==0)
    print("twin_svm_enable:"+str(twin_svm_enable))
    if(twin_svm_enable):
        W_B,clf = twin(X_B,y_B)
    curr_Neuron.insert('B', Wb)
    curr_Neuron.B.X = W_B
    curr_Neuron.B.twin_node = twin_svm_enable
    curr_Neuron.B.twin_classifier = clf
    return X_B, y_B, W_B,twin_svm_enable,clf


def classifier(X, y, W, delta, alpha, curr_Neuron,twin_en,clff):
    print("y:")
    print(y)
    y_res = y_hat(X, W, delta, y,twin_en,clff)
    print("y_res:")
    print(y_res)
    C1, C2, C3, C4 = classes(y, y_res)
    print("C1len:"+str(len(C1)))
    print("C2len:"+str(len(C2)))
    print("twin_en:"+str(twin_en))
    if len(C3)!= 0:
        X_A, y_A, W_A,twin_enable,clf = addneuron_A(X, y, W, C1, C3, C4, delta, alpha, curr_Neuron)
        classifier(X_A, y_A, W_A, delta, alpha, curr_Neuron.A,twin_enable,clf)
    if len(C4)!= 0:
        X_B, y_B, W_B,twin_enable,clf = addneuron_B(X, y, W, C2, C3, C4, delta, alpha, curr_Neuron)
        classifier(X_B, y_B, W_B, delta, alpha, curr_Neuron.B,twin_enable,clf)
    else:
        print("terminated")
        return 'Terminated'

X = np.array([[1,1,8],[1,1,1],[1,4,5],[1,4,4],[1,6,5],[1,6,4],[1,10,8],[1,10,1],[1,2,6],[1,2,3],[1,8,6],[1,8,3]])
y=np.array([1,1,1,1,1,1,1,1,0,0,0,0])

alpha = 0
delta = 1
curr_Neuron = Neuron(3,0,None)
W,clf1 = optimize(X, y, delta, alpha)
curr_Neuron.X = W

classifier(X, y, W, delta, alpha, curr_Neuron,0,clf1)
print("###Verification###")
print(curr_Neuron.calculate(X))
