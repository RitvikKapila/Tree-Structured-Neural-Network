# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:53:29 2020

@author: Harini
"""
import numpy as np
from scipy.optimize import linprog 
from twinsvm import twinsvmclassifier

params2 = {'Epsilon1': 0.1, 'Epsilon2': 0.1, 'C1': 1, 'C2': 1,'kernel_type':2,'kernel_param': 2}
clf=twinsvmclassifier(**params2)


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
            # print("V:"+str(V))
            A = np.vstack((A,V)) 
            # print("A:"+str(A))
            for j in range(n):
                A[j][i] = X[i][j]
                A[j+n][i] = -1*X[i][j]
                B[i] = delta
    A = np.vstack((A, np.zeros((n,m))))   # dimension of A is 2n+2m+n, m
    # print("A:"+str(A))
    s1 = np.zeros((3*n+2*m))
    s2 = np.zeros((3*n+2*m))
    # print("s1:"+str(s1))
    # print("s2:"+str(s2))
    W = np.zeros((2*n+2*m+n))
    # print("W:"+str(W))
    for i in range(m):
        if y[i] == 0:
            W[2*i+2*n] = 0                  #
            W[2*i+2*n+1] = 1/K              #
            s2[2*n+2*i] = -1
        else:
            W[2*i+2*n] = 0                  #
            W[2*i+2*n+1] = 1/M              #
            s1[2*n+2*i] = -1
    # print("W:"+str(W))
    # print("s1:"+str(s1))
    # print("s2:"+str(s2))
    for i in range(n):
        W[i+2*n+2*m] = alpha   ##########
    # print("W:"+str(W))    
    A = A.T
    # print("A:"+str(A))
    #Ax=B the equality contraints
    I = np.eye(n,n)
    # print("I:"+str(I))
    C = np.zeros((2*n, 3*n+2*m))
    D = np.zeros((2*n+2))
    # print("C:"+str(C))
    # print("D:"+str(D))
    D[2*n+2-1] = -1
    D[2*n+2-2] = -1
    # print("D:"+str(D))
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
    return np.matrix(wd)


def y_hat(X, weights, delta, y,twin_svm_en):
    if(twin_svm_en == 0):
        n = len(X)
        y_res = np.asarray(np.dot(weights, X.T))
        for i in range(n):
    #         print(y[0][i])
            if y_res[0][i] >= delta:
                y_res[0][i] = 1
            elif y_res[0][i] <= -delta:
                y_res[0][i] = 0
            else :
                y_res[0][i] = 1-y[i]
        # print("y_res:"+str(y_res))
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

def find_Wa(X, C3, weights):
    res = 0
    y=0
    for i in C3:
        y = np.asarray(np.dot(weights, X[i].T))
        print("y[0][0]:"+str(y[0][0]))
        res = min(res, y[0][0])
#         print(res[0][0].shape)
    return -res+0.01

# res = classes(y, y_out)
# C3 = res[2]
# find_Wa(X, C3, wd)

def find_Wb(X, C4, weights):
    res = 0
    y=0
    for i in C4:
        y = np.asarray(np.dot(weights, X[i].T))
        print("y[0][0]:"+str(y[0][0]))
        res = max(res, y[0][0])
#         print(res[0][0].shape)
    return -res-0.01


# C4 = res[3]
# find_Wb(X, C4, wd)

class Neuron:
    
    def __init__(self,n):
        self.X = np.zeros((1,n))
        self.inp = n
        self.A = None
        self.B = None
        self.weight = 0
        
    def insert(self,neuron_type, weight):     #### Type is a restricted variable, Type was always set to A, weigth to 1
        if neuron_type == 'A':
            self.A = Neuron(self.inp)
#             print(self.X)
#             print( np.array([weight]) )
            print("self.X:"+str(self.X))
            self.X = np.hstack((self.X, np.array([[weight]])))
            print("$$$$$$$$$$")
            print("A:"+str(self.X))
            print("$$$$$$$$$$")
        else:
            self.B = Neuron(self.inp)
            print("self.X:"+str(self.X))
            self.X = np.hstack((self.X, np.array([[weight]])))
            print("$$$$$$$$$$")
            print("B:"+str(self.X))
            print("$$$$$$$$$$")
#     def calculate(self,X):
#         if self.A != None and self.B != None:
#             y_1 = calculate(self.A)
#             y_2 = calculate(self.B)
#             return np.sum(np.multiply(self.X, np.vstack((X, y_1, y_2))))
#         elif self.A != None:
#             y_1 = calculate(self.A)
#             return np.sum(np.multiply(self.X, np.vstack((X, y_1))))
#         elif self.B != None:
#             y_2 = calculate(self.B)
#             return np.sum(np.multiply(self.X, np.vstack((X, y_2))))
#         else:
#             return np.sum(np.multiply(self.X, X))
def twin(X,y):
    XX = X[:,1:len(X[0])]
    yy = y
    clf.fit(XX,yy)
    w1,b1,w2,b2 = clf.getwb()
    w1 = w1.T
    w2 = w2.T
    b1 = np.array([b1])
    b2 = np.array([b2])
    wd = np.hstack((w1,b1,w2,b2))
    print("wdtwin:"+str(wd))
    return wd
    
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
    print("X:"+str(X))
    print("C3:"+str(C3))
    print("W:"+str(W))
    Wa = find_Wa(X, C3, W)
    print("Wa:"+str(Wa))
#         print(Wa)     
    W_A = optimize(X_A, y_A, delta, alpha)
    y_res = y_hat(X_A, W_A, delta, y_A,0)
    Cc1, Cc2, Cc3, Cc4 = classes(y_A, y_res)
    twin_svm_enable = int(len(Cc1)==0 or len(Cc2)==0)
    print("twin_svm_enable:"+str(twin_svm_enable))
    if(twin_svm_enable):
        W_A = twin(X_A,y_A)
    curr_Neuron.insert('A', Wa)
    curr_Neuron.A.X = W_A
  
    return X_A, y_A, W_A,twin_svm_enable


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
    print("X:"+str(X))
    print("C4:"+str(C4))
    print("W:"+str(W))
    Wb = find_Wb(X, C4, W)
    print("Wb:"+str(Wb))
    W_B = optimize(X_B, y_B, delta, alpha)
    y_res = y_hat(X_B, W_B, delta, y_B,0)
    Cc1, Cc2, Cc3, Cc4 = classes(y_B, y_res)
    twin_svm_enable = int(len(Cc1)==0 or len(Cc2)==0)
    print("twin_svm_enable:"+str(twin_svm_enable))
    if(twin_svm_enable):
        W_B = twin(X_B,y_B)
    curr_Neuron.insert('B', Wb)
    curr_Neuron.B.X = W_B

    return X_B, y_B, W_B,twin_svm_enable


def classifier(X, y, W, delta, alpha, curr_Neuron,twin_en):
    print("y:")
    print(y)
    y_res = y_hat(X, W, delta, y,twin_en)
    print("y_res:")
    print(y_res)
    C1, C2, C3, C4 = classes(y, y_res)
    print("C1len:"+str(len(C1)))
    print("C2len:"+str(len(C2)))
    # twin_svm_enable = int(len(C1)==0 or len(C2)==0)
    print("twin_en:"+str(twin_en))
    if len(C3)!= 0:
        X_A, y_A, W_A,twin_en = addneuron_A(X, y, W, C1, C3, C4, delta, alpha, curr_Neuron)
        classifier(X_A, y_A, W_A, delta, alpha, curr_Neuron.A,twin_en)
    if len(C4)!= 0:
        X_B, y_B, W_B,twin_en = addneuron_B(X, y, W, C2, C3, C4, delta, alpha, curr_Neuron)
        classifier(X_B, y_B, W_B, delta, alpha, curr_Neuron.B,twin_en)
    else:
        return 'Terminated'

# X = np.array([[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
# y=np.array([1,0,0,1])
# X = np.array([[1,1,1],[1,4,4],[1,6,4],[1,10,1],[1,2,3],[1,8,3]])
# y=np.array([1,1,1,1,0,0])
X = np.array([[1,1,8],[1,1,1],[1,4,5],[1,4,4],[1,6,5],[1,6,4],[1,10,8],[1,10,1],[1,2,6],[1,2,3],[1,8,6],[1,8,3]])
y=np.array([1,1,1,1,1,1,1,1,0,0,0,0])
# X = np.array([[1,1,8],[1,4,5],[1,1,1],[1,4,4],[1,6,5],[1,6,4],[1,10,8],[1,8,1],[1,2,6],[1,2,3],[1,8,6],[1,8,3]])
# y=np.array([1,1,1,1,1,1,1,1,0,0,0,0])
alpha = 0
delta = 1
curr_Neuron = Neuron(3)
W = optimize(X, y, delta, alpha)
curr_Neuron.X = W


classifier(X, y, W, delta, alpha, curr_Neuron,0)
# print(curr_Neuron.calculate(X))
