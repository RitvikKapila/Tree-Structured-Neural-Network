# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:00:08 2020

@author: Harini
"""

import numpy as np
from sklearn import preprocessing
import Kernel 
import plane1
import plane2

class twinsvmclassifier():
    def __init__(self,Epsilon1=0.1, Epsilon2=0.1, C1=1, C2=1,kernel_type=0,kernel_param=1,regulz1=1, regulz2=1,_estimator_type="classifier"):
        self.Epsilon1=Epsilon1
        self.Epsilon2=Epsilon2
        self.C1=C1
        self.C2=C2
        self.regulz1 = regulz1
        self.regulz2 = regulz2
        self.kernel_type=kernel_type
        self.kernel_param=kernel_param
        
    def fit(self, X, Y):
        assert (type(self.Epsilon1) in [float,int])
        assert (type(self.Epsilon2) in [float,int])
        assert (type(self.C1) in [float,int])
        assert (type(self.C2) in [float,int])
        assert (type(self.regulz1) in [float,int])
        assert (type(self.regulz2) in [float,int])
        assert (type(self.kernel_param) in [float,int])
        assert (self.kernel_type in [0,1,2])
        Data = sorted(zip(Y,X), key=lambda pair: pair[0], reverse = True)
        Total_Data = np.array([np.array(x) for y,x in Data])
        A=np.array([np.array(x) for y,x in Data if (y==1)])
        B=np.array([np.array(x) for y,x in Data if (y==0)])
        # print("Total_Data:"+str(Total_Data))
        # print("A:"+str(A))
        # print("B:"+str(B))
        m1 = A.shape[0]
        m2 = B.shape[0]
        e1 = -np.ones((m1,1))
        e2 = -np.ones((m2,1))
        if(self.kernel_type==0):
            S = np.hstack((A,-e1))
            R = np.hstack((B,-e2))
        else:
            S = np.zeros((A.shape[0],Total_Data.shape[0]))
            for i in range(A.shape[0]):
                for j in range(Total_Data.shape[0]):
                    S[i][j] = Kernel.kernelfunction(self.kernel_type,A[i],Total_Data[j],self.kernel_param)
            S = np.hstack((S,-e1))
            R = np.zeros((B.shape[0],Total_Data.shape[0]))
            for i in range(B.shape[0]):
                for j in range(Total_Data.shape[0]):
                    R[i][j] = Kernel.kernelfunction(self.kernel_type,B[i],Total_Data[j],self.kernel_param)
            R = np.hstack((R,-e2))
        [w1,b1] = plane1.Twin_plane_1(R,S,self.C1,self.Epsilon1,self.regulz1)
        [w2,b2] = plane2.Twin_plane_2(S,R,self.C2,self.Epsilon2,self.regulz2)
        self.plane1_coeff_ = w1
        self.plane1_offset_ = b1
        self.plane2_coeff_ = w2
        self.plane2_offset_ = b2
        self.data_ = Total_Data
        self.A_ = A
        self.B_ = B
        # print("w1:"+str(self.plane1_coeff_))
        # print("b1:"+str(self.plane1_offset_))
        # print("w2:"+str(self.plane2_coeff_))
        # print("b2:"+str(self.plane2_offset_))
        # print("self:"+str(self))
        return self

    def getwb(self):
        return self.plane1_coeff_,self.plane1_offset_,self.plane2_coeff_,self.plane2_offset_
    
    def get_params(self, deep=True):
        return {"Epsilon1": self.Epsilon1, "Epsilon2": self.Epsilon2, "C1": self.C1, "C2": self.C2, "regulz1": self.regulz1,
                "regulz2":self.regulz2, "kernel_type": self.kernel_type, "kernel_param": self.kernel_param}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

    def predict(self, X, y=None):
        if(self.kernel_type==0):
            S = X
            w1mod = np.linalg.norm(self.plane1_coeff_)
            w2mod = np.linalg.norm(self.plane2_coeff_)
        else:
            S = np.zeros((self.data_.shape[0],self.data_.shape[0]))
            for i in range(self.data_.shape[0]):
                for j in range(self.data_.shape[0]):
                    S[i][j] = Kernel.kernelfunction(self.kernel_type,self.data_[i],self.data_[j],self.kernel_param)
            w1mod = np.sqrt(np.dot(np.dot(self.plane1_coeff_.T,S),self.plane1_coeff_))
            w2mod = np.sqrt(np.dot(np.dot(self.plane2_coeff_.T,S),self.plane2_coeff_))
            S = np.zeros((X.shape[0],self.data_.shape[0]))
            for i in range(X.shape[0]):
                for j in range(self.data_.shape[0]):
                    S[i][j] = Kernel.kernelfunction(self.kernel_type,X[i],self.data_[j],self.kernel_param)
        y1 = np.dot(S,self.plane1_coeff_)+ ((self.plane1_offset_)*(np.ones((X.shape[0],1))))
        y2 = np.dot(S,self.plane2_coeff_)+ ((self.plane2_offset_)*(np.ones((X.shape[0],1))))
        yPredicted=np.zeros((X.shape[0],1))
        distFromPlane1 = y1/w1mod 
        distFromPlane2 = y2/w2mod 
        for i in range(len(distFromPlane1)):
            if (distFromPlane1[i]<distFromPlane2[i]):
                yPredicted[i][0]=0;
            else:
                yPredicted[i][0]=1;

        return yPredicted.transpose()[0]    
