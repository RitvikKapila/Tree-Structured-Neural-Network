# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:00:40 2020

@author: Harini
"""

import numpy as np
from numpy import linalg
from cvxopt import solvers,matrix

solvers.options['show_progress'] = False
def Twin_plane_1(R,S,C1,Epsi1,regulz1):
	StS = np.dot(S.T,S)
	StS = StS + regulz1*(np.identity(StS.shape[0]))
	StSRt = linalg.solve(StS,R.T)  
	RtStSRt = np.dot(R,StSRt) 
	RtStSRt = (RtStSRt+(RtStSRt.T))/2
	m2 = R.shape[0]
	e2 = -np.ones((m2,1))
	vlb = np.zeros((m2,1))
	vub = C1*(np.ones((m2,1)))
	# x<=vub
	# x>=vlb -> -x<=-vlb
	# cdx<=vcd
	cd = np.vstack((np.identity(m2),-np.identity(m2)))
	vcd = np.vstack((vub,-vlb))
	alpha = solvers.qp(matrix(RtStSRt,tc='d'),matrix(e2,tc='d'),matrix(cd,tc='d'),matrix(vcd,tc='d'))
	alphasol = np.array(alpha['x'])
	z = -np.dot(StSRt,alphasol)
	w1 = z[:len(z)-1]
	b1 = z[len(z)-1]
	return [w1,b1]