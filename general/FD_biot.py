import numpy as np
import scipy as sp
import tensorflow as tf
import time as time
import scipy.io
from cheb_poly import *

class FD_biot(object):
    
    def __init__(self,LB,RB,T,LBC,RBC,IC,n_x,n_t,theta,indices_t=0,indices_x=0,cheb=True,c1=20.,c2=1.,c3=1.,bi = 0,T0=0.):
        self.xmesh = np.linspace(LB,RB,n_x+1)
        self.tmesh = np.linspace(T0,T,n_t+1)
        self.LBC  = LBC(self.tmesh)
        self.RBC  = RBC(self.tmesh)
        self.IC  = IC(self.xmesh)
        self.Dx   = (RB-LB)/n_x
        self.Dt   = T/n_t
        self.n_x  = n_x
        self.n_t  = n_t
        self.thet = theta
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.chebx = cheb_poly([LB,RB])
        self.chebt = cheb_poly([0,T])
        self.eta = c2*self.Dt/(self.Dx**2);
        self.alph  = c3*self.Dt/self.Dx;
        self.indices_t = indices_t
        self.indices_x = indices_x
        self.cheb = cheb
        self.bi = bi
        alpha = self.alph
        eta = self.eta
        xmesh = self.xmesh
        A = np.zeros([n_t+1,3,n_x+1])
        B = np.zeros([n_t+1,3,n_x+1])
        for i in range(1,n_x):
            A[:,0,i+1] = theta*(-eta - (alpha/(2*xmesh[i])))
            A[:,2,i-1] = theta*(-eta + (alpha/(2*xmesh[i])))
            B[:,0,i+1] = (1-theta)*(eta + (alpha/(2*xmesh[i])))
            B[:,2,i-1] = (1-theta)*(eta - (alpha/(2*xmesh[i])))        
        A[:,1,n_x] = 1
        A[:,1,0] = 1
        B[:,1,n_x] = 1
        B[:,1,0] = 1
        self.A = A
        self.B = B

    def build_mats(self,coefs):
        chebx = self.chebx
        chebt = self.chebt
        indices_x = self.indices_x
        indices_t = self.indices_t
        xmesh = self.xmesh
        tmesh = self.tmesh
        n_x   = self.n_x
        thet  = self.thet
        eta   = self.eta
        alph  = self.alph
        Dt = self.Dt
        if self.cheb == True:
            self.bi = sum(coefs[j]*chebx.P(xmesh,indices_x[j],tensor=False).reshape([1,-1])*chebt.P(tmesh,indices_t[j],tensor=False).reshape([-1,1]) for j in range(np.array(coefs).shape[0])) 
#            Biot = np.maximum(Biot,np.zeros_like(Biot))
#            Biot = np.minimum(Biot,200*np.ones_like(Biot))
        Biot = self.bi
        A = self.A
        B = self.B
        c1 = self.c1
        c2 = self.c2
        c3 = self.c3
        for j in range(1,n_x):
            A[:,1,j]   = c1 + thet*(2*eta + Dt*Biot[:,j+1])
            B[:,1,j]   = c1 + (1-thet)*(-2*eta - Dt*Biot[:,j])
        return A,B   

    def solve(self, coefs=0):
        U = self.IC.reshape([-1,1])
        Ui = self.IC
        LBC = self.LBC
        RBC = self.RBC
        A,B = self.build_mats(coefs)
        tridiag_mult = self.tridiag_mult
        for i in range(self.n_t):
            b = tridiag_mult(B[i],Ui.reshape([-1,1])).flatten()
#            b = np.matmul(B[i],Ui.reshape([-1,1])).flatten()
            b[0]= LBC[i];
            b[-1] = RBC[i];
            Ui = sp.linalg.solve_banded((1,1),A[i],b).reshape([-1,1]);
#            Ui = self.thomas();
            U = np.hstack([U,Ui]);

            
        return U
    
    def tridiag_mult(self,B, Ui):
        n = Ui.shape[0]
        x = np.zeros([n,1])
        x[0] = Ui[0]
        x[-1] = Ui[-1]
        x[1:-1,0] = B[0,2:]*Ui[2:,0] + B[1,1:-1]*Ui[1:-1,0] + B[2,:-2]*Ui[:-2,0]
#        for i in range(1,n-1):
#            x[i] = B[i,i-1]*Ui[i-1] + B[i,i]*Ui[i] + B[i,i+1]*Ui[i+1]
        
        return x
    
    
    
    
