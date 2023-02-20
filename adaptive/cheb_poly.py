import numpy as np
import tensorflow as tf
import time as time
import scipy.io

class cheb_poly(object):
    
    def __init__(self,lims):
        self.lims = lims

    
    def P(self,X,n,tensor=True):
        lims = self.lims
        X = 2*(X-lims[0])/(lims[1]-lims[0])-1
        if n==0:
            if tensor == True:
                T = tf.ones_like(X)
            else:
                T = np.ones_like(X)
        elif n==1:
            T = X
        elif n==2:
            T = 2*X**2-1
        elif n==3:
            T = 4*X**3 - 3*X
        elif n==4:
            T = 8*X**4 - 8*X**2 + 1
        elif n==5:
            T = 16*X**5 - 20*X**3 + 5*X
        elif n==6:
            T = 32*X**6 - 48*X**4 + 18*X**2 - 1        
        elif n==7:
            T = 64*X**7 - 112*X**5 + 56*X**3 - 7*X
        elif n==8:
            T = 128*X**8 - 256*X**6 + 160*X**4 - 32*X**2 + 1        
        elif n==9:
            T = 256*X**9 - 576*X**7 + 432*X**5 - 120*X**3 + 9*X        
        elif n==10:
            T = 512*X**10 - 1280*X**8 + 1120*X**6 - 400*X**4 + 50*X**2 - 1        
        elif n==11:
            T = 1024*X**11 - 2816*X**9 + 2816*X**7 - 1232*X**5 + 220*X**3 - 11*X        
        return T


    def U(self,X,n,tensor=True):
        lims = self.lims
        X = 2*(X-lims[0])/(lims[1]-lims[0])-1
        if n==0:
            if tensor == True:
                T = tf.ones_like(X)
            else:
                T = np.ones_like(X)
        elif n==1:
            T = 2*X
        elif n==2:
            T = 4*X**2 - 1
        elif n==3:
            T = 8*X**3 - 4*X
        elif n==4:
            T = 16*X**4 - 12*X**2 + 1
        elif n==5:
            T = 32*X**5 - 32*X**3 + 6*X
        elif n==6:
            T = 64*X**6 - 80*X**4 + 24*X**2 - 1        
        elif n==7:
            T = 128*X**7 - 192*X**5 + 80*X**3 - 8*X
        elif n==8:
            T = 256*X**8 - 448*X**6 + 240*X**4 - 40*X**2 + 1        
        elif n==9:
            T = 512*X**9 - 1024*X**7 + 672*X**5 - 160*X**3 + 10*X        
        elif n==10:
            T = 512*X**10 - 1280*X**8 + 1120*X**6 - 400*X**4 + 50*X**2 - 1        
        elif n==11:
            T = 1024*X**11 - 2816*X**9 + 2816*X**7 - 1232*X**5 + 220*X**3 - 11*X        
        return T
