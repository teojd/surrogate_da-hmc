import numpy as np
import tensorflow as tf
import scipy as sp
import time as time
import scipy.io
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import kv 
from scipy.special import gamma as gamma_fun
from scipy.spatial.distance import pdist, cdist, squareform
import matplotlib.pyplot as plt


class biot(object):
    def __init__(self,n_t,n_x):
        self.n_t = n_t
        self.n_x = n_x
    
    def matern(self,dists,sigma,nu,rho_x,full=True):
        dists=dists/rho_x
        if nu == 0.5:
            K = np.exp(-dists)
        elif nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1. + K) * np.exp(-K)
        elif nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1. + K + K ** 2 / 3.0) * np.exp(-K)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = (math.sqrt(2 * nu) * K)
            K.fill((2 ** (1. - nu)) / gamma_fun(nu))
            K *= tmp ** nu
            K *= kv(nu, tmp)
        if full == False:
            K = squareform(K)
            np.fill_diagonal(K, 1)
        return sigma*K
    
            
    def rbf(self,dists,sigma,rho):
        K = np.exp(-dists**2/(2*rho**2))
        return sigma*K
    
    def generate_biot(self,kernel_t='rbf',kernel_x='rbf',sigma=20,rho_t=1200,rho_x=0.25,nu_t=0.5,nu_x=2.5,tlim=[0.,3600.],xlim=[0.3,1.]):
        t_pts = np.linspace(tlim[0],tlim[1],self.n_t).reshape([-1,1])
        x_pts = np.linspace(xlim[0],xlim[1],self.n_x).reshape([-1,1])

        xx_gp, tt_gp = np.meshgrid(x_pts,t_pts)
        xx_gp = xx_gp.reshape([-1,1])
        tt_gp = tt_gp.reshape([-1,1])

        dist_x = pdist(xx_gp)
        dist_t = pdist(tt_gp)

        if kernel_x =='matern':
            cov_x = self.matern(dist_x,sigma,nu_x,rho_x)
        else:
            cov_x = self.rbf(dist_x,sigma,rho_x)
        
    
        if kernel_t =='matern':
            cov_t = self.matern(dist_t,sigma,nu_t,rho_t)
        else:
            cov_t = self.rbf(dist_t,sigma,rho_t)
    
    
        cov = squareform(cov_t*cov_x)
        np.fill_diagonal(cov, sigma**2)
        y = np.random.multivariate_normal(np.zeros(self.n_t*self.n_x),cov,1).reshape([self.n_t,self.n_x])+3*sigma
        
        return tt_gp.reshape([self.n_t,self.n_x]),xx_gp.reshape([self.n_t,self.n_x]), y
    
    
    def plot(self,t,x,y):
        fig, axs = plt.subplots(1)
        axs.pcolormesh(t, x, y.reshape([self.n_x,self.n_t]), cmap=cm.coolwarm)
        
        
        
