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
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from biot import biot
from FD_biot import FD_biot
from FD_flux import FD_flux


class data(object):
    
    def __init__(self):
        self.biot = biot(50,30)
        self.xlim = []
        self.tlim = []
    
    def read_raw(self,data_csv):
        data = np.array(pd.read_csv(data_csv,header=None))
        return data
    
    def reformat_for_model(self,raw_data,seconds):
        data_thinned = raw_data[1:-1:seconds].astype(np.float32)
        radial_locs = raw_data[0,1:].astype(np.float32)
        radial_locs = np.repeat(radial_locs, data_thinned.shape[0]).reshape([-1,1])
        data_long = data_thinned[:,0]
        data_long = np.tile(data_long,data_thinned.shape[1]-1).reshape([-1,1])
        data_long = np.hstack([data_long[:,0].reshape([-1,1]),radial_locs])
        temps = np.zeros_like(radial_locs)
        k=0
        for i in range(data_thinned.shape[1]-1):
            for j in range(data_thinned.shape[0]):
                temps[k] = data_thinned[j,i+1]
                k=k+1
        data_long = np.hstack([data_long,temps]).astype(np.float32)
        data_long = data_long[data_long[:,0]>0]
        
        return data_long

    def read_formatted(self,data_csv,seconds,x_multiplier = 1.):
        raw_data = self.read_raw(data_csv)
        formatted_data = self.reformat_for_model(raw_data,seconds)
        self.dat_scale = np.max(formatted_data[:,2])
        formatted_data[:,2] = formatted_data[:,2]/self.dat_scale
        formatted_data[:,1] = formatted_data[:,1]*x_multiplier
        self.tlim.append(np.min(formatted_data[:,0]))
        self.tlim.append(np.max(formatted_data[:,0]))
        self.xlim.append(np.min(formatted_data[:,1]))
        self.xlim.append(np.max(formatted_data[:,1]))
        return formatted_data
    
    
    def generate_data(self,dat_nt = 20, dat_nx = 10, FD_nx=301,FD_nt=301,biot_in=None,L = None,R = None, I = None, c1=12000.,c2=1.,c3=1.,tlim=[0.,3600.],xlim=[0.3,1.],std=20,rho_t=1200,rho_x=0.25,nu_t=1.5,nu_x=2.5,kernel_t='rbf',kernel_x='rbf',full=False,kind='flux'):
        # solution domain
        if I==None:
            self.xlim.append(xlim[0])
            self.xlim.append(xlim[1])
            self.tlim.append(tlim[0])
            self.tlim.append(tlim[1])
        else:
            self.xlim.append(I(0,return_domain=True)[0])
            self.xlim.append(I(0,return_domain=True)[1])
            self.tlim.append(L(0,return_domain=True)[0])
            self.tlim.append(L(0,return_domain=True)[1])
            
        self.xlim = self.xlim[:2]
        self.tlim = self.tlim[:2]

        self.dat_scale = 1.

        x_pts = np.linspace(self.xlim[0],self.xlim[1],FD_nx)
        t_pts = np.linspace(self.tlim[0],self.tlim[1],FD_nt)

        if biot_in==None:
            tt_gp,xx_gp,bi_gp = self.biot.generate_biot(sigma=std,rho_t=rho_t,rho_x=rho_x,nu_t=nu_t,nu_x=nu_x,tlim=tlim,xlim=xlim,kernel_t=kernel_t,kernel_x=kernel_x)
            
        xx, tt = np.meshgrid(x_pts,t_pts)
        biot_out = griddata(np.hstack([xx_gp.reshape([-1,1]),tt_gp.reshape([-1,1])]), bi_gp.reshape([-1,1]), (xx, tt), method='linear')
        
        biot_out = (biot_out.reshape([FD_nt,FD_nx]))
        biot_out = np.maximum(biot_out,np.zeros_like(biot_out))
        #biot_out = np.minimum(biot_out,200*np.ones_like(biot_out))
        
        
        if L==None:
            def L(t):
                return np.ones_like(t)*self.xlim[0]
        if R==None:
            def R(t):
                return np.ones_like(t)*self.xlim[1]
        if I==None:
            def I(x):
                return x
        
        
        if kind=='biot':
            FD_true = FD_biot(self.xlim[0],self.xlim[1],self.tlim[1],L,R,I,FD_nx-1,FD_nt-1,0.5,cheb=False,bi=biot_out,c1=c1,c2=c2,c3=c3,T0=tlim[0])
        else:
            FD_true = FD_flux(self.xlim[0],self.xlim[1],self.tlim[1],L,R,I,FD_nx-1,FD_nt-1,0.5,cheb=False,bi=biot_out,c1=c1,c2=c2,c3=c3,T0=tlim[0])
        
        U = FD_true.solve()
        
        n_xdat = dat_nx #+ 2
        n_tdat = dat_nt #+ 1
                
        fig, axs = plt.subplots(2,1)
        im=axs[1].pcolormesh(tt, xx, U.T, cmap=cm.coolwarm)
        plt.colorbar(im,ax=axs[1])
        
        im = axs[0].pcolormesh(tt, xx, biot_out, cmap=cm.coolwarm)
        plt.colorbar(im,ax=axs[0])
        
        inds_t = np.linspace(0,FD_nt-1,n_tdat).astype(int)
        inds_x = np.linspace(0,FD_nx-1,n_xdat).astype(int)
        inds_t,inds_x = np.meshgrid(inds_t,inds_x)
        t_dat = tt[inds_t,inds_x].reshape([-1,1])
        x_dat = xx[inds_t,inds_x].reshape([-1,1])
        z_dat = (U.T)[inds_t,inds_x].reshape([-1,1])
        
        x_dat = x_dat[t_dat>self.tlim[0]]
        z_dat = z_dat[t_dat>self.tlim[0]]
        t_dat = t_dat[t_dat>self.tlim[0]]
        
        t_dat = t_dat[(x_dat<self.xlim[1]).astype(int)+(x_dat>self.xlim[0]).astype(int)==2].reshape([-1,1])
        z_dat = z_dat[(x_dat<self.xlim[1]).astype(int)+(x_dat>self.xlim[0]).astype(int)==2].reshape([-1,1])
        x_dat = x_dat[(x_dat<self.xlim[1]).astype(int)+(x_dat>self.xlim[0]).astype(int)==2].reshape([-1,1])
        
        
        n_dat = z_dat.size
        
        
        #plt.scatter(t_dat,x_dat,c=z_dat,cmap=cm.coolwarm,vmax = np.max(U),vmin = np.min(U))
        axs[1].scatter(t_dat,x_dat)
        
        z_dat = (z_dat + np.random.normal(loc=0,scale=0.04,size=[n_dat,1])).astype(np.float32)
        if full == True:
            return np.hstack([t_dat, x_dat, z_dat]), U.T, biot_out,[tt,xx]
        else:
            return np.hstack([t_dat, x_dat, z_dat])