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
from biot import *
from data import *
from FD_biot import *
from scipy.interpolate import UnivariateSpline





class BC(object):
    
    def getBC(self,data_formatted,smooth_lbc,smooth_rbc,smooth_ic):
        lbc_dat = data_formatted[data_formatted[:,1]==np.min(data_formatted[:,1])]
        lbc_dat = lbc_dat[lbc_dat[:,0].argsort()]
        rbc_dat = data_formatted[data_formatted[:,1]==np.max(data_formatted[:,1])]
        rbc_dat = rbc_dat[rbc_dat[:,0].argsort()]
        ic_dat  = data_formatted[data_formatted[:,0]<np.min(data_formatted[:,0])+60]
        ic_dat = ic_dat[ic_dat[:,1].argsort()]
        lbc_spline = UnivariateSpline(lbc_dat[:,0],lbc_dat[:,2])
        rbc_spline = UnivariateSpline(rbc_dat[:,0],rbc_dat[:,2])        
        ic_spline  = UnivariateSpline(ic_dat[:,1],ic_dat[:,2])
        lbc_spline.set_smoothing_factor(1000000)
        rbc_spline.set_smoothing_factor(1000000)
        ic_spline.set_smoothing_factor(1000000)
        lbc_spline.set_smoothing_factor(smooth_lbc)
        rbc_spline.set_smoothing_factor(smooth_rbc)
        ic_spline.set_smoothing_factor(smooth_ic)

        tlim = [np.min(lbc_dat[:,0]),np.max(lbc_dat[:,0])]
        xlim = [np.min(ic_dat[:,1]),np.max(ic_dat[:,1])]

        ic_left = ic_spline(xlim[0])
        ic_right = ic_spline(xlim[1])

        lbc_initial = lbc_spline(tlim[0])
        rbc_initial = rbc_spline(tlim[0])

        def lbc_adj(t):
            if ic_left>lbc_initial:
                y = np.maximum(0.1*(ic_left-lbc_initial)*(0.1*tlim[1] + 0.9*tlim[0] - t)/(tlim[1]*0.1 - tlim[0]*0.1),np.zeros_like(t))
            else:
                y = np.minimum(0.1*(ic_left-lbc_initial)*(0.1*tlim[1] + 0.9*tlim[0] - t)/(tlim[1]*0.1 - tlim[0]*0.1),np.zeros_like(t))
            return y
            
        def rbc_adj(t):
            if ic_right>rbc_initial:
                y = np.maximum(0.1*(ic_right-rbc_initial)*(0.1*tlim[1] + 0.9*tlim[0] - t)/(tlim[1]*0.1 - tlim[0]*0.1),np.zeros_like(t))
            else:
                y = np.minimum(0.1*(ic_right-rbc_initial)*(0.1*tlim[1] + 0.9*tlim[0] - t)/(tlim[1]*0.1 - tlim[0]*0.1),np.zeros_like(t))
            return y

        def ic_adj(x):
            if ic_right>rbc_initial:
                y = -np.maximum(0.9*(ic_right-rbc_initial)*((0.1*xlim[0] + 0.9*xlim[1] - x)/(xlim[0]*0.1 - xlim[1]*0.1)),np.zeros_like(x))
            else:
                y = -np.minimum(0.9*(ic_right-rbc_initial)*((0.1*xlim[0] + 0.9*xlim[1] - x)/(xlim[0]*0.1 - xlim[1]*0.1)),np.zeros_like(x))
            if ic_left>lbc_initial:
                y = y - np.maximum(0.9*(ic_left-lbc_initial)*((0.1*xlim[1] + 0.9*xlim[0] - x)/(xlim[1]*0.1 - xlim[0]*0.1)),np.zeros_like(x))
            else:
                y = y - np.minimum(0.9*(ic_left-lbc_initial)*((0.1*xlim[1] + 0.9*xlim[0] - x)/(xlim[1]*0.1 - xlim[0]*0.1)),np.zeros_like(x))
            return y


        def lbc(t,return_domain=False):
            if return_domain==False:
                return lbc_spline(t) + lbc_adj(t)
            else:
                return [np.min(lbc_spline.get_knots()),np.max(lbc_spline.get_knots())]
                

        def rbc(t,return_domain=False):
            if return_domain==False:
                return rbc_spline(t) + rbc_adj(t)        
            else:
                return [np.min(rbc_spline.get_knots()),np.max(rbc_spline.get_knots())]
        
        def ic(x,return_domain=False):
            if return_domain==False:
                return ic_spline(x) + ic_adj(x)
            else:
                return [np.min(ic_spline.get_knots()),np.max(ic_spline.get_knots())]
        
        return lbc,rbc,ic
    
    def plotBC(self,lbc,rbc,ic,data_formatted,bound_dat=False):
        lbc_dat = data_formatted[data_formatted[:,1]==np.min(data_formatted[:,1])]
        lbc_dat = lbc_dat[lbc_dat[:,0].argsort()]
        rbc_dat = data_formatted[data_formatted[:,1]==np.max(data_formatted[:,1])]
        rbc_dat = rbc_dat[rbc_dat[:,0].argsort()]
        ic_dat  = data_formatted[data_formatted[:,0]<np.min(data_formatted[:,0])+60]
        ic_dat = ic_dat[ic_dat[:,1].argsort()]
        xx = np.linspace(np.min(ic_dat[:,1]),np.max(ic_dat[:,1]),500)
        
        
        fig, axs = plt.subplots(3, 1)

        tt = np.linspace(np.min(lbc_dat[:,0]),np.max(lbc_dat[:,0]),500)
        if bound_dat==True:
            im = axs[0].scatter(lbc_dat[:,0],lbc_dat[:,2],alpha=0.4)
            im = axs[1].scatter(rbc_dat[:,0],rbc_dat[:,2],alpha=0.4)
            im = axs[2].scatter(ic_dat[:,1],ic_dat[:,2])

        im = axs[0].title.set_text('LBC')
        im = axs[0].plot(tt,lbc(tt),'r')
        
        im = axs[1].title.set_text('RBC')
        im = axs[1].plot(tt,rbc(tt),'r')
        
        im = axs[2].title.set_text('IC')
        im = axs[2].plot(xx,ic(xx),'r')

        plt.show()
