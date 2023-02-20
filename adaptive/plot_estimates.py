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
import tensorflow_probability as tfp
from scipy.interpolate import griddata
import arviz as az
tfd = tfp.distributions
from matplotlib.gridspec import GridSpec

from BC import BC
from data import data
from biot import biot 
from dense_net import dense_net
from cheb_poly import cheb_poly
from FD_biot import FD_biot
from FD_flux import FD_flux

class plot_estimates(object):
    def __init__(self):
        pass
        
    def plot_map_function(self,res_t=500,res_x=200,uncertainty=True):
        nn_res = [300,150]
    
        self.map_pars_untransformed = self.sess.run(self.pars_tf[:-1])
        map_pars = self.transform_pars(self.map_pars_untransformed)
    
    
        
        map_hess_untransformed = self.sess.run(tf.hessians(-self.loss_map, self.pars_tf)[0])[:-1,:-1]
        post_cov = np.linalg.inv(-map_hess_untransformed)
        self.inv_neg_hess_untransformed = post_cov
        
    
        x_pts = np.linspace(self.xlim[0],self.xlim[1],res_x)
        t_pts = np.linspace(self.tlim[0],self.tlim[1],res_t)
        xx, tt = np.meshgrid(x_pts,t_pts)
        
        x_pts = np.linspace(self.xlim[0],self.xlim[1],nn_res[1])
        t_pts = np.linspace(self.tlim[0],self.tlim[1],nn_res[0])
        xx_nn, tt_nn = np.meshgrid(x_pts,t_pts)
    
        
        cheb_tt_nn = np.zeros([66,nn_res[0],nn_res[1]])
        cheb_xx_nn = np.zeros([66,nn_res[0],nn_res[1]])
    
        for i in range(66):
            cheb_tt_nn[i] = self.chebt.P(tt_nn,self.indices_t[i],tensor=False)
            cheb_xx_nn[i] = self.chebx.P(xx_nn,self.indices_x[i],tensor=False)
    
        
        map_Bi = self.flux_scale*sum(map_pars[i].reshape([-1,1,1])*np.reshape(cheb_xx_nn[i]*cheb_tt_nn[i],[1,nn_res[0],nn_res[1]]) for i in range(self.indices_t.shape[0]))[0]
    
        if self.positive==True:
            map_Bi = np.maximum(map_Bi,np.zeros_like(map_Bi))
    
            
        if self.kind=='exp biot' or self.kind=='exp flux':
            NN_solution = self.sess.run(tf.exp(self.u_eval(tt_nn, xx_nn, map_pars*np.ones([tt_nn.size,1])))).reshape(nn_res)
        else:
            NN_solution = self.sess.run(self.u_eval(tt_nn, xx_nn, map_pars*np.ones([tt_nn.size,1]))).reshape(nn_res)
    
    
    
        map_Bi_var = np.zeros(nn_res)
            
        if self.csv_name == None:
            fig, axs = plt.subplots(3, 1)
    
            if uncertainty==True:
                for i in range(66):
                    for j in range(66):
                            map_Bi_var = map_Bi_var + (post_cov[i,j]/(self.scale[i]*self.scale[j]))*cheb_tt_nn[i]*cheb_xx_nn[i]*cheb_tt_nn[j]*cheb_xx_nn[j]
                if self.kind=='exp biot' or self.kind=='direct biot':          
                    post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)*NN_solution
                else:
                    post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)
                miv = np.min(post_Bi_sd)
                mav = np.max(post_Bi_sd)
                im = axs[2].pcolormesh(tt_nn, xx_nn, post_Bi_sd, cmap=cm.coolwarm, vmin=miv, vmax=mav)
                fig.colorbar(im, ax=axs[2])    
    
    
            if self.kind=='exp biot' or self.kind=='direct biot':
                if self.csv_name == None:
                    mi = np.min([np.min(map_Bi*NN_solution),np.min(self.flux_scale*self.true_biot*self.true_sol)])
                    ma = np.max([np.max(map_Bi*NN_solution),np.max(self.flux_scale*self.true_biot*self.true_sol)])
                else:
                    mi = np.min(map_Bi*NN_solution)
                    ma = np.max(map_Bi*NN_solution)                    
            else:
                if self.csv_name == None:
                    mi = np.min([np.min(map_Bi),np.min(self.flux_scale*self.true_biot)])
                    ma = np.max([np.max(map_Bi),np.max(self.flux_scale*self.true_biot)])
                else:
                    mi = np.min(map_Bi)
                    ma = np.max(map_Bi)                    
                
    
    
            
            if self.csv_name==None:
                tt_biot = np.linspace(self.tlim[0],self.tlim[1],self.true_sol.shape[0])
                xx_biot = np.linspace(self.xlim[0],self.xlim[1],self.true_sol.shape[1])
                tt_biot,xx_biot = np.meshgrid(tt_biot,xx_biot)
                if self.kind=='exp biot' or self.kind=='direct biot':
                    im = axs[0].pcolormesh(tt_biot, xx_biot, self.flux_scale*(self.true_biot*self.true_sol).T, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                else:
                    im = axs[0].pcolormesh(tt_biot, xx_biot, self.flux_scale*(self.true_biot).T, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                    
    
            if self.kind=='exp biot' or self.kind=='direct biot':          
                im = axs[1].pcolormesh(tt_nn, xx_nn, map_Bi*NN_solution, cmap=cm.coolwarm, vmin=mi, vmax=ma)
            else:
                im = axs[1].pcolormesh(tt_nn, xx_nn, map_Bi, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                
            fig.colorbar(im, ax=[axs[0],axs[1]])        
        
            im = axs[0].title.set_text('True Flux')
            im = axs[1].title.set_text('MAP Flux')
            im = axs[2].title.set_text('Uncertainty')
    
    
        else:
            fig, axs = plt.subplots(2, 1)
            
            if uncertainty==True:
                for i in range(66):
                    for j in range(66):
                            map_Bi_var = map_Bi_var + (post_cov[i,j]/(self.scale[i]*self.scale[j]))*cheb_tt_nn[i]*cheb_xx_nn[i]*cheb_tt_nn[j]*cheb_xx_nn[j]
                if self.kind=='exp biot' or self.kind=='direct biot':          
                    post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)*NN_solution
                else:
                    post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)
                miv = np.min(post_Bi_sd)
                mav = np.max(post_Bi_sd)
                im = axs[1].pcolormesh(tt_nn, xx_nn, post_Bi_sd, cmap=cm.coolwarm, vmin=miv, vmax=mav)
                fig.colorbar(im, ax=axs[1])    
    
    
    
            if self.kind=='exp biot' or self.kind=='direct biot':
                if self.csv_name == None:
                    mi = np.min([np.min(map_Bi*NN_solution),np.min(self.flux_scale*self.true_biot*self.true_sol)])
                    ma = np.max([np.max(map_Bi*NN_solution),np.max(self.flux_scale*self.true_biot*self.true_sol)])
                else:
                    mi = np.min(map_Bi*NN_solution)
                    ma = np.max(map_Bi*NN_solution)                    
            else:
                if self.csv_name == None:
                    mi = np.min([np.min(map_Bi),np.min(self.flux_scale*self.true_biot)])
                    ma = np.max([np.max(map_Bi),np.max(self.flux_scale*self.true_biot)])
                else:
                    mi = np.min(map_Bi)
                    ma = np.max(map_Bi)                    
                
    
    
                    
    
            if self.kind=='exp biot' or self.kind=='direct biot':          
                im = axs[0].pcolormesh(tt_nn, xx_nn, map_Bi*NN_solution, cmap=cm.coolwarm, vmin=mi, vmax=ma)
            else:
                im = axs[0].pcolormesh(tt_nn, xx_nn, map_Bi, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                
            fig.colorbar(im, ax=axs[0])
            
    
            im = axs[0].title.set_text('MAP Flux')
            im = axs[1].title.set_text('Uncertainty')
    
    
            plt.show()
        
    
    
        if self.kind=='exp biot' or self.kind=='direct biot':
            if self.csv_name == None:
                mi = np.min([np.min(map_Bi),np.min(self.flux_scale*self.true_biot)])
                ma = np.max([np.max(map_Bi),np.max(self.flux_scale*self.true_biot)])
            else:
                mi = np.min(map_Bi)
                ma = np.max(map_Bi)   
            post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)
            miv = np.min(post_Bi_sd)
            mav = np.max(post_Bi_sd)                 
            fig, axs = plt.subplots(3, 1)
            if self.csv_name==None:
                im = axs[0].pcolormesh(tt_biot, xx_biot, self.flux_scale*(self.true_biot).T, cmap=cm.coolwarm, vmin=mi, vmax=ma)        
            im = axs[1].pcolormesh(tt_nn, xx_nn, map_Bi, cmap=cm.coolwarm, vmin=mi, vmax=ma)
            fig.colorbar(im, ax=[axs[0],axs[1]])
            im = axs[2].pcolormesh(tt_nn, xx_nn, self.flux_scale*np.sqrt(map_Bi_var), cmap=cm.coolwarm, vmin=0, vmax=mav)
            fig.colorbar(im, ax=axs[2])
            plt.show()
            im = axs[0].title.set_text('True biot')
            im = axs[1].title.set_text('Inferred biot')
            im = axs[2].title.set_text('Uncertainty')
        

    def plot_map_cross(self,res_t=500,res_x=200,uncertainty=True):
        
        nn_res = [300,150]
    
        self.map_pars_untransformed = self.sess.run(self.pars_tf[:-1])
        map_pars = self.transform_pars(self.map_pars_untransformed)
    
    
        
        map_hess_untransformed = self.sess.run(tf.hessians(-self.loss_map, self.pars_tf)[0])[:-1,:-1]
        post_cov = np.linalg.inv(-map_hess_untransformed)
        self.inv_neg_hess_untransformed = post_cov
        
    
        x_pts = np.linspace(self.xlim[0],self.xlim[1],res_x)
        t_pts = np.linspace(self.tlim[0],self.tlim[1],res_t)
        xx, tt = np.meshgrid(x_pts,t_pts)
        
        x_pts = np.linspace(self.xlim[0],self.xlim[1],nn_res[1])
        t_pts = np.linspace(self.tlim[0],self.tlim[1],nn_res[0])
        xx_nn, tt_nn = np.meshgrid(x_pts,t_pts)
    
        
        cheb_tt_nn = np.zeros([66,nn_res[0],nn_res[1]])
        cheb_xx_nn = np.zeros([66,nn_res[0],nn_res[1]])
    
        for i in range(66):
            cheb_tt_nn[i] = self.chebt.P(tt_nn,self.indices_t[i],tensor=False)
            cheb_xx_nn[i] = self.chebx.P(xx_nn,self.indices_x[i],tensor=False)
    
        
        map_Bi = self.flux_scale*sum(map_pars[i].reshape([-1,1,1])*np.reshape(cheb_xx_nn[i]*cheb_tt_nn[i],[1,nn_res[0],nn_res[1]]) for i in range(self.indices_t.shape[0]))[0]
    
        if self.positive==True:
            map_Bi = np.maximum(map_Bi,np.zeros_like(map_Bi))
    
        
        if self.kind=='exp biot' or self.kind=='exp flux':
            NN_solution = self.sess.run(tf.exp(self.u_eval(tt_nn, xx_nn, map_pars*np.ones([tt_nn.size,1])))).reshape(nn_res)
        else:
            NN_solution = self.sess.run(self.u_eval(tt_nn, xx_nn, map_pars*np.ones([tt_nn.size,1]))).reshape(nn_res)
    
    
    
        map_Bi_var = np.zeros(nn_res)
            
        if self.csv_name == None:
            if uncertainty==True:
                for i in range(66):
                    for j in range(66):
                            map_Bi_var = map_Bi_var + (post_cov[i,j]/(self.scale[i]*self.scale[j]))*cheb_tt_nn[i]*cheb_xx_nn[i]*cheb_tt_nn[j]*cheb_xx_nn[j]
                if self.kind=='exp biot' or self.kind=='direct biot':          
                    post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)*NN_solution
                else:
                    post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)
    
    
            if self.kind=='exp biot' or self.kind=='direct biot':
                if self.csv_name == None:
                    mi = np.min([np.min(map_Bi*NN_solution),np.min(self.flux_scale*self.true_biot*self.true_sol)])
                    ma = np.max([np.max(map_Bi*NN_solution),np.max(self.flux_scale*self.true_biot*self.true_sol)])
                else:
                    mi = np.min(map_Bi*NN_solution)
                    ma = np.max(map_Bi*NN_solution)                    
            else:
                if self.csv_name == None:
                    mi = np.min([np.min(map_Bi),np.min(self.flux_scale*self.true_biot)])
                    ma = np.max([np.max(map_Bi),np.max(self.flux_scale*self.true_biot)])
                else:
                    mi = np.min(map_Bi)
                    ma = np.max(map_Bi)                    
                
    
    
            
    
        else:
            if uncertainty==True:
                for i in range(66):
                    for j in range(66):
                            map_Bi_var = map_Bi_var + (post_cov[i,j]/(self.scale[i]*self.scale[j]))*cheb_tt_nn[i]*cheb_xx_nn[i]*cheb_tt_nn[j]*cheb_xx_nn[j]
                if self.kind=='exp biot' or self.kind=='direct biot':          
                    post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)*NN_solution
                else:
                    post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)
    
    
    
            if self.kind=='exp biot' or self.kind=='direct biot':
                if self.csv_name == None:
                    mi = np.min([np.min(map_Bi*NN_solution),np.min(self.flux_scale*self.true_biot*self.true_sol)])
                    ma = np.max([np.max(map_Bi*NN_solution),np.max(self.flux_scale*self.true_biot*self.true_sol)])
                else:
                    mi = np.min(map_Bi*NN_solution)
                    ma = np.max(map_Bi*NN_solution)                    
            else:
                if self.csv_name == None:
                    mi = np.min([np.min(map_Bi),np.min(self.flux_scale*self.true_biot)])
                    ma = np.max([np.max(map_Bi),np.max(self.flux_scale*self.true_biot)])
                else:
                    mi = np.min(map_Bi)
                    ma = np.max(map_Bi)                    
                
    
    
                        
    
    
             # Plot 1 dimensional spatial cross sections
        t_cross = (self.tlim[1]-self.tlim[0])*np.array([0.2,0.4,0.6,0.8])
        ind_tcross = (nn_res[0]*t_cross/(self.tlim[1]-self.tlim[0])).astype(int)
        plti = [0,1,2,3]
        
        xx_1d = np.linspace(self.xlim[0],self.xlim[1],nn_res[1])
        if self.csv_name == None:
            xx_1d2 = np.linspace(self.xlim[0],self.xlim[1],self.true_biot.shape[1])
        
        '''
        fig, axs = plt.subplots(2,3)
        
        for j in range(6):
            post_Bi_samples_1dx = sum(coefs_post[:,i].reshape([-1,1])*np.reshape(chebx.P(xx_1d,indices_x[i],tensor=False)*chebt.P(t_cross[j],indices_t[i],tensor=False),[1,nx_1d]) for i in range(indices_t.shape[0]))
            post_Bi_samples_1dx = np.maximum(post_Bi_samples_1dx, np.zeros_like(post_Bi_samples_1dx))
            post_Bi_samples_1dx = np.minimum(post_Bi_samples_1dx, 200*np.ones_like(post_Bi_samples_1dx))
            post_Bi_mean_1dx = np.mean(post_Bi_samples_1dx,axis = 0)
            post_Bi_sd = np.sqrt(np.var(post_Bi_samples_1dx,axis=0))
            funci1 = post_Bi_mean_1dx + 1.645*post_Bi_sd
            funci0 = post_Bi_mean_1dx - 1.645*post_Bi_sd
            funci0 = np.maximum(funci0, 0*np.ones_like(funci0))
            biot_1d = biot[round(t_cross[j]/3*(nt_FD-1) + 1)]
            axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_mean_1dx)
            axs[plti[j],pltj[j]].fill_between(xx_1d, funci1, funci0, facecolor='blue', alpha=0.3)
            axs[plti[j],pltj[j]].plot(xx_1d,biot_1d,label=r"True $Bi(x)$", color='tomato')
        
        
        '''
        fig, axs = plt.subplots(1,4)
                
        if self.kind=='exp biot' or self.kind=='direct biot':        
            post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)
            ''' for j in range(6):
                sol_1dx = self.sess.run(self.u_eval(t_cross[j]*np.ones_like(xx_1d),xx_1d,map_pars*(np.ones_like(xx_1d).reshape([-1,1])))).T
                post_Bi_var = post_Bi_sd[ind_tcross[j],:]
                post_Bi_median_1dx = sol_1dx.flatten()*map_Bi[ind_tcross[j],:]
                ma = np.max((map_Bi + 1.645*post_Bi_sd)*NN_solution)
                mi = np.min((map_Bi - 1.645*post_Bi_sd)*NN_solution)
    #                if self.positive==True:
    #                    post_Bi_samples_1dx = np.maximum(post_Bi_samples_1dx, np.zeros_like(post_Bi_samples_1dx))
                #post_Bi_samples_1dx = np.minimum(post_Bi_samples_1dx, 200*np.ones_like(post_Bi_samples_1dx))
            #    post_Bi_median_1dx = np.mean(post_Bi_samples_1dx,axis = 0)
            #    post_Bi_var = np.sqrt(np.var(post_Bi_samples_1dx,axis=0))
                funci1 = post_Bi_median_1dx + 1.645*post_Bi_var*sol_1dx.flatten()#np.percentile(post_Bi_samples_1dx,97.5,axis=0)# post_Bi_mean_1dx + post_Bi_var
                funci0 = post_Bi_median_1dx - 1.645*post_Bi_var*sol_1dx.flatten()# post_Bi_mean_1dx - post_Bi_var
                if self.csv_name == None:
                    biot_1d = self.true_biot[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]
                axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_median_1dx)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[0],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.3)],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.6)],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                axs[plti[j],pltj[j]].fill_between(xx_1d, funci1.flatten(), funci0.flatten(), facecolor='blue', alpha=0.3)
                axs[plti[j],pltj[j]].title.set_text('Inferred flux at t =' + str(int(t_cross[j])))
                axs[plti[j],pltj[j]].set_ylim(mi,ma)
                if self.csv_name == None:
                    axs[plti[j],pltj[j]].plot(xx_1d2,biot_1d,label=r"True $Bi(x)$", color='tomato')
            fig, axs = plt.subplots(2,3)'''
            for j in range(len(plti)):
                sol_1dx = self.sess.run(self.u_eval(t_cross[j]*np.ones_like(xx_1d),xx_1d,map_pars*(np.ones_like(xx_1d).reshape([-1,1])))).T
                post_Bi_var = post_Bi_sd[ind_tcross[j],:]
                post_Bi_median_1dx = map_Bi[ind_tcross[j],:]
                ma = np.max((map_Bi + 1.645*post_Bi_sd))
                mi = np.min((map_Bi - 1.645*post_Bi_sd))
                funci1 = post_Bi_median_1dx + 1.645*post_Bi_var*sol_1dx.flatten()#np.percentile(post_Bi_samples_1dx,97.5,axis=0)# post_Bi_mean_1dx + post_Bi_var
                funci0 = post_Bi_median_1dx - 1.645*post_Bi_var*sol_1dx.flatten()# post_Bi_mean_1dx - post_Bi_var
                if self.csv_name == None:
                    biot_1d = self.true_biot[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]
                axs[plti[j]].plot(xx_1d,post_Bi_median_1dx)
                axs[plti[j]].fill_between(xx_1d, funci1.flatten(), funci0.flatten(), facecolor='blue', alpha=0.3)
                axs[plti[j]].title.set_text('Inferred biot at t =' + str(int(t_cross[j])))
                axs[plti[j]].set_ylim(mi,ma)
                if self.csv_name == None:
                    axs[plti[j]].plot(xx_1d2,biot_1d,label=r"True $Bi(x)$", color='tomato')
        else:
            post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)
            for j in range(len(plti)):
                sol_1dx = self.sess.run(self.u_eval(t_cross[j]*np.ones_like(xx_1d),xx_1d,map_pars*(np.ones_like(xx_1d).reshape([-1,1])))).T
                post_Bi_var = post_Bi_sd[ind_tcross[j],:]
                post_Bi_median_1dx = map_Bi[ind_tcross[j],:]
                ma = np.max(map_Bi + 1.645*post_Bi_sd)
                mi = np.min(map_Bi - 1.645*post_Bi_sd)
                funci1 = post_Bi_median_1dx + 1.645*post_Bi_var#np.percentile(post_Bi_samples_1dx,97.5,axis=0)# post_Bi_mean_1dx + post_Bi_var
                funci0 = post_Bi_median_1dx - 1.645*post_Bi_var# post_Bi_mean_1dx - post_Bi_var
                if self.csv_name == None:
                    biot_1d = self.true_biot[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]
                axs[plti[j]].plot(xx_1d,post_Bi_median_1dx)
                axs[plti[j]].fill_between(xx_1d, funci1, funci0, facecolor='blue', alpha=0.3)
                axs[plti[j]].title.set_text('Inferred flux at t =' + str(int(t_cross[j])))
                axs[plti[j]].set_ylim(mi,ma)
                if self.csv_name == None:
                    axs[plti[j]].plot(xx_1d2,biot_1d,label=r"True $Bi(x)$", color='tomato')


    def plot_map_fit(self,res_t=500,res_x=200,uncertainty=True):
        
        nn_res = [300,150]
    
        self.map_pars_untransformed = self.sess.run(self.pars_tf[:-1])
        map_pars = self.transform_pars(self.map_pars_untransformed)
    
    
        
        map_hess_untransformed = self.sess.run(tf.hessians(-self.loss_map, self.pars_tf)[0])[:-1,:-1]
        post_cov = np.linalg.inv(-map_hess_untransformed)
        self.inv_neg_hess_untransformed = post_cov
        
    
        x_pts = np.linspace(self.xlim[0],self.xlim[1],res_x)
        t_pts = np.linspace(self.tlim[0],self.tlim[1],res_t)
        xx, tt = np.meshgrid(x_pts,t_pts)
    
        xx_plt = xx
        tt_plt = tt
    
    
        x_pts = np.linspace(self.xlim[0],self.xlim[1],nn_res[1])
        t_pts = np.linspace(self.tlim[0],self.tlim[1],nn_res[0])
        xx_nn, tt_nn = np.meshgrid(x_pts,t_pts)
    
        
        cheb_tt_nn = np.zeros([66,nn_res[0],nn_res[1]])
        cheb_xx_nn = np.zeros([66,nn_res[0],nn_res[1]])
    
        for i in range(66):
            cheb_tt_nn[i] = self.chebt.P(tt_nn,self.indices_t[i],tensor=False)
            cheb_xx_nn[i] = self.chebx.P(xx_nn,self.indices_x[i],tensor=False)
    
        
        map_Bi = self.flux_scale*sum(map_pars[i].reshape([-1,1,1])*np.reshape(cheb_xx_nn[i]*cheb_tt_nn[i],[1,nn_res[0],nn_res[1]]) for i in range(self.indices_t.shape[0]))[0]
    
        if self.positive==True:
            map_Bi = np.maximum(map_Bi,np.zeros_like(map_Bi))
    
        
        map_Bi_FD = griddata(np.hstack([xx_nn.reshape([-1,1]),tt_nn.reshape([-1,1])]), map_Bi.reshape([-1,1]), (xx, tt), method='linear').reshape([res_t,res_x])
    
        if self.kind=='exp biot' or self.kind=='direct biot':          
            FD_solver = FD_biot(self.xlim[0],self.xlim[1],self.tlim[1],self.L,self.R,self.I,res_x-1,res_t-1,0.5,cheb=False,bi=map_Bi_FD/self.flux_scale,c1=self.c1,c2=self.c2,c3=self.c3,T0=self.tlim[0])
        else:
            FD_solver = FD_flux(self.xlim[0],self.xlim[1],self.tlim[1],self.L,self.R,self.I,res_x-1,res_t-1,0.5,cheb=False,bi=map_Bi_FD/self.flux_scale,c1=self.c1,c2=self.c2,c3=self.c3,T0=self.tlim[0])
            
        FD_solution = FD_solver.solve()
        if self.kind=='exp biot' or self.kind=='exp flux':
            NN_solution = self.sess.run(tf.exp(self.u_eval(tt_nn, xx_nn, map_pars*np.ones([tt_nn.size,1])))).reshape(nn_res)
        else:
            NN_solution = self.sess.run(self.u_eval(tt_nn, xx_nn, map_pars*np.ones([tt_nn.size,1]))).reshape(nn_res)
    
    
    
        map_Bi_var = np.zeros(nn_res)
            
        if self.csv_name == None:
            if uncertainty==True:
                for i in range(66):
                    for j in range(66):
                            map_Bi_var = map_Bi_var + (post_cov[i,j]/(self.scale[i]*self.scale[j]))*cheb_tt_nn[i]*cheb_xx_nn[i]*cheb_tt_nn[j]*cheb_xx_nn[j]
    
    
            if self.kind=='exp biot' or self.kind=='direct biot':
                if self.csv_name == None:
                    mi = np.min([np.min(map_Bi*NN_solution),np.min(self.flux_scale*self.true_biot*self.true_sol)])
                    ma = np.max([np.max(map_Bi*NN_solution),np.max(self.flux_scale*self.true_biot*self.true_sol)])
                else:
                    mi = np.min(map_Bi*NN_solution)
                    ma = np.max(map_Bi*NN_solution)                    
            else:
                if self.csv_name == None:
                    mi = np.min([np.min(map_Bi),np.min(self.flux_scale*self.true_biot)])
                    ma = np.max([np.max(map_Bi),np.max(self.flux_scale*self.true_biot)])
                else:
                    mi = np.min(map_Bi)
                    ma = np.max(map_Bi)                    
                
    
    
            
            if self.csv_name==None:
                tt_biot = np.linspace(self.tlim[0],self.tlim[1],self.true_sol.shape[0])
                xx_biot = np.linspace(self.xlim[0],self.xlim[1],self.true_sol.shape[1])
                tt_biot,xx_biot = np.meshgrid(tt_biot,xx_biot)
                    

        mi = np.min(NN_solution)
        ma = np.max(NN_solution)
    
        if self.csv_name == None:        
            fig, axs = plt.subplots(3, 1)
            
            if self.csv_name==None:
                mi = np.min([np.min(FD_solution),np.min(NN_solution)])
                ma = np.max([np.max(FD_solution),np.max(NN_solution)])
                im = axs[0].pcolormesh(tt_biot, xx_biot, self.true_sol.T, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                
                
            im = axs[1].pcolormesh(tt_nn, xx_nn, NN_solution, cmap=cm.coolwarm, vmin=mi, vmax=ma)
            im = axs[2].pcolormesh(tt_plt, xx_plt, FD_solution.T, cmap=cm.coolwarm, vmin=mi, vmax=ma)
    
            fig.colorbar(im, ax=axs.ravel().tolist())
            plt.show()
    
            im = axs[0].title.set_text('True Solution')
            im = axs[1].title.set_text('NN solution')
            im = axs[2].title.set_text('Crank-Nicolson solution')
        else:
            fig, axs = plt.subplots(3, 1)
            
            mi = np.min(self.z_dat)
            ma = np.max(self.z_dat)
                
            im = axs[0].scatter(self.t_dat, self.x_dat, s=50, c=self.z_dat, vmin=mi, vmax=ma,  cmap=cm.coolwarm)
            im = axs[1].pcolormesh(tt_nn, xx_nn, NN_solution, cmap=cm.coolwarm, vmin=mi, vmax=ma)
            im = axs[2].pcolormesh(tt_plt, xx_plt, FD_solution.T, cmap=cm.coolwarm, vmin=mi, vmax=ma)
    
            fig.colorbar(im, ax=axs.ravel().tolist())
            plt.show()
    
            im = axs[0].title.set_text('Data')
            im = axs[1].title.set_text('NN solution')
            im = axs[2].title.set_text('Crank-Nicolson solution')
    
             # Plot 1 dimensional spatial cross sections
        t_cross = (self.tlim[1]-self.tlim[0])*np.array([0,0.2,0.4,0.6,0.8,0.99])
        plti = [0,0,0,1,1,1]
        pltj = [0,1,2,0,1,2]
        
        xx_1d = np.linspace(self.xlim[0],self.xlim[1],nn_res[1])
 
        
        fig, axs = plt.subplots(3, 1)
        
        if self.kind=='exp biot' or self.kind=='exp flux':
            pred_z = self.sess.run(tf.exp(self.u_eval(self.t_dat, self.x_dat, map_pars*np.ones([self.t_dat.size,1]))))
        else:
            pred_z = self.sess.run(self.u_eval(self.t_dat, self.x_dat, map_pars*np.ones([self.t_dat.size,1])))
    
    
        resid  = pred_z - self.z_dat
    
        axs[0].axhline(y=0., color='r', linestyle='-')
        axs[0].scatter(self.t_dat,resid,alpha=0.5,s=15)
        
        
        axs[1].axhline(y=0., color='r', linestyle='-')
        axs[1].scatter(self.x_dat,resid,alpha=0.5,s=15)
    
    
        NN_solution = griddata(np.hstack([xx_nn.reshape([-1,1]),tt_nn.reshape([-1,1])]), NN_solution.reshape([-1,1]), (xx, tt), method='linear').reshape([res_t,res_x])
        im = axs[2].pcolormesh(tt_plt, xx_plt, NN_solution - FD_solution.T, cmap=cm.coolwarm, vmin=-0.005, vmax=0.005)
        fig.colorbar(im, ax=axs.ravel().tolist())
        plt.show()
    
    
    
        im = axs[0].title.set_text('Residuals vs time')
        im = axs[1].title.set_text('Residuals vs radial loc')
        im = axs[2].title.set_text('NN-error')
    
        plt.show()
        print(np.mean(np.abs(NN_solution - FD_solution.T)))
     
        
    
        fig, axs = plt.subplots(2,3)
    
        mas = np.max(NN_solution)*self.data.dat_scale
        mis = np.min(NN_solution)*self.data.dat_scale
                
        for j in range(6):
            t_loc = self.t_dat[np.argmin(np.abs(t_cross[j]-self.t_dat))]
            tdat_1dx = self.t_dat[self.t_dat==t_loc]
            xdat_1dx = self.x_dat[self.t_dat==t_loc]
            zdat_1dx = self.z_dat[self.t_dat==t_loc]*self.data.dat_scale
            sol_1dx = self.sess.run(self.u_eval(tdat_1dx[0]*np.ones_like(xx_1d),xx_1d,map_pars*(np.ones_like(xx_1d).reshape([-1,1]))))*self.data.dat_scale
            axs[plti[j],pltj[j]].plot(xx_1d,sol_1dx)
            axs[plti[j],pltj[j]].scatter(xdat_1dx,zdat_1dx,label=r"True $Bi(x)$", color='red',alpha = 0.4)
            axs[plti[j],pltj[j]].title.set_text('PDE solution at t =' + str(int(t_cross[j])))
            axs[plti[j],pltj[j]].set_ylim(mis-5,mas+5)


    def plot_map_est(self,res_t=500,res_x=200,uncertainty=True):
        
        nn_res = [300,150]
    
        self.map_pars_untransformed = self.sess.run(self.pars_tf[:-1])
        map_pars = self.transform_pars(self.map_pars_untransformed)
    
    
        
        map_hess_untransformed = self.sess.run(tf.hessians(-self.loss_map, self.pars_tf)[0])[:-1,:-1]
        post_cov = np.linalg.inv(-map_hess_untransformed)
        self.inv_neg_hess_untransformed = post_cov
        
    
        x_pts = np.linspace(self.xlim[0],self.xlim[1],res_x)
        t_pts = np.linspace(self.tlim[0],self.tlim[1],res_t)
        xx, tt = np.meshgrid(x_pts,t_pts)
    
        xx_plt = xx
        tt_plt = tt
    
    
        x_pts = np.linspace(self.xlim[0],self.xlim[1],nn_res[1])
        t_pts = np.linspace(self.tlim[0],self.tlim[1],nn_res[0])
        xx_nn, tt_nn = np.meshgrid(x_pts,t_pts)
    
        
        cheb_tt_nn = np.zeros([66,nn_res[0],nn_res[1]])
        cheb_xx_nn = np.zeros([66,nn_res[0],nn_res[1]])
    
        for i in range(66):
            cheb_tt_nn[i] = self.chebt.P(tt_nn,self.indices_t[i],tensor=False)
            cheb_xx_nn[i] = self.chebx.P(xx_nn,self.indices_x[i],tensor=False)
    
        
        map_Bi = self.flux_scale*sum(map_pars[i].reshape([-1,1,1])*np.reshape(cheb_xx_nn[i]*cheb_tt_nn[i],[1,nn_res[0],nn_res[1]]) for i in range(self.indices_t.shape[0]))[0]
    
        if self.positive==True:
            map_Bi = np.maximum(map_Bi,np.zeros_like(map_Bi))
    
        
        map_Bi_FD = griddata(np.hstack([xx_nn.reshape([-1,1]),tt_nn.reshape([-1,1])]), map_Bi.reshape([-1,1]), (xx, tt), method='linear').reshape([res_t,res_x])
    
        if self.kind=='exp biot' or self.kind=='direct biot':          
            FD_solver = FD_biot(self.xlim[0],self.xlim[1],self.tlim[1],self.L,self.R,self.I,res_x-1,res_t-1,0.5,cheb=False,bi=map_Bi_FD/self.flux_scale,c1=self.c1,c2=self.c2,c3=self.c3,T0=self.tlim[0])
        else:
            FD_solver = FD_flux(self.xlim[0],self.xlim[1],self.tlim[1],self.L,self.R,self.I,res_x-1,res_t-1,0.5,cheb=False,bi=map_Bi_FD/self.flux_scale,c1=self.c1,c2=self.c2,c3=self.c3,T0=self.tlim[0])
            
        FD_solution = FD_solver.solve()
        if self.kind=='exp biot' or self.kind=='exp flux':
            NN_solution = self.sess.run(tf.exp(self.u_eval(tt_nn, xx_nn, map_pars*np.ones([tt_nn.size,1])))).reshape(nn_res)
        else:
            NN_solution = self.sess.run(self.u_eval(tt_nn, xx_nn, map_pars*np.ones([tt_nn.size,1]))).reshape(nn_res)
    
    
    
        map_Bi_var = np.zeros(nn_res)
            
        if self.csv_name == None:
            fig, axs = plt.subplots(3, 1)
    
            if uncertainty==True:
                for i in range(66):
                    for j in range(66):
                            map_Bi_var = map_Bi_var + (post_cov[i,j]/(self.scale[i]*self.scale[j]))*cheb_tt_nn[i]*cheb_xx_nn[i]*cheb_tt_nn[j]*cheb_xx_nn[j]
                if self.kind=='exp biot' or self.kind=='direct biot':          
                    post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)*NN_solution
                else:
                    post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)
                miv = np.min(post_Bi_sd)
                mav = np.max(post_Bi_sd)
                im = axs[2].pcolormesh(tt_nn, xx_nn, post_Bi_sd, cmap=cm.coolwarm, vmin=miv, vmax=mav)
                fig.colorbar(im, ax=axs[2])    
    
    
            if self.kind=='exp biot' or self.kind=='direct biot':
                if self.csv_name == None:
                    mi = np.min([np.min(map_Bi*NN_solution),np.min(self.flux_scale*self.true_biot*self.true_sol)])
                    ma = np.max([np.max(map_Bi*NN_solution),np.max(self.flux_scale*self.true_biot*self.true_sol)])
                else:
                    mi = np.min(map_Bi*NN_solution)
                    ma = np.max(map_Bi*NN_solution)                    
            else:
                if self.csv_name == None:
                    mi = np.min([np.min(map_Bi),np.min(self.flux_scale*self.true_biot)])
                    ma = np.max([np.max(map_Bi),np.max(self.flux_scale*self.true_biot)])
                else:
                    mi = np.min(map_Bi)
                    ma = np.max(map_Bi)                    
                
    
    
            
            if self.csv_name==None:
                tt_biot = np.linspace(self.tlim[0],self.tlim[1],self.true_sol.shape[0])
                xx_biot = np.linspace(self.xlim[0],self.xlim[1],self.true_sol.shape[1])
                tt_biot,xx_biot = np.meshgrid(tt_biot,xx_biot)
                if self.kind=='exp biot' or self.kind=='direct biot':
                    im = axs[0].pcolormesh(tt_biot, xx_biot, self.flux_scale*(self.true_biot*self.true_sol).T, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                else:
                    im = axs[0].pcolormesh(tt_biot, xx_biot, self.flux_scale*(self.true_biot).T, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                    
    
            if self.kind=='exp biot' or self.kind=='direct biot':          
                im = axs[1].pcolormesh(tt_nn, xx_nn, map_Bi*NN_solution, cmap=cm.coolwarm, vmin=mi, vmax=ma)
            else:
                im = axs[1].pcolormesh(tt_nn, xx_nn, map_Bi, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                
            fig.colorbar(im, ax=[axs[0],axs[1]])
            
    
    
    
            im = axs[0].title.set_text('True Flux')
            im = axs[1].title.set_text('MAP Flux')
            im = axs[2].title.set_text('Uncertainty')
    
    
        else:
            fig, axs = plt.subplots(2, 1)
            
            if uncertainty==True:
                for i in range(66):
                    for j in range(66):
                            map_Bi_var = map_Bi_var + (post_cov[i,j]/(self.scale[i]*self.scale[j]))*cheb_tt_nn[i]*cheb_xx_nn[i]*cheb_tt_nn[j]*cheb_xx_nn[j]
                if self.kind=='exp biot' or self.kind=='direct biot':          
                    post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)*NN_solution
                else:
                    post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)
                miv = np.min(post_Bi_sd)
                mav = np.max(post_Bi_sd)
                im = axs[1].pcolormesh(tt_nn, xx_nn, post_Bi_sd, cmap=cm.coolwarm, vmin=miv, vmax=mav)
                fig.colorbar(im, ax=axs[1])    
    
    
    
            if self.kind=='exp biot' or self.kind=='direct biot':
                if self.csv_name == None:
                    mi = np.min([np.min(map_Bi*NN_solution),np.min(self.flux_scale*self.true_biot*self.true_sol)])
                    ma = np.max([np.max(map_Bi*NN_solution),np.max(self.flux_scale*self.true_biot*self.true_sol)])
                else:
                    mi = np.min(map_Bi*NN_solution)
                    ma = np.max(map_Bi*NN_solution)                    
            else:
                if self.csv_name == None:
                    mi = np.min([np.min(map_Bi),np.min(self.flux_scale*self.true_biot)])
                    ma = np.max([np.max(map_Bi),np.max(self.flux_scale*self.true_biot)])
                else:
                    mi = np.min(map_Bi)
                    ma = np.max(map_Bi)                    
                
    
    
                    
    
            if self.kind=='exp biot' or self.kind=='direct biot':          
                im = axs[0].pcolormesh(tt_nn, xx_nn, map_Bi*NN_solution, cmap=cm.coolwarm, vmin=mi, vmax=ma)
            else:
                im = axs[0].pcolormesh(tt_nn, xx_nn, map_Bi, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                
            fig.colorbar(im, ax=axs[0])
            
    
            im = axs[0].title.set_text('MAP Flux')
            im = axs[1].title.set_text('Uncertainty')
    
    
            plt.show()
        
    
    
        if self.kind=='exp biot' or self.kind=='direct biot':
            if self.csv_name == None:
                mi = np.min([np.min(map_Bi),np.min(self.flux_scale*self.true_biot)])
                ma = np.max([np.max(map_Bi),np.max(self.flux_scale*self.true_biot)])
            else:
                mi = np.min(map_Bi)
                ma = np.max(map_Bi)   
            post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)
            miv = np.min(post_Bi_sd)
            mav = np.max(post_Bi_sd)                 
            fig, axs = plt.subplots(3, 1)
            if self.csv_name==None:
                im = axs[0].pcolormesh(tt_biot, xx_biot, self.flux_scale*(self.true_biot).T, cmap=cm.coolwarm, vmin=mi, vmax=ma)        
            im = axs[1].pcolormesh(tt_nn, xx_nn, map_Bi, cmap=cm.coolwarm, vmin=mi, vmax=ma)
            fig.colorbar(im, ax=[axs[0],axs[1]])
            im = axs[2].pcolormesh(tt_nn, xx_nn, self.flux_scale*np.sqrt(map_Bi_var), cmap=cm.coolwarm, vmin=0, vmax=mav)
            fig.colorbar(im, ax=axs[2])
            plt.show()
            im = axs[0].title.set_text('True biot')
            im = axs[1].title.set_text('Inferred biot')
            im = axs[2].title.set_text('Uncertainty')
        
        mi = np.min(NN_solution)
        ma = np.max(NN_solution)
    
        if self.csv_name == None:        
            fig, axs = plt.subplots(3, 1)
            
            if self.csv_name==None:
                mi = np.min([np.min(FD_solution),np.min(NN_solution)])
                ma = np.max([np.max(FD_solution),np.max(NN_solution)])
                im = axs[0].pcolormesh(tt_biot, xx_biot, self.true_sol.T, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                
                
            im = axs[1].pcolormesh(tt_nn, xx_nn, NN_solution, cmap=cm.coolwarm, vmin=mi, vmax=ma)
            im = axs[2].pcolormesh(tt_plt, xx_plt, FD_solution.T, cmap=cm.coolwarm, vmin=mi, vmax=ma)
    
            fig.colorbar(im, ax=axs.ravel().tolist())
            plt.show()
    
            im = axs[0].title.set_text('True Solution')
            im = axs[1].title.set_text('NN solution')
            im = axs[2].title.set_text('Crank-Nicolson solution')
        else:
            fig, axs = plt.subplots(3, 1)
            
            mi = np.min(self.z_dat)
            ma = np.max(self.z_dat)
                
            im = axs[0].scatter(self.t_dat, self.x_dat, s=50, c=self.z_dat, vmin=mi, vmax=ma,  cmap=cm.coolwarm)
            im = axs[1].pcolormesh(tt_nn, xx_nn, NN_solution, cmap=cm.coolwarm, vmin=mi, vmax=ma)
            im = axs[2].pcolormesh(tt_plt, xx_plt, FD_solution.T, cmap=cm.coolwarm, vmin=mi, vmax=ma)
    
            fig.colorbar(im, ax=axs.ravel().tolist())
            plt.show()
    
            im = axs[0].title.set_text('Data')
            im = axs[1].title.set_text('NN solution')
            im = axs[2].title.set_text('Crank-Nicolson solution')
    
             # Plot 1 dimensional spatial cross sections
        t_cross = (self.tlim[1]-self.tlim[0])*np.array([0,0.2,0.4,0.6,0.8,0.99])
        ind_tcross = (nn_res[0]*t_cross/(self.tlim[1]-self.tlim[0])).astype(int)
        plti = [0,0,0,1,1,1]
        pltj = [0,1,2,0,1,2]
        
        xx_1d = np.linspace(self.xlim[0],self.xlim[1],nn_res[1])
        if self.csv_name == None:
            xx_1d2 = np.linspace(self.xlim[0],self.xlim[1],self.true_biot.shape[1])
        
        '''
        fig, axs = plt.subplots(2,3)
        
        for j in range(6):
            post_Bi_samples_1dx = sum(coefs_post[:,i].reshape([-1,1])*np.reshape(chebx.P(xx_1d,indices_x[i],tensor=False)*chebt.P(t_cross[j],indices_t[i],tensor=False),[1,nx_1d]) for i in range(indices_t.shape[0]))
            post_Bi_samples_1dx = np.maximum(post_Bi_samples_1dx, np.zeros_like(post_Bi_samples_1dx))
            post_Bi_samples_1dx = np.minimum(post_Bi_samples_1dx, 200*np.ones_like(post_Bi_samples_1dx))
            post_Bi_mean_1dx = np.mean(post_Bi_samples_1dx,axis = 0)
            post_Bi_sd = np.sqrt(np.var(post_Bi_samples_1dx,axis=0))
            funci1 = post_Bi_mean_1dx + 1.645*post_Bi_sd
            funci0 = post_Bi_mean_1dx - 1.645*post_Bi_sd
            funci0 = np.maximum(funci0, 0*np.ones_like(funci0))
            biot_1d = biot[round(t_cross[j]/3*(nt_FD-1) + 1)]
            axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_mean_1dx)
            axs[plti[j],pltj[j]].fill_between(xx_1d, funci1, funci0, facecolor='blue', alpha=0.3)
            axs[plti[j],pltj[j]].plot(xx_1d,biot_1d,label=r"True $Bi(x)$", color='tomato')
        
        
        '''
        fig, axs = plt.subplots(2,3)
                
        if self.kind=='exp biot' or self.kind=='direct biot':        
            post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)
            ''' for j in range(6):
                sol_1dx = self.sess.run(self.u_eval(t_cross[j]*np.ones_like(xx_1d),xx_1d,map_pars*(np.ones_like(xx_1d).reshape([-1,1])))).T
                post_Bi_var = post_Bi_sd[ind_tcross[j],:]
                post_Bi_median_1dx = sol_1dx.flatten()*map_Bi[ind_tcross[j],:]
                ma = np.max((map_Bi + 1.645*post_Bi_sd)*NN_solution)
                mi = np.min((map_Bi - 1.645*post_Bi_sd)*NN_solution)
    #                if self.positive==True:
    #                    post_Bi_samples_1dx = np.maximum(post_Bi_samples_1dx, np.zeros_like(post_Bi_samples_1dx))
                #post_Bi_samples_1dx = np.minimum(post_Bi_samples_1dx, 200*np.ones_like(post_Bi_samples_1dx))
            #    post_Bi_median_1dx = np.mean(post_Bi_samples_1dx,axis = 0)
            #    post_Bi_var = np.sqrt(np.var(post_Bi_samples_1dx,axis=0))
                funci1 = post_Bi_median_1dx + 1.645*post_Bi_var*sol_1dx.flatten()#np.percentile(post_Bi_samples_1dx,97.5,axis=0)# post_Bi_mean_1dx + post_Bi_var
                funci0 = post_Bi_median_1dx - 1.645*post_Bi_var*sol_1dx.flatten()# post_Bi_mean_1dx - post_Bi_var
                if self.csv_name == None:
                    biot_1d = self.true_biot[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]
                axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_median_1dx)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[0],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.3)],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.6)],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                axs[plti[j],pltj[j]].fill_between(xx_1d, funci1.flatten(), funci0.flatten(), facecolor='blue', alpha=0.3)
                axs[plti[j],pltj[j]].title.set_text('Inferred flux at t =' + str(int(t_cross[j])))
                axs[plti[j],pltj[j]].set_ylim(mi,ma)
                if self.csv_name == None:
                    axs[plti[j],pltj[j]].plot(xx_1d2,biot_1d,label=r"True $Bi(x)$", color='tomato')
            fig, axs = plt.subplots(2,3)'''
            for j in range(6):
                sol_1dx = self.sess.run(self.u_eval(t_cross[j]*np.ones_like(xx_1d),xx_1d,map_pars*(np.ones_like(xx_1d).reshape([-1,1])))).T
                post_Bi_var = post_Bi_sd[ind_tcross[j],:]
                post_Bi_median_1dx = map_Bi[ind_tcross[j],:]
                ma = np.max((map_Bi + 1.645*post_Bi_sd))
                mi = np.min((map_Bi - 1.645*post_Bi_sd))
    #                if self.positive==True:
    #                    post_Bi_samples_1dx = np.maximum(post_Bi_samples_1dx, np.zeros_like(post_Bi_samples_1dx))
                #post_Bi_samples_1dx = np.minimum(post_Bi_samples_1dx, 200*np.ones_like(post_Bi_samples_1dx))
            #    post_Bi_median_1dx = np.mean(post_Bi_samples_1dx,axis = 0)
            #    post_Bi_var = np.sqrt(np.var(post_Bi_samples_1dx,axis=0))
                funci1 = post_Bi_median_1dx + 1.645*post_Bi_var*sol_1dx.flatten()#np.percentile(post_Bi_samples_1dx,97.5,axis=0)# post_Bi_mean_1dx + post_Bi_var
                funci0 = post_Bi_median_1dx - 1.645*post_Bi_var*sol_1dx.flatten()# post_Bi_mean_1dx - post_Bi_var
                if self.csv_name == None:
                    biot_1d = self.true_biot[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]
                axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_median_1dx)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[0],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.3)],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.6)],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                axs[plti[j],pltj[j]].fill_between(xx_1d, funci1.flatten(), funci0.flatten(), facecolor='blue', alpha=0.3)
                axs[plti[j],pltj[j]].title.set_text('Inferred biot at t =' + str(int(t_cross[j])))
                axs[plti[j],pltj[j]].set_ylim(mi,ma)
                if self.csv_name == None:
                    axs[plti[j],pltj[j]].plot(xx_1d2,biot_1d,label=r"True $Bi(x)$", color='tomato')
        else:
            post_Bi_sd = self.flux_scale*np.sqrt(map_Bi_var)
            for j in range(6):
                sol_1dx = self.sess.run(self.u_eval(t_cross[j]*np.ones_like(xx_1d),xx_1d,map_pars*(np.ones_like(xx_1d).reshape([-1,1])))).T
                post_Bi_var = post_Bi_sd[ind_tcross[j],:]
                post_Bi_median_1dx = map_Bi[ind_tcross[j],:]
                ma = np.max(map_Bi + 1.645*post_Bi_sd)
                mi = np.min(map_Bi - 1.645*post_Bi_sd)
    #                if self.positive==True:
    #                    post_Bi_samples_1dx = np.maximum(post_Bi_samples_1dx, np.zeros_like(post_Bi_samples_1dx))
                #post_Bi_samples_1dx = np.minimum(post_Bi_samples_1dx, 200*np.ones_like(post_Bi_samples_1dx))
                funci1 = post_Bi_median_1dx + 1.645*post_Bi_var#np.percentile(post_Bi_samples_1dx,97.5,axis=0)# post_Bi_mean_1dx + post_Bi_var
                funci0 = post_Bi_median_1dx - 1.645*post_Bi_var# post_Bi_mean_1dx - post_Bi_var
                if self.csv_name == None:
                    biot_1d = self.true_biot[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]
                axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_median_1dx)
    #                axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[0],label=r"True $Bi(x)$", color='green',alpha = 0.1)
    #                axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.3)],label=r"True $Bi(x)$", color='green',alpha = 0.1)
    #                axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.6)],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                axs[plti[j],pltj[j]].fill_between(xx_1d, funci1, funci0, facecolor='blue', alpha=0.3)
                axs[plti[j],pltj[j]].title.set_text('Inferred flux at t =' + str(int(t_cross[j])))
                axs[plti[j],pltj[j]].set_ylim(mi,ma)
                if self.csv_name == None:
                    axs[plti[j],pltj[j]].plot(xx_1d2,biot_1d,label=r"True $Bi(x)$", color='tomato')
    
    
    
    
        
        fig, axs = plt.subplots(3, 1)
        
        if self.kind=='exp biot' or self.kind=='exp flux':
            pred_z = self.sess.run(tf.exp(self.u_eval(self.t_dat, self.x_dat, map_pars*np.ones([self.t_dat.size,1]))))
        else:
            pred_z = self.sess.run(self.u_eval(self.t_dat, self.x_dat, map_pars*np.ones([self.t_dat.size,1])))
    
    
        resid  = pred_z - self.z_dat
    
        axs[0].axhline(y=0., color='r', linestyle='-')
        axs[0].scatter(self.t_dat,resid,alpha=0.5,s=15)
        
        
        axs[1].axhline(y=0., color='r', linestyle='-')
        axs[1].scatter(self.x_dat,resid,alpha=0.5,s=15)
    
    
        NN_solution = griddata(np.hstack([xx_nn.reshape([-1,1]),tt_nn.reshape([-1,1])]), NN_solution.reshape([-1,1]), (xx, tt), method='linear').reshape([res_t,res_x])
        im = axs[2].pcolormesh(tt_plt, xx_plt, NN_solution - FD_solution.T, cmap=cm.coolwarm, vmin=-0.005, vmax=0.005)
        fig.colorbar(im, ax=axs.ravel().tolist())
        plt.show()
    
    
    
        im = axs[0].title.set_text('Residuals vs time')
        im = axs[1].title.set_text('Residuals vs radial loc')
        im = axs[2].title.set_text('NN-error')
    
        plt.show()
        print(np.mean(np.abs(NN_solution - FD_solution.T)))
     
        
    
        fig, axs = plt.subplots(2,3)
    
        mas = np.max(NN_solution)*self.data.dat_scale
        mis = np.min(NN_solution)*self.data.dat_scale
                
        for j in range(6):
            t_loc = self.t_dat[np.argmin(np.abs(t_cross[j]-self.t_dat))]
            tdat_1dx = self.t_dat[self.t_dat==t_loc]
            xdat_1dx = self.x_dat[self.t_dat==t_loc]
            zdat_1dx = self.z_dat[self.t_dat==t_loc]*self.data.dat_scale
            sol_1dx = self.sess.run(self.u_eval(tdat_1dx[0]*np.ones_like(xx_1d),xx_1d,map_pars*(np.ones_like(xx_1d).reshape([-1,1]))))*self.data.dat_scale
            axs[plti[j],pltj[j]].plot(xx_1d,sol_1dx)
            axs[plti[j],pltj[j]].scatter(xdat_1dx,zdat_1dx,label=r"True $Bi(x)$", color='red',alpha = 0.4)
            axs[plti[j],pltj[j]].title.set_text('PDE solution at t =' + str(int(t_cross[j])))
            axs[plti[j],pltj[j]].set_ylim(mis-5,mas+5)


    def L1_error(self,res_t=500,res_x=200,pars = None, uncertainty=True):
        
        nn_res = [300,150]
    
        self.map_pars_untransformed = self.sess.run(self.pars_tf[:-1])
        
        map_pars = self.transform_pars(self.map_pars_untransformed)
        
        
    
        
        map_hess_untransformed = self.sess.run(tf.hessians(-self.loss_map, self.pars_tf)[0])[:-1,:-1]
        post_cov = np.linalg.inv(-map_hess_untransformed)
        self.inv_neg_hess_untransformed = post_cov
        
    
        x_pts = np.linspace(self.xlim[0],self.xlim[1],res_x)
        t_pts = np.linspace(self.tlim[0],self.tlim[1],res_t)
        xx, tt = np.meshgrid(x_pts,t_pts)
    
    
        x_pts = np.linspace(self.xlim[0],self.xlim[1],nn_res[1])
        t_pts = np.linspace(self.tlim[0],self.tlim[1],nn_res[0])
        xx_nn, tt_nn = np.meshgrid(x_pts,t_pts)
    
        
        cheb_tt_nn = np.zeros([66,nn_res[0],nn_res[1]])
        cheb_xx_nn = np.zeros([66,nn_res[0],nn_res[1]])
    
        for i in range(66):
            cheb_tt_nn[i] = self.chebt.P(tt_nn,self.indices_t[i],tensor=False)
            cheb_xx_nn[i] = self.chebx.P(xx_nn,self.indices_x[i],tensor=False)
    
        
        if pars!=None:
            map_pars = pars
            
        map_Bi = self.flux_scale*sum(map_pars[i].reshape([-1,1,1])*np.reshape(cheb_xx_nn[i]*cheb_tt_nn[i],[1,nn_res[0],nn_res[1]]) for i in range(self.indices_t.shape[0]))[0]
    
        if self.positive==True:
            map_Bi = np.maximum(map_Bi,np.zeros_like(map_Bi))
    
        
        map_Bi_FD = griddata(np.hstack([xx_nn.reshape([-1,1]),tt_nn.reshape([-1,1])]), map_Bi.reshape([-1,1]), (xx, tt), method='linear').reshape([res_t,res_x])
    
        if self.kind=='exp biot' or self.kind=='direct biot':          
            FD_solver = FD_biot(self.xlim[0],self.xlim[1],self.tlim[1],self.L,self.R,self.I,res_x-1,res_t-1,0.5,cheb=False,bi=map_Bi_FD/self.flux_scale,c1=self.c1,c2=self.c2,c3=self.c3,T0=self.tlim[0])
        else:
            FD_solver = FD_flux(self.xlim[0],self.xlim[1],self.tlim[1],self.L,self.R,self.I,res_x-1,res_t-1,0.5,cheb=False,bi=map_Bi_FD/self.flux_scale,c1=self.c1,c2=self.c2,c3=self.c3,T0=self.tlim[0])
            
        FD_solution = FD_solver.solve()
        if self.kind=='exp biot' or self.kind=='exp flux':
            NN_solution = self.sess.run(tf.exp(self.u_eval(tt_nn, xx_nn, map_pars*np.ones([tt_nn.size,1])))).reshape(nn_res)
        else:
            NN_solution = self.sess.run(self.u_eval(tt_nn, xx_nn, map_pars*np.ones([tt_nn.size,1]))).reshape(nn_res)
 
        NN_solution = griddata(np.hstack([xx_nn.reshape([-1,1]),tt_nn.reshape([-1,1])]), NN_solution.reshape([-1,1]), (xx, tt), method='linear').reshape([res_t,res_x])

        print(np.mean(np.abs(NN_solution-FD_solution.T)))


    def plot_mcmc_traces(self,burnin_prop = 0.8):
        n_samples = len(self.sig_eps_sample)
        burnin = round(n_samples*burnin_prop)
        
        coefs_exburnin = np.array(self.coefs_sample[burnin:])
        sig_eps_exburnin = np.array(self.sig_eps_sample[burnin:])
        # Trace plot
        fig, axs = plt.subplots(12,3)
        
        #coefs_exburnin = scale*coefs_exburnin
        axs[0,0].plot(sig_eps_exburnin)
        axs[1,0].plot(np.array(coefs_exburnin)[:,0])
        axs[2,0].plot(np.array(coefs_exburnin)[:,1])
        axs[3,0].plot(np.array(coefs_exburnin)[:,2])
        axs[4,0].plot(np.array(coefs_exburnin)[:,3])
        axs[5,0].plot(np.array(coefs_exburnin)[:,4])
        axs[6,0].plot(np.array(coefs_exburnin)[:,5])
        axs[7,0].plot(np.array(coefs_exburnin)[:,6])
        axs[8,0].plot(np.array(coefs_exburnin)[:,7])
        axs[9,0].plot(np.array(coefs_exburnin)[:,8])
        axs[10,0].plot(np.array(coefs_exburnin)[:,9])
        axs[11,0].plot(np.array(coefs_exburnin)[:,10])
        
        
        axs[1,1].plot(np.array(coefs_exburnin)[:,11])
        axs[2,1].plot(np.array(coefs_exburnin)[:,21])
        axs[3,1].plot(np.array(coefs_exburnin)[:,30])
        axs[4,1].plot(np.array(coefs_exburnin)[:,38])
        axs[5,1].plot(np.array(coefs_exburnin)[:,45])
        axs[6,1].plot(np.array(coefs_exburnin)[:,51])
        axs[7,1].plot(np.array(coefs_exburnin)[:,56])
        axs[8,1].plot(np.array(coefs_exburnin)[:,60])
        axs[9,1].plot(np.array(coefs_exburnin)[:,63])
        axs[10,1].plot(np.array(coefs_exburnin)[:,65])
        #axs[11,0].plot(np.array(coefs_exburnin)[:,66])
        
        axs[1,2].plot(np.array(coefs_exburnin)[:,0])
        axs[2,2].plot(np.array(coefs_exburnin)[:,1])
        axs[3,2].plot(np.array(coefs_exburnin)[:,11])
        axs[4,2].plot(np.array(coefs_exburnin)[:,2])
        axs[5,2].plot(np.array(coefs_exburnin)[:,21])
        axs[6,2].plot(np.array(coefs_exburnin)[:,12])
        axs[7,2].plot(np.array(coefs_exburnin)[:,3])
        axs[8,2].plot(np.array(coefs_exburnin)[:,13])
        axs[9,2].plot(np.array(coefs_exburnin)[:,22])
        axs[10,2].plot(np.array(coefs_exburnin)[:,30])
        
        fig, axs = plt.subplots(12,3)
        
        #coefs_exburnin = scale*coefs_exburnin
        axs[0,0].plot(sig_eps_exburnin)
        axs[1,0].plot(np.array(coefs_exburnin)[:,12])
        axs[2,0].plot(np.array(coefs_exburnin)[:,13])
        axs[3,0].plot(np.array(coefs_exburnin)[:,14])
        axs[4,0].plot(np.array(coefs_exburnin)[:,15])
        axs[5,0].plot(np.array(coefs_exburnin)[:,16])
        axs[6,0].plot(np.array(coefs_exburnin)[:,17])
        axs[7,0].plot(np.array(coefs_exburnin)[:,18])
        axs[8,0].plot(np.array(coefs_exburnin)[:,19])
        axs[9,0].plot(np.array(coefs_exburnin)[:,20])
        axs[10,0].plot(np.array(coefs_exburnin)[:,21])
        axs[11,0].plot(np.array(coefs_exburnin)[:,22])
        
        
        axs[1,1].plot(np.array(coefs_exburnin)[:,23])
        axs[2,1].plot(np.array(coefs_exburnin)[:,24])
        axs[3,1].plot(np.array(coefs_exburnin)[:,25])
        axs[4,1].plot(np.array(coefs_exburnin)[:,26])
        axs[5,1].plot(np.array(coefs_exburnin)[:,27])
        axs[6,1].plot(np.array(coefs_exburnin)[:,28])
        axs[7,1].plot(np.array(coefs_exburnin)[:,29])
        axs[8,1].plot(np.array(coefs_exburnin)[:,30])
        axs[9,1].plot(np.array(coefs_exburnin)[:,63])
        axs[10,1].plot(np.array(coefs_exburnin)[:,65])
        #axs[11,0].plot(np.array(coefs_exburnin)[:,66])
        
        axs[1,2].plot(np.array(coefs_exburnin)[:,31])
        axs[2,2].plot(np.array(coefs_exburnin)[:,32])
        axs[3,2].plot(np.array(coefs_exburnin)[:,33])
        axs[4,2].plot(np.array(coefs_exburnin)[:,34])
        axs[5,2].plot(np.array(coefs_exburnin)[:,35])
        axs[6,2].plot(np.array(coefs_exburnin)[:,36])
        axs[7,2].plot(np.array(coefs_exburnin)[:,37])
        axs[8,2].plot(np.array(coefs_exburnin)[:,38])
        axs[9,2].plot(np.array(coefs_exburnin)[:,39])
        axs[10,2].plot(np.array(coefs_exburnin)[:,40])
        
        
        fig, axs = plt.subplots(12,3)
        
        #coefs_exburnin = scale*coefs_exburnin
        axs[0,0].plot(sig_eps_exburnin)
        axs[1,0].plot(np.array(coefs_exburnin)[:,41])
        axs[2,0].plot(np.array(coefs_exburnin)[:,42])
        axs[3,0].plot(np.array(coefs_exburnin)[:,43])
        axs[4,0].plot(np.array(coefs_exburnin)[:,44])
        axs[5,0].plot(np.array(coefs_exburnin)[:,45])
        axs[6,0].plot(np.array(coefs_exburnin)[:,46])
        axs[7,0].plot(np.array(coefs_exburnin)[:,47])
        axs[8,0].plot(np.array(coefs_exburnin)[:,48])
        axs[9,0].plot(np.array(coefs_exburnin)[:,49])
        axs[10,0].plot(np.array(coefs_exburnin)[:,50])
        axs[11,0].plot(np.array(coefs_exburnin)[:,51])
        
        
        axs[1,1].plot(np.array(coefs_exburnin)[:,52])
        axs[2,1].plot(np.array(coefs_exburnin)[:,53])
        axs[3,1].plot(np.array(coefs_exburnin)[:,54])
        axs[4,1].plot(np.array(coefs_exburnin)[:,55])
        axs[5,1].plot(np.array(coefs_exburnin)[:,56])
        axs[6,1].plot(np.array(coefs_exburnin)[:,57])
        axs[7,1].plot(np.array(coefs_exburnin)[:,58])
        axs[8,1].plot(np.array(coefs_exburnin)[:,59])
        axs[9,1].plot(np.array(coefs_exburnin)[:,60])
        axs[10,1].plot(np.array(coefs_exburnin)[:,61])
        #axs[11,0].plot(np.array(coefs_exburnin)[:,66])
        
        axs[1,2].plot(np.array(coefs_exburnin)[:,62])
        axs[2,2].plot(np.array(coefs_exburnin)[:,63])
        axs[3,2].plot(np.array(coefs_exburnin)[:,64])
        axs[4,2].plot(np.array(coefs_exburnin)[:,65])        


    def plot_mcmc_marginals(self,burnin_prop = 0.8):

        n_samples = len(self.sig_eps_sample)
        burnin = round(n_samples*burnin_prop)
        
        coefs_exburnin = np.array(self.coefs_sample[burnin:])
        
        terms = [2,4,3,5]
        
        fig, axs = plt.subplots(4,4)
        az.plot_kde(np.array(coefs_exburnin)[:,0], values2=np.array(coefs_exburnin)[:,1],ax = axs[0,0])
        az.plot_kde(np.array(coefs_exburnin)[:,2], values2=np.array(coefs_exburnin)[:,3],ax = axs[0,1],contour_kwargs={"colors":None,"levels":3}, contourf_kwargs={"levels":3},bw=2000)
        az.plot_kde(np.array(coefs_exburnin)[:,4], values2=np.array(coefs_exburnin)[:,5],ax = axs[0,2],contour_kwargs={"colors":None,"levels":4}, contourf_kwargs={"levels":4},bw=40)
        az.plot_kde(np.array(coefs_exburnin)[:,6], values2=np.array(coefs_exburnin)[:,7],ax = axs[0,3],contour_kwargs={"colors":None,"levels":5}, contourf_kwargs={"levels":5},bw=60)
        az.plot_kde(np.array(coefs_exburnin)[:,8], values2=np.array(coefs_exburnin)[:,9],ax = axs[1,0],contour_kwargs={"colors":None,"levels":6}, contourf_kwargs={"levels":6},bw=8)
        az.plot_kde(np.array(coefs_exburnin)[:,10], values2=np.array(coefs_exburnin)[:,11],ax = axs[1,1])
        az.plot_kde(np.array(coefs_exburnin)[:,12], values2=np.array(coefs_exburnin)[:,13],ax = axs[1,2],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,14], values2=np.array(coefs_exburnin)[:,15],ax = axs[1,3],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,16], values2=np.array(coefs_exburnin)[:,17],ax = axs[2,0],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,18], values2=np.array(coefs_exburnin)[:,19],ax = axs[2,1],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,20], values2=np.array(coefs_exburnin)[:,21],ax = axs[2,2])
        az.plot_kde(np.array(coefs_exburnin)[:,22], values2=np.array(coefs_exburnin)[:,23],ax = axs[2,3],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,24], values2=np.array(coefs_exburnin)[:,25],ax = axs[3,0],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,26], values2=np.array(coefs_exburnin)[:,27],ax = axs[3,1],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,28], values2=np.array(coefs_exburnin)[:,29],ax = axs[3,2],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,30], values2=np.array(coefs_exburnin)[:,31],ax = axs[3,3])

        
        fig, axs = plt.subplots(4,4)
        az.plot_kde(np.array(coefs_exburnin)[:,62], values2=np.array(coefs_exburnin)[:,63],ax = axs[0,0])
        az.plot_kde(np.array(coefs_exburnin)[:,32], values2=np.array(coefs_exburnin)[:,33],ax = axs[0,1],contour_kwargs={"colors":None,"levels":3}, contourf_kwargs={"levels":3},bw=2000)
        az.plot_kde(np.array(coefs_exburnin)[:,34], values2=np.array(coefs_exburnin)[:,35],ax = axs[0,2],contour_kwargs={"colors":None,"levels":4}, contourf_kwargs={"levels":4},bw=40)
        az.plot_kde(np.array(coefs_exburnin)[:,36], values2=np.array(coefs_exburnin)[:,37],ax = axs[0,3],contour_kwargs={"colors":None,"levels":5}, contourf_kwargs={"levels":5},bw=60)
        az.plot_kde(np.array(coefs_exburnin)[:,38], values2=np.array(coefs_exburnin)[:,39],ax = axs[1,0],contour_kwargs={"colors":None,"levels":6}, contourf_kwargs={"levels":6},bw=8)
        az.plot_kde(np.array(coefs_exburnin)[:,40], values2=np.array(coefs_exburnin)[:,41],ax = axs[1,1])
        az.plot_kde(np.array(coefs_exburnin)[:,42], values2=np.array(coefs_exburnin)[:,43],ax = axs[1,2],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,44], values2=np.array(coefs_exburnin)[:,45],ax = axs[1,3],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,46], values2=np.array(coefs_exburnin)[:,47],ax = axs[2,0],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,48], values2=np.array(coefs_exburnin)[:,49],ax = axs[2,1],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,50], values2=np.array(coefs_exburnin)[:,51],ax = axs[2,2])
        az.plot_kde(np.array(coefs_exburnin)[:,52], values2=np.array(coefs_exburnin)[:,53],ax = axs[2,3],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,54], values2=np.array(coefs_exburnin)[:,55],ax = axs[3,0],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,56], values2=np.array(coefs_exburnin)[:,57],ax = axs[3,1],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,58], values2=np.array(coefs_exburnin)[:,59],ax = axs[3,2],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,60], values2=np.array(coefs_exburnin)[:,61],ax = axs[3,3])




        terms = [1,11,12,-1]

        fig, axs = plt.subplots(4,4)
#        az.plot_kde(np.array(coefs_exburnin)[:,terms[0]], values2=np.array(coefs_exburnin)[:,terms[0]],ax = axs[0,0])
        az.plot_kde(np.array(coefs_exburnin)[:,terms[0]], values2=np.array(coefs_exburnin)[:,terms[1]],ax = axs[0,1],contour_kwargs={"colors":None,"levels":3}, contourf_kwargs={"levels":3},bw=2000)
        az.plot_kde(np.array(coefs_exburnin)[:,terms[0]], values2=np.array(coefs_exburnin)[:,terms[2]],ax = axs[0,2],contour_kwargs={"colors":None,"levels":4}, contourf_kwargs={"levels":4},bw=40)
        az.plot_kde(np.array(coefs_exburnin)[:,terms[0]], values2=np.array(coefs_exburnin)[:,terms[3]],ax = axs[0,3],contour_kwargs={"colors":None,"levels":5}, contourf_kwargs={"levels":5},bw=60)
        az.plot_kde(np.array(coefs_exburnin)[:,terms[1]], values2=np.array(coefs_exburnin)[:,terms[0]],ax = axs[1,0],contour_kwargs={"colors":None,"levels":6}, contourf_kwargs={"levels":6},bw=8)
#        az.plot_kde(np.array(coefs_exburnin)[:,terms[1]], values2=np.array(coefs_exburnin)[:,terms[1]],ax = axs[1,1])
        az.plot_kde(np.array(coefs_exburnin)[:,terms[1]], values2=np.array(coefs_exburnin)[:,terms[2]],ax = axs[1,2],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,terms[1]], values2=np.array(coefs_exburnin)[:,terms[3]],ax = axs[1,3],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,terms[2]], values2=np.array(coefs_exburnin)[:,terms[0]],ax = axs[2,0],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,terms[2]], values2=np.array(coefs_exburnin)[:,terms[1]],ax = axs[2,1],contour_kwargs={"colors":None})
#        az.plot_kde(np.array(coefs_exburnin)[:,terms[2]], values2=np.array(coefs_exburnin)[:,terms[2]],ax = axs[2,2])
        az.plot_kde(np.array(coefs_exburnin)[:,terms[2]], values2=np.array(coefs_exburnin)[:,terms[3]],ax = axs[2,3],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,terms[3]], values2=np.array(coefs_exburnin)[:,terms[0]],ax = axs[3,0],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,terms[3]], values2=np.array(coefs_exburnin)[:,terms[1]],ax = axs[3,1],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,terms[3]], values2=np.array(coefs_exburnin)[:,terms[2]],ax = axs[3,2],contour_kwargs={"colors":None})


    def plot_mcmc_function(self,res_t=2000,res_x=1000,burnin_prop = 0.8,n_thinned=500):
        n_samples = len(self.sig_eps_sample)
        burnin = round(n_samples*burnin_prop)
        
        nn_res = [300,150]

        x_pts = np.linspace(self.xlim[0],self.xlim[1],nn_res[1])
        t_pts = np.linspace(self.tlim[0],self.tlim[1],nn_res[0])
        xx_nn, tt_nn = np.meshgrid(x_pts,t_pts)
        

        x_pts = np.linspace(self.xlim[0],self.xlim[1],res_x)
        t_pts = np.linspace(self.tlim[0],self.tlim[1],res_t)
        xx, tt = np.meshgrid(x_pts,t_pts)
        
        coefs_exburnin = np.array(self.coefs_sample[burnin:])
        sig_eps_exburnin = np.array(self.sig_eps_sample[burnin:])
        

        
        cheb_tt_nn = np.zeros([66,nn_res[0],nn_res[1]])
        cheb_xx_nn = np.zeros([66,nn_res[0],nn_res[1]])

        for i in range(66):
            cheb_tt_nn[i] = self.chebt.P(tt_nn,self.indices_t[i],tensor=False)
            cheb_xx_nn[i] = self.chebx.P(xx_nn,self.indices_x[i],tensor=False)
        
        
        thin_inds = np.linspace(0,len(sig_eps_exburnin)-1,n_thinned).astype(int)
        
        
        
        coefs_post = coefs_exburnin[thin_inds]
        
        coefs_est = np.mean(coefs_post,axis=0)
        



        
        post_Bi_samples = sum(coefs_post[:,i].reshape([-1,1,1])*np.reshape(cheb_xx_nn[i]*cheb_tt_nn[i],[1,nn_res[0],nn_res[1]]) for i in range(self.indices_t.shape[0]))
        if self.positive == True:
            post_Bi_samples = np.maximum(post_Bi_samples, 0*np.zeros_like(post_Bi_samples))

    
        post_Bi_mean = np.mean(post_Bi_samples,axis = 0)
        post_Bi_var = np.sqrt(np.var(post_Bi_samples,axis=0))

        #print('the max inferred function is', np.max(post_Bi_mean) )


        if self.kind=='exp biot' or self.kind=='exp flux':
            NN_solution = self.sess.run(tf.exp(self.u_eval(tt_nn, xx_nn, coefs_est*np.ones([tt_nn.size,1])))).reshape(nn_res)
        else:
            NN_solution = self.sess.run(self.u_eval(tt_nn, xx_nn, coefs_est*np.ones([tt_nn.size,1]))).reshape(nn_res)
        
        if self.kind=='exp biot' or self.kind=='direct biot':
            if self.csv_name == None:
                mi = self.flux_scale*np.min([np.min(post_Bi_mean*NN_solution),np.min(self.true_biot*self.true_sol)])
                ma = self.flux_scale*np.max([np.max(post_Bi_mean*NN_solution),np.max(self.true_biot*self.true_sol)])
            else:
                mi = self.flux_scale*np.min(post_Bi_mean*NN_solution)
                ma = self.flux_scale*np.max(post_Bi_mean*NN_solution)                    
        else:
            if self.csv_name == None:
                mi = self.flux_scale*np.min([np.min(post_Bi_mean),np.min(self.true_biot)])
                ma = self.flux_scale*np.max([np.max(post_Bi_mean),np.max(self.true_biot)])
            else:
                mi = self.flux_scale*np.min(post_Bi_mean)
                ma = self.flux_scale*np.max(post_Bi_mean)                    

        
        mav = np.max(self.flux_scale*post_Bi_var)
        
        if self.csv_name == None:                
            fig, axs = plt.subplots(3, 1)
            
            if self.csv_name == None:                
                tt_biot = np.linspace(self.tlim[0],self.tlim[1],self.true_sol.shape[0])
                xx_biot = np.linspace(self.xlim[0],self.xlim[1],self.true_sol.shape[1])
                tt_biot,xx_biot = np.meshgrid(tt_biot,xx_biot)
                if self.kind=='exp biot' or self.kind=='direct biot':
                    im = axs[0].pcolormesh(tt_biot, xx_biot, self.flux_scale*(self.true_biot*self.true_sol).T, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                else:
                    im = axs[0].pcolormesh(tt_biot, xx_biot, self.flux_scale*(self.true_biot).T, cmap=cm.coolwarm, vmin=mi, vmax=ma)
    
            if self.kind=='exp biot' or self.kind=='direct biot':
                mav = np.max(self.flux_scale*post_Bi_var*NN_solution)
                im = axs[1].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_mean*NN_solution, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                fig.colorbar(im, ax=[axs[0],axs[1]])
                im = axs[2].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_var*NN_solution, cmap=cm.coolwarm, vmin=0, vmax=mav)
                fig.colorbar(im, ax=axs[2])
            else:
                mav = np.max(self.flux_scale*post_Bi_var)
                im = axs[1].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_mean, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                fig.colorbar(im, ax=[axs[0],axs[1]])
                im = axs[2].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_var, cmap=cm.coolwarm, vmin=0, vmax=mav)
                fig.colorbar(im, ax=axs[2])
    
    
            plt.show()
            im = axs[0].title.set_text('True flux')
            im = axs[1].title.set_text('Inferred flux')
            im = axs[2].title.set_text('Uncertainty')
        else:
            fig, axs = plt.subplots(2, 1)
            
            if self.kind=='exp biot' or self.kind=='direct biot':
                mav = np.max(self.flux_scale*post_Bi_var*NN_solution)
                im = axs[0].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_mean*NN_solution, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                fig.colorbar(im, ax=[axs[0],axs[1]])
                im = axs[1].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_var*NN_solution, cmap=cm.coolwarm, vmin=0, vmax=mav)
                fig.colorbar(im, ax=axs[1])
            else:
                mav = np.max(self.flux_scale*post_Bi_var)
                im = axs[0].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_mean, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                fig.colorbar(im, ax=axs[0])
                im = axs[1].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_var, cmap=cm.coolwarm, vmin=0, vmax=mav)
                fig.colorbar(im, ax=axs[1])
    
    
            plt.show()
            im = axs[0].title.set_text('Inferred flux')
            im = axs[1].title.set_text('Uncertainty')

        #axs[0].colorbar()
        
        
        
        
        
        
        if self.kind=='exp biot' or self.kind=='direct biot':
            fig, axs = plt.subplots(3, 1)
            if self.csv_name == None:
                mi = self.flux_scale*np.min([np.min(post_Bi_mean),np.min(self.true_biot)])
                ma = self.flux_scale*np.max([np.max(post_Bi_mean),np.max(self.true_biot)])
                im = axs[0].pcolormesh(tt_biot, xx_biot, self.flux_scale*(self.true_biot).T, cmap=cm.coolwarm, vmin=mi, vmax=ma)        
            else:
                mi = self.flux_scale*np.min(post_Bi_mean)
                ma = self.flux_scale*np.max(post_Bi_mean)                    
            im = axs[1].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_mean, cmap=cm.coolwarm, vmin=mi, vmax=ma)
            fig.colorbar(im, ax=[axs[0],axs[1]])
            im = axs[2].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_var, cmap=cm.coolwarm, vmin=0, vmax=np.max(self.flux_scale*post_Bi_var))
            fig.colorbar(im, ax=axs[2])
            plt.show()
            im = axs[0].title.set_text('True biot')
            im = axs[1].title.set_text('Inferred biot')
            im = axs[2].title.set_text('Uncertainty')
        
        if self.kind=='exp biot' or self.kind=='direct biot':
            fig, axs = plt.subplots(3, 1)
            if self.csv_name == None:
                mi = self.flux_scale*np.min([np.min(post_Bi_mean),np.min(self.true_biot)])
                ma = self.flux_scale*np.max([np.max(post_Bi_mean),np.max(self.true_biot)])
            else:
                mi = self.flux_scale*np.min(post_Bi_mean)
                ma = self.flux_scale*np.max(post_Bi_mean)                    
            im = axs[0].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_samples[0], cmap=cm.coolwarm, vmin=mi, vmax=ma)        
            im = axs[1].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_samples[100], cmap=cm.coolwarm, vmin=mi, vmax=ma)        
            im = axs[2].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_samples[200], cmap=cm.coolwarm, vmin=mi, vmax=ma)        
            fig.colorbar(im, ax=[axs[0],axs[1],axs[2]])
            plt.show()
            im = axs[0].title.set_text('Some posterior samples')




        x_pts = np.linspace(self.xlim[0],self.xlim[1],nn_res[1])
        t_pts = np.linspace(self.tlim[0],self.tlim[1],nn_res[0])
        xx_nn, tt_nn = np.meshgrid(x_pts,t_pts)
        

        x_pts = np.linspace(self.xlim[0],self.xlim[1],res_x)
        t_pts = np.linspace(self.tlim[0],self.tlim[1],res_t)
        xx, tt = np.meshgrid(x_pts,t_pts)
        #coefs_exburnin = np.array(coefs_sample[4000:5100])
        #sig_eps_exburnin = np.array(sig_eps_sample[4000:5100])
        
        coefs_exburnin = np.array(self.coefs_sample[burnin:])
        sig_eps_exburnin = np.array(self.sig_eps_sample[burnin:])
        
        cheb_tt_nn = np.zeros([66,nn_res[0],nn_res[1]])
        cheb_xx_nn = np.zeros([66,nn_res[0],nn_res[1]])

        for i in range(66):
            cheb_tt_nn[i] = self.chebt.P(tt_nn,self.indices_t[i],tensor=False)
            cheb_xx_nn[i] = self.chebx.P(xx_nn,self.indices_x[i],tensor=False)
        
        thin_inds = np.linspace(0,len(sig_eps_exburnin)-1,n_thinned).astype(int)
        
        
        
        coefs_post = coefs_exburnin[thin_inds]
        
        coefs_est = np.mean(coefs_post,axis=0)
        



        
        post_Bi_samples = sum(coefs_post[:,i].reshape([-1,1,1])*np.reshape(cheb_xx_nn[i]*cheb_tt_nn[i],[1,nn_res[0],nn_res[1]]) for i in range(self.indices_t.shape[0]))
        if self.positive == True:
            post_Bi_samples = np.maximum(post_Bi_samples, 0*np.zeros_like(post_Bi_samples))

    
        post_Bi_mean = np.mean(post_Bi_samples,axis = 0)
        post_Bi_var = np.sqrt(np.var(post_Bi_samples,axis=0))



        if self.kind=='exp biot' or self.kind=='exp flux':
            NN_solution = self.sess.run(tf.exp(self.u_eval(tt_nn, xx_nn, coefs_est*np.ones([tt_nn.size,1])))).reshape(nn_res)
        else:
            NN_solution = self.sess.run(self.u_eval(tt_nn, xx_nn, coefs_est*np.ones([tt_nn.size,1]))).reshape(nn_res)
        
        if self.kind=='exp biot' or self.kind=='direct biot':
            if self.csv_name == None:
                mi = self.flux_scale*np.min([np.min(post_Bi_mean*NN_solution),np.min(self.true_biot*self.true_sol)])
                ma = self.flux_scale*np.max([np.max(post_Bi_mean*NN_solution),np.max(self.true_biot*self.true_sol)])
            else:
                mi = self.flux_scale*np.min(post_Bi_mean*NN_solution)
                ma = self.flux_scale*np.max(post_Bi_mean*NN_solution)                    
        else:
            if self.csv_name == None:
                mi = self.flux_scale*np.min([np.min(post_Bi_mean),np.min(self.true_biot)])
                ma = self.flux_scale*np.max([np.max(post_Bi_mean),np.max(self.true_biot)])
            else:
                mi = self.flux_scale*np.min(post_Bi_mean)
                ma = self.flux_scale*np.max(post_Bi_mean)                    

        
        if self.csv_name == None:                
            tt_biot = np.linspace(self.tlim[0],self.tlim[1],self.true_sol.shape[0])
            xx_biot = np.linspace(self.xlim[0],self.xlim[1],self.true_sol.shape[1])
            tt_biot,xx_biot = np.meshgrid(tt_biot,xx_biot)
    
    

        #axs[0].colorbar()
        
        
        
        
        
        
        
        # Plot 1 dimensional spatial cross sections
        t_cross = (self.tlim[1]-self.tlim[0])*np.array([0.2,0.4,0.6,0.8])
        plti = [0,1,2,3]
        
        nx_1d = res_x
        xx_1d = np.linspace(self.xlim[0],self.xlim[1],res_x)
        if self.csv_name == None:
            xx_1d2 = np.linspace(self.xlim[0],self.xlim[1],self.true_biot.shape[1])
        
        fig, axs = plt.subplots(1,4)
        
        if self.kind=='exp biot' or self.kind=='direct biot':        
            for j in range(len(plti)):
                sol_1dx = self.sess.run(self.u_eval(t_cross[j]*np.ones_like(xx_1d),xx_1d,coefs_est*(np.ones_like(xx_1d).reshape([-1,1])))).T
                post_Bi_samples_1dx = self.flux_scale*sum(coefs_post[:,i].reshape([-1,1])*np.reshape(self.chebx.P(xx_1d,self.indices_x[i],tensor=False)*self.chebt.P(t_cross[j],self.indices_t[i],tensor=False),[1,nx_1d]) for i in range(self.indices_t.shape[0]))
                if self.positive==True:
                    post_Bi_samples_1dx = np.maximum(post_Bi_samples_1dx, np.zeros_like(post_Bi_samples_1dx))
                #post_Bi_samples_1dx = np.minimum(post_Bi_samples_1dx, 200*np.ones_like(post_Bi_samples_1dx))
                post_Bi_median_1dx = np.mean(post_Bi_samples_1dx,axis = 0)
#                post_Bi_var = np.sqrt(np.var(post_Bi_samples_1dx,axis=0))
                ma = self.flux_scale*np.max(post_Bi_mean + 1.645*post_Bi_var)
                mi = self.flux_scale*np.min(post_Bi_mean - 1.645*post_Bi_var) 
                funci1 = np.percentile(post_Bi_samples_1dx,97.5,axis=0)# post_Bi_mean_1dx + post_Bi_var
                funci0 = np.percentile(post_Bi_samples_1dx,2.5,axis=0)# post_Bi_mean_1dx - post_Bi_var
                if self.csv_name == None:
                    biot_1d = self.true_biot[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]
                axs[plti[j]].plot(xx_1d,post_Bi_median_1dx)
                #axs[plti[j]].plot(xx_1d,post_Bi_samples_1dx[0],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                #axs[plti[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.3)],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                #axs[plti[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.6)],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                axs[plti[j]].fill_between(xx_1d, funci1, funci0, facecolor='blue', alpha=0.3)
                axs[plti[j]].title.set_text('Inferred biot at t =' + str(int(t_cross[j])))
                axs[plti[j]].set_ylim(mi,ma)
                if self.csv_name == None:
                    axs[plti[j]].plot(xx_1d2,biot_1d,label=r"True $Bi(x)$", color='tomato')
            fig, axs = plt.subplots(1,4)
            for j in range(len(plti)):
                sol_1dx = self.sess.run(self.u_eval(t_cross[j]*np.ones_like(xx_1d),xx_1d,coefs_est*(np.ones_like(xx_1d).reshape([-1,1])))).T
                post_Bi_samples_1dx = self.flux_scale*sum(coefs_post[:,i].reshape([-1,1])*np.reshape(self.chebx.P(xx_1d,self.indices_x[i],tensor=False)*self.chebt.P(t_cross[j],self.indices_t[i],tensor=False),[1,nx_1d]) for i in range(self.indices_t.shape[0]))
                if self.positive==True:
                    post_Bi_samples_1dx = np.maximum(post_Bi_samples_1dx, np.zeros_like(post_Bi_samples_1dx))
                #post_Bi_samples_1dx = np.minimum(post_Bi_samples_1dx, 200*np.ones_like(post_Bi_samples_1dx))
                post_Bi_median_1dx = np.mean(post_Bi_samples_1dx,axis = 0)
#                post_Bi_var = np.sqrt(np.var(post_Bi_samples_1dx,axis=0))
                ma = self.flux_scale*np.max(post_Bi_mean + 1.645*post_Bi_var)
                mi = self.flux_scale*np.min(post_Bi_mean - 1.645*post_Bi_var) 
                funci1 = np.percentile(post_Bi_samples_1dx,97.5,axis=0)# post_Bi_mean_1dx + post_Bi_var
                funci0 = np.percentile(post_Bi_samples_1dx,2.5,axis=0)# post_Bi_mean_1dx - post_Bi_var
                if self.csv_name == None:
                    biot_1d = self.true_biot[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]
                axs[plti[j]].plot(xx_1d,post_Bi_median_1dx)
                axs[plti[j]].plot(xx_1d,post_Bi_samples_1dx[0],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                axs[plti[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.3)],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                axs[plti[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.6)],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                #axs[plti[j]].fill_between(xx_1d, funci1, funci0, facecolor='blue', alpha=0.3)
                axs[plti[j]].title.set_text('Inferred biot at t =' + str(int(t_cross[j])))
                axs[plti[j]].set_ylim(mi,ma)
                if self.csv_name == None:
                    axs[plti[j]].plot(xx_1d2,biot_1d,label=r"True $Bi(x)$", color='tomato')
            fig, axs = plt.subplots(1,4)
            for j in range(len(plti)):
                sol_1dx = self.sess.run(self.u_eval(t_cross[j]*np.ones_like(xx_1d),xx_1d,coefs_est*(np.ones_like(xx_1d).reshape([-1,1])))).T
                post_Bi_samples_1dx = self.flux_scale*sol_1dx*sum(coefs_post[:,i].reshape([-1,1])*np.reshape(self.chebx.P(xx_1d,self.indices_x[i],tensor=False)*self.chebt.P(t_cross[j],self.indices_t[i],tensor=False),[1,nx_1d]) for i in range(self.indices_t.shape[0]))
                if self.positive==True:
                    post_Bi_samples_1dx = np.maximum(post_Bi_samples_1dx, np.zeros_like(post_Bi_samples_1dx))
                #post_Bi_samples_1dx = np.minimum(post_Bi_samples_1dx, 200*np.ones_like(post_Bi_samples_1dx))
                post_Bi_median_1dx = np.mean(post_Bi_samples_1dx,axis = 0)
#                post_Bi_var = np.sqrt(np.var(post_Bi_samples_1dx,axis=0))
                ma = self.flux_scale*np.max(post_Bi_mean*NN_solution + 1.645*post_Bi_var)
                mi = self.flux_scale*np.min(post_Bi_mean*NN_solution - 1.645*post_Bi_var) 
                funci1 = np.percentile(post_Bi_samples_1dx,97.5,axis=0)# post_Bi_mean_1dx + post_Bi_var
                funci0 = np.percentile(post_Bi_samples_1dx,2.5,axis=0)# post_Bi_mean_1dx - post_Bi_var
                if self.csv_name == None:
                    biot_1d = self.true_biot[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]*self.true_sol[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]
                axs[plti[j]].plot(xx_1d,post_Bi_median_1dx)
                #axs[plti[j]].plot(xx_1d,post_Bi_samples_1dx[0],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                #axs[plti[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.3)],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                #axs[plti[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.6)],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                axs[plti[j]].fill_between(xx_1d, funci1, funci0, facecolor='blue', alpha=0.3)
                axs[plti[j]].title.set_text('Inferred flux at t =' + str(int(t_cross[j])))
                axs[plti[j]].set_ylim(mi,ma)
                if self.csv_name == None:
                    axs[plti[j]].plot(xx_1d2,biot_1d,label=r"True flux", color='tomato')
            fig, axs = plt.subplots(1,4)
            for j in range(len(plti)):
                sol_1dx = self.sess.run(self.u_eval(t_cross[j]*np.ones_like(xx_1d),xx_1d,coefs_est*(np.ones_like(xx_1d).reshape([-1,1])))).T
                post_Bi_samples_1dx = self.flux_scale*sol_1dx*sum(coefs_post[:,i].reshape([-1,1])*np.reshape(self.chebx.P(xx_1d,self.indices_x[i],tensor=False)*self.chebt.P(t_cross[j],self.indices_t[i],tensor=False),[1,nx_1d]) for i in range(self.indices_t.shape[0]))
                if self.positive==True:
                    post_Bi_samples_1dx = np.maximum(post_Bi_samples_1dx, np.zeros_like(post_Bi_samples_1dx))
                #post_Bi_samples_1dx = np.minimum(post_Bi_samples_1dx, 200*np.ones_like(post_Bi_samples_1dx))
                post_Bi_median_1dx = np.mean(post_Bi_samples_1dx,axis = 0)
#                post_Bi_var = np.sqrt(np.var(post_Bi_samples_1dx,axis=0))
                ma = self.flux_scale*np.max(post_Bi_mean*NN_solution + 1.645*post_Bi_var)
                mi = self.flux_scale*np.min(post_Bi_mean*NN_solution - 1.645*post_Bi_var) 
                funci1 = np.percentile(post_Bi_samples_1dx,97.5,axis=0)# post_Bi_mean_1dx + post_Bi_var
                funci0 = np.percentile(post_Bi_samples_1dx,2.5,axis=0)# post_Bi_mean_1dx - post_Bi_var
                if self.csv_name == None:
                    biot_1d = self.true_biot[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]*self.true_sol[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]
                axs[plti[j]].plot(xx_1d,post_Bi_median_1dx)
                axs[plti[j]].plot(xx_1d,post_Bi_samples_1dx[0],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                axs[plti[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.3)],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                axs[plti[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.6)],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                #axs[plti[j]].fill_between(xx_1d, funci1, funci0, facecolor='blue', alpha=0.3)
                axs[plti[j]].title.set_text('Inferred flux at t =' + str(int(t_cross[j])))
                axs[plti[j]].set_ylim(mi,ma)
                if self.csv_name == None:
                    axs[plti[j]].plot(xx_1d2,biot_1d,label=r"True flux", color='tomato')

        else:
            for j in range(len(plti)):
                post_Bi_samples_1dx = self.flux_scale*sum(coefs_post[:,i].reshape([-1,1])*np.reshape(self.chebx.P(xx_1d,self.indices_x[i],tensor=False)*self.chebt.P(t_cross[j],self.indices_t[i],tensor=False),[1,nx_1d]) for i in range(self.indices_t.shape[0]))
                if self.positive==True:
                    post_Bi_samples_1dx = np.maximum(post_Bi_samples_1dx, np.zeros_like(post_Bi_samples_1dx))
                #post_Bi_samples_1dx = np.minimum(post_Bi_samples_1dx, 200*np.ones_like(post_Bi_samples_1dx))
                post_Bi_median_1dx = np.mean(post_Bi_samples_1dx,axis = 0)
 #               post_Bi_var = np.sqrt(np.var(post_Bi_samples_1dx,axis=0))
                ma = self.flux_scale*np.max(post_Bi_mean + 1.645*post_Bi_var) 
                mi = self.flux_scale*np.min(post_Bi_mean - 1.645*post_Bi_var) 
                funci1 = np.percentile(post_Bi_samples_1dx,97.5,axis=0)# post_Bi_mean_1dx + post_Bi_var
                funci0 = np.percentile(post_Bi_samples_1dx,2.5,axis=0)# post_Bi_mean_1dx - post_Bi_var
                if self.csv_name == None:
                    biot_1d = self.true_biot[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]
                axs[plti[j]].plot(xx_1d,post_Bi_median_1dx)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[0],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.3)],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.6)],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                axs[plti[j]].fill_between(xx_1d, funci1, funci0, facecolor='blue', alpha=0.3)
                axs[plti[j]].title.set_text('Inferred flux at t =' + str(int(t_cross[j])))
                axs[plti[j]].set_ylim(mi,ma)
                if self.csv_name == None:
                    axs[plti[j]].plot(xx_1d2,biot_1d,label=r"True $Bi(x)$", color='tomato')
            fig, axs = plt.subplots(1,4)
            for j in range(len(plti)):
                post_Bi_samples_1dx = self.flux_scale*sum(coefs_post[:,i].reshape([-1,1])*np.reshape(self.chebx.P(xx_1d,self.indices_x[i],tensor=False)*self.chebt.P(t_cross[j],self.indices_t[i],tensor=False),[1,nx_1d]) for i in range(self.indices_t.shape[0]))
                if self.positive==True:
                    post_Bi_samples_1dx = np.maximum(post_Bi_samples_1dx, np.zeros_like(post_Bi_samples_1dx))
                #post_Bi_samples_1dx = np.minimum(post_Bi_samples_1dx, 200*np.ones_like(post_Bi_samples_1dx))
                post_Bi_median_1dx = np.mean(post_Bi_samples_1dx,axis = 0)
 #               post_Bi_var = np.sqrt(np.var(post_Bi_samples_1dx,axis=0))
                ma = self.flux_scale*np.max(post_Bi_mean + 1.645*post_Bi_var) 
                mi = self.flux_scale*np.min(post_Bi_mean - 1.645*post_Bi_var) 
                funci1 = np.percentile(post_Bi_samples_1dx,97.5,axis=0)# post_Bi_mean_1dx + post_Bi_var
                funci0 = np.percentile(post_Bi_samples_1dx,2.5,axis=0)# post_Bi_mean_1dx - post_Bi_var
                if self.csv_name == None:
                    biot_1d = self.true_biot[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]
                axs[plti[j]].plot(xx_1d,post_Bi_median_1dx)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[0],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.3)],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.6)],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                axs[plti[j]].fill_between(xx_1d, funci1, funci0, facecolor='blue', alpha=0.3)
                axs[plti[j]].title.set_text('Inferred flux at t =' + str(int(t_cross[j])))
                axs[plti[j]].set_ylim(mi,ma)
                if self.csv_name == None:
                    axs[plti[j]].plot(xx_1d2,biot_1d,label=r"True $Bi(x)$", color='tomato')
        
        
    def plot_mcmc_fit(self,res_t=2000,res_x=1000,burnin_prop = 0.8,n_thinned=500,pars = [None]):        
        n_samples = len(self.sig_eps_sample)
        burnin = round(n_samples*burnin_prop)
        
        nn_res = [300,150]

        x_pts = np.linspace(self.xlim[0],self.xlim[1],nn_res[1])
        t_pts = np.linspace(self.tlim[0],self.tlim[1],nn_res[0])
        xx_nn, tt_nn = np.meshgrid(x_pts,t_pts)
        

        x_pts = np.linspace(self.xlim[0],self.xlim[1],res_x)
        t_pts = np.linspace(self.tlim[0],self.tlim[1],res_t)
        xx, tt = np.meshgrid(x_pts,t_pts)
        #coefs_exburnin = np.array(coefs_sample[4000:5100])
        #sig_eps_exburnin = np.array(sig_eps_sample[4000:5100])
        
        coefs_exburnin = np.array(self.coefs_sample[burnin:])
        sig_eps_exburnin = np.array(self.sig_eps_sample[burnin:])

        if pars[0] == None:
            pars = np.mean(coefs_exburnin,axis=0)

        if self.kind=='exp biot' or self.kind=='direct biot':          
            FD_solver = FD_biot(self.xlim[0],self.xlim[1],self.tlim[1],self.L,self.R,self.I,res_x-1,res_t-1,0.5,indices_t=self.indices_t,indices_x=self.indices_x,c1=self.c1,c2=self.c2,c3=self.c3,T0=self.tlim[0])
        else:
            FD_solver = FD_flux(self.xlim[0],self.xlim[1],self.tlim[1],self.L,self.R,self.I,res_x-1,res_t-1,0.5,indices_t=self.indices_t,indices_x=self.indices_x,c1=self.c1,c2=self.c2,c3=self.c3,T0=self.tlim[0])
            
        FD_solution = FD_solver.solve(pars)

        
        FD_pred = sp.interpolate.RectBivariateSpline(FD_solver.tmesh,FD_solver.xmesh, FD_solution.T).ev(self.t_dat.reshape([-1,1]),self.x_dat.reshape([-1,1]))

        
#        resid = np.array(self.z_dat)-self.sess.run(self.u_eval(self.t_dat,self.x_dat,pars.astype(np.float32)*np.ones_like(self.x_dat)))
        resid = np.array(self.z_dat)-FD_pred
        
        fig, axs = plt.subplots(2,1)
        axs[1].axhline(y=0., color='r', linestyle='-')
        axs[1].scatter(self.t_dat,resid,alpha=0.5,s=15)


        axs[0].axhline(y=0., color='r', linestyle='-')
        axs[0].scatter(self.x_dat,resid,alpha=0.5,s=15)

        # plots solutions
        n_thinned = 100
        
        cheb_tt_nn = np.zeros([66,nn_res[0],nn_res[1]])
        cheb_xx_nn = np.zeros([66,nn_res[0],nn_res[1]])

        for i in range(66):
            cheb_tt_nn[i] = self.chebt.P(tt_nn,self.indices_t[i],tensor=False)
            cheb_xx_nn[i] = self.chebx.P(xx_nn,self.indices_x[i],tensor=False)
        
        
        # plots biot numbers
#        n_thinned = n_samples
        
#        thin_dist = math.floor(len(sig_eps_exburnin)/n_thinned)
        thin_inds = np.linspace(0,len(sig_eps_exburnin)-1,n_thinned).astype(int)
        
#        math.floor(len(sig_eps_exburnin)/n_thinned)
        
        
        coefs_post = coefs_exburnin[thin_inds]
        
        coefs_est = np.mean(coefs_post,axis=0)
#        biot_coefs_est = sum(coefs_est[i].reshape([-1,1,1])*np.reshape(cheb_xx_nn[i]*cheb_tt_nn[i],[1,nn_res[0],nn_res[1]]) for i in range(self.indices_t.shape[0]))
        #chebx = cheb_poly([xlim[0]-0.1,xlim[0]+0.1])
        #chebt = cheb_poly([xlim[0]-0.4,xlim[0]+0.4])
        



        
        post_Bi_samples = sum(coefs_post[:,i].reshape([-1,1,1])*np.reshape(cheb_xx_nn[i]*cheb_tt_nn[i],[1,nn_res[0],nn_res[1]]) for i in range(self.indices_t.shape[0]))
        if self.positive == True:
            post_Bi_samples = np.maximum(post_Bi_samples, 0*np.zeros_like(post_Bi_samples))
#        post_Bi_samples = np.minimum(post_Bi_samples, 200*np.ones_like(post_Bi_samples))

    
        post_Bi_mean = np.mean(post_Bi_samples,axis = 0)
        thin_dist = math.floor(coefs_exburnin.shape[0]/n_thinned)
        
        
        coefs_post = coefs_exburnin[0:burnin:thin_dist]

        if self.kind=='exp biot' or self.kind=='direct biot':          
            FD_solver = FD_biot(self.xlim[0],self.xlim[1],self.tlim[1],self.L,self.R,self.I,res_x-1,res_t-1,0.5,cheb=True,indices_t=self.indices_t,indices_x=self.indices_x,bi=post_Bi_mean/self.flux_scale,c1=self.c1,c2=self.c2,c3=self.c3,T0=self.tlim[0])
        else:
            FD_solver = FD_flux(self.xlim[0],self.xlim[1],self.tlim[1],self.L,self.R,self.I,res_x-1,res_t-1,0.5,cheb=True,indices_t=self.indices_t,indices_x=self.indices_x,bi=post_Bi_mean/self.flux_scale,c1=self.c1,c2=self.c2,c3=self.c3,T0=self.tlim[0])
            
        FD_solution = FD_solver.solve(coefs_est)
        if self.kind=='exp biot' or self.kind=='exp flux':
            NN_solution = self.sess.run(tf.exp(self.u_eval(tt_nn, xx_nn, coefs_est*np.ones([tt_nn.size,1])))).reshape(nn_res)
        else:
            NN_solution = self.sess.run(self.u_eval(tt_nn, xx_nn, coefs_est*np.ones([tt_nn.size,1]))).reshape(nn_res)
        
        
        mi = self.data.dat_scale*np.min([np.min(FD_solution),np.min(NN_solution)])
        ma = self.data.dat_scale*np.max([np.max(FD_solution),np.max(NN_solution)])
        



        NN_solution = griddata(np.hstack([xx_nn.reshape([-1,1]),tt_nn.reshape([-1,1])]), NN_solution.reshape([-1,1]), (xx, tt), method='linear').reshape([res_t,res_x])

        fig, axs = plt.subplots(3, 1)
        
        im = axs[0].scatter(self.t_dat, self.x_dat, s=50, c=self.data.dat_scale*self.z_dat, vmin=mi, vmax=ma,  cmap=cm.coolwarm)
        im = axs[1].pcolormesh(tt, xx, self.data.dat_scale*NN_solution, cmap=cm.coolwarm, vmin=mi, vmax=ma)
        fig.colorbar(im, ax=[axs[0],axs[1],axs[2]])
        im = axs[2].axhline(y=0., color='r', linestyle='-')
        im = axs[2].scatter(self.t_dat,resid,alpha=0.5,s=15)
#        im = axs[2].pcolormesh(tt, xx, self.data.dat_scale*FD_solution[:,:round(res_t*time_cutoff)].T, cmap=cm.coolwarm, vmin=mi, vmax=ma)
#        im = axs[2].pcolormesh(tt, xx, FD_solution[:,:round(res_t*time_cutoff)].T - NN_solution, cmap=cm.coolwarm, vmin=-0.005, vmax=0.005)
#        fig.colorbar(im, ax=[axs[2]])
        
        plt.show()
        
#        mi2 = np.min([np.min(FD_solution-True_solution),np.min(NN_solution.T-True_solution[:,:round(res_t*time_cutoff)])])
#        ma2 = np.max([np.max(FD_solution-True_solution),np.max(NN_solution.T-True_solution[:,:round(res_t*time_cutoff)])])
        
#        fig, axs = plt.subplots(2, 1)
        
#        axs[0].pcolormesh(tt_plt, xx_plt, FD_solution[:,:round(nt_FD*time_cutoff)].T - U[:,:round(nt_FD*time_cutoff)].T, cmap=cm.coolwarm, vmin=mi2, vmax=ma2)
#        im = axs[1].pcolormesh(tt_plt, xx_plt, NN_solution - U[:,:round(nt_FD*time_cutoff)].T, cmap=cm.coolwarm, vmin=mi2, vmax=ma2)
        
#        fig.colorbar(im, ax=axs.ravel().tolist())
#        plt.show()
        
        mi3 = np.min(FD_solution.T - NN_solution)
        ma3 = np.max(FD_solution.T - NN_solution)
        
        fig,ax = plt.subplots(1)
        im = plt.pcolormesh(tt, xx, FD_solution.T - NN_solution, cmap=cm.coolwarm, vmin=mi3, vmax=ma3)
        fig.colorbar(im)

        # Plot 1 dimensional spatial cross sections
        t_cross = (self.tlim[1]-self.tlim[0])*np.array([0.,0.2,0.4,0.6,0.8,0.99])
        plti = [0,0,0,1,1,1]
        pltj = [0,1,2,0,1,2]
        
        xx_1d = np.linspace(self.xlim[0],self.xlim[1],res_x)


        fig, axs = plt.subplots(2,3)

        mas = np.max(NN_solution)*self.data.dat_scale
        mis = np.min(NN_solution)*self.data.dat_scale
                
        for j in range(6):
            t_loc = self.t_dat[np.argmin(np.abs(t_cross[j]-self.t_dat))]
            tdat_1dx = self.t_dat[self.t_dat==t_loc]
            xdat_1dx = self.x_dat[self.t_dat==t_loc]
            zdat_1dx = self.z_dat[self.t_dat==t_loc]*self.data.dat_scale
            sol_1dx = self.sess.run(self.u_eval(tdat_1dx[0]*np.ones_like(xx_1d),xx_1d,coefs_est*(np.ones_like(xx_1d).reshape([-1,1]))))*self.data.dat_scale
            axs[plti[j],pltj[j]].plot(xx_1d,sol_1dx)
            axs[plti[j],pltj[j]].scatter(xdat_1dx,zdat_1dx,label=r"True $Bi(x)$", color='red',alpha = 0.4)
            axs[plti[j],pltj[j]].title.set_text('PDE solution at t =' + str(int(t_cross[j])))
            axs[plti[j],pltj[j]].set_ylim(mis-5,mas+5)


    def plot_mcmc_est(self,res_t=2000,res_x=1000,burnin_prop = 0.8,n_thinned=500):


        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(3, 4, figure=fig)
        ax1 = fig.add_subplot(gs[0,:])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[1,1])
        ax4 = fig.add_subplot(gs[1,2])
        ax5 = fig.add_subplot(gs[1,3])
        ax6 = fig.add_subplot(gs[2,:])

        n_samples = len(self.sig_eps_sample)
        burnin = round(n_samples*burnin_prop)
        
        nn_res = [300,150]

        x_pts = np.linspace(self.xlim[0],self.xlim[1],nn_res[1])
        t_pts = np.linspace(self.tlim[0],self.tlim[1],nn_res[0])
        xx_nn, tt_nn = np.meshgrid(x_pts,t_pts)
        

        x_pts = np.linspace(self.xlim[0],self.xlim[1],res_x)
        t_pts = np.linspace(self.tlim[0],self.tlim[1],res_t)
        xx, tt = np.meshgrid(x_pts,t_pts)
        #coefs_exburnin = np.array(coefs_sample[4000:5100])
        #sig_eps_exburnin = np.array(sig_eps_sample[4000:5100])
        
        coefs_exburnin = np.array(self.coefs_sample[burnin:])
        sig_eps_exburnin = np.array(self.sig_eps_sample[burnin:])
        
        
        if self.kind=='exp biot' or self.kind=='direct biot':          
            FD_solver = FD_biot(self.xlim[0],self.xlim[1],self.tlim[1],self.L,self.R,self.I,res_x-1,res_t-1,0.5,indices_t=self.indices_t,indices_x=self.indices_x,c1=self.c1,c2=self.c2,c3=self.c3,T0=self.tlim[0])
        else:
            FD_solver = FD_flux(self.xlim[0],self.xlim[1],self.tlim[1],self.L,self.R,self.I,res_x-1,res_t-1,0.5,indices_t=self.indices_t,indices_x=self.indices_x,c1=self.c1,c2=self.c2,c3=self.c3,T0=self.tlim[0])
            
        FD_solution = FD_solver.solve(np.mean(coefs_exburnin,axis=0))

        
        FD_pred = sp.interpolate.RectBivariateSpline(FD_solver.tmesh,FD_solver.xmesh, FD_solution.T).ev(self.t_dat.reshape([-1,1]),self.x_dat.reshape([-1,1]))

        
#        resid = np.array(self.z_dat)-self.sess.run(self.u_eval(self.t_dat,self.x_dat,pars.astype(np.float32)*np.ones_like(self.x_dat)))
        resid = np.array(self.z_dat)-FD_pred
    
        fig, axs = plt.subplots(2,1)
        axs[1].axhline(y=0., color='r', linestyle='-')
        axs[1].scatter(self.t_dat,resid,alpha=0.5,s=15)

        ax6.axhline(y=0., color='r', linestyle='-')
        ax6.scatter(self.t_dat,resid,alpha=0.5,s=15)

        axs[0].axhline(y=0., color='r', linestyle='-')
        axs[0].scatter(self.x_dat,resid,alpha=0.5,s=15)

#        axs[0].axhline(y=0., color='r', linestyle='-')
#        axs[0].scatter(self.t_dat,np.array(self.z_dat)-FD_pred)
        
        #m,b = np.polyfit((np.array(z_dat)).flatten(),(np.array(z_dat)-sess.run(PDE(t_dat,x_dat,np.mean(coefs_exburnin,axis=0).astype(np.float32)))).flatten(), 1)
        '''
        # Trace plot
        fig, axs = plt.subplots(12,3)
        
        #coefs_exburnin = scale*coefs_exburnin
        axs[0,0].plot(sig_eps_exburnin)
        axs[1,0].plot(np.array(coefs_exburnin)[:,0])
        axs[2,0].plot(np.array(coefs_exburnin)[:,1])
        axs[3,0].plot(np.array(coefs_exburnin)[:,2])
        axs[4,0].plot(np.array(coefs_exburnin)[:,3])
        axs[5,0].plot(np.array(coefs_exburnin)[:,4])
        axs[6,0].plot(np.array(coefs_exburnin)[:,5])
        axs[7,0].plot(np.array(coefs_exburnin)[:,6])
        axs[8,0].plot(np.array(coefs_exburnin)[:,7])
        axs[9,0].plot(np.array(coefs_exburnin)[:,8])
        axs[10,0].plot(np.array(coefs_exburnin)[:,9])
        axs[11,0].plot(np.array(coefs_exburnin)[:,10])
        
        
        axs[1,1].plot(np.array(coefs_exburnin)[:,11])
        axs[2,1].plot(np.array(coefs_exburnin)[:,21])
        axs[3,1].plot(np.array(coefs_exburnin)[:,30])
        axs[4,1].plot(np.array(coefs_exburnin)[:,38])
        axs[5,1].plot(np.array(coefs_exburnin)[:,45])
        axs[6,1].plot(np.array(coefs_exburnin)[:,51])
        axs[7,1].plot(np.array(coefs_exburnin)[:,56])
        axs[8,1].plot(np.array(coefs_exburnin)[:,60])
        axs[9,1].plot(np.array(coefs_exburnin)[:,63])
        axs[10,1].plot(np.array(coefs_exburnin)[:,65])
        #axs[11,0].plot(np.array(coefs_exburnin)[:,66])
        
        axs[1,2].plot(np.array(coefs_exburnin)[:,0])
        axs[2,2].plot(np.array(coefs_exburnin)[:,1])
        axs[3,2].plot(np.array(coefs_exburnin)[:,11])
        axs[4,2].plot(np.array(coefs_exburnin)[:,2])
        axs[5,2].plot(np.array(coefs_exburnin)[:,21])
        axs[6,2].plot(np.array(coefs_exburnin)[:,12])
        axs[7,2].plot(np.array(coefs_exburnin)[:,3])
        axs[8,2].plot(np.array(coefs_exburnin)[:,13])
        axs[9,2].plot(np.array(coefs_exburnin)[:,22])
        axs[10,2].plot(np.array(coefs_exburnin)[:,30])
        
        fig, axs = plt.subplots(12,3)
        
        #coefs_exburnin = scale*coefs_exburnin
        axs[0,0].plot(sig_eps_exburnin)
        axs[1,0].plot(np.array(coefs_exburnin)[:,12])
        axs[2,0].plot(np.array(coefs_exburnin)[:,13])
        axs[3,0].plot(np.array(coefs_exburnin)[:,14])
        axs[4,0].plot(np.array(coefs_exburnin)[:,15])
        axs[5,0].plot(np.array(coefs_exburnin)[:,16])
        axs[6,0].plot(np.array(coefs_exburnin)[:,17])
        axs[7,0].plot(np.array(coefs_exburnin)[:,18])
        axs[8,0].plot(np.array(coefs_exburnin)[:,19])
        axs[9,0].plot(np.array(coefs_exburnin)[:,20])
        axs[10,0].plot(np.array(coefs_exburnin)[:,21])
        axs[11,0].plot(np.array(coefs_exburnin)[:,22])
        
        
        axs[1,1].plot(np.array(coefs_exburnin)[:,23])
        axs[2,1].plot(np.array(coefs_exburnin)[:,24])
        axs[3,1].plot(np.array(coefs_exburnin)[:,25])
        axs[4,1].plot(np.array(coefs_exburnin)[:,26])
        axs[5,1].plot(np.array(coefs_exburnin)[:,27])
        axs[6,1].plot(np.array(coefs_exburnin)[:,28])
        axs[7,1].plot(np.array(coefs_exburnin)[:,29])
        axs[8,1].plot(np.array(coefs_exburnin)[:,30])
        axs[9,1].plot(np.array(coefs_exburnin)[:,63])
        axs[10,1].plot(np.array(coefs_exburnin)[:,65])
        #axs[11,0].plot(np.array(coefs_exburnin)[:,66])
        
        axs[1,2].plot(np.array(coefs_exburnin)[:,31])
        axs[2,2].plot(np.array(coefs_exburnin)[:,32])
        axs[3,2].plot(np.array(coefs_exburnin)[:,33])
        axs[4,2].plot(np.array(coefs_exburnin)[:,34])
        axs[5,2].plot(np.array(coefs_exburnin)[:,35])
        axs[6,2].plot(np.array(coefs_exburnin)[:,36])
        axs[7,2].plot(np.array(coefs_exburnin)[:,37])
        axs[8,2].plot(np.array(coefs_exburnin)[:,38])
        axs[9,2].plot(np.array(coefs_exburnin)[:,39])
        axs[10,2].plot(np.array(coefs_exburnin)[:,40])
        
        
        fig, axs = plt.subplots(12,3)
        
        #coefs_exburnin = scale*coefs_exburnin
        axs[0,0].plot(sig_eps_exburnin)
        axs[1,0].plot(np.array(coefs_exburnin)[:,41])
        axs[2,0].plot(np.array(coefs_exburnin)[:,42])
        axs[3,0].plot(np.array(coefs_exburnin)[:,43])
        axs[4,0].plot(np.array(coefs_exburnin)[:,44])
        axs[5,0].plot(np.array(coefs_exburnin)[:,45])
        axs[6,0].plot(np.array(coefs_exburnin)[:,46])
        axs[7,0].plot(np.array(coefs_exburnin)[:,47])
        axs[8,0].plot(np.array(coefs_exburnin)[:,48])
        axs[9,0].plot(np.array(coefs_exburnin)[:,49])
        axs[10,0].plot(np.array(coefs_exburnin)[:,50])
        axs[11,0].plot(np.array(coefs_exburnin)[:,51])
        
        
        axs[1,1].plot(np.array(coefs_exburnin)[:,52])
        axs[2,1].plot(np.array(coefs_exburnin)[:,53])
        axs[3,1].plot(np.array(coefs_exburnin)[:,54])
        axs[4,1].plot(np.array(coefs_exburnin)[:,55])
        axs[5,1].plot(np.array(coefs_exburnin)[:,56])
        axs[6,1].plot(np.array(coefs_exburnin)[:,57])
        axs[7,1].plot(np.array(coefs_exburnin)[:,58])
        axs[8,1].plot(np.array(coefs_exburnin)[:,59])
        axs[9,1].plot(np.array(coefs_exburnin)[:,60])
        axs[10,1].plot(np.array(coefs_exburnin)[:,61])
        #axs[11,0].plot(np.array(coefs_exburnin)[:,66])
        
        axs[1,2].plot(np.array(coefs_exburnin)[:,62])
        axs[2,2].plot(np.array(coefs_exburnin)[:,63])
        axs[3,2].plot(np.array(coefs_exburnin)[:,64])
        axs[4,2].plot(np.array(coefs_exburnin)[:,65])
        
        
        # Marginal plots
        
        terms = [2,4,3,5]
        
        fig, axs = plt.subplots(4,4)
        az.plot_kde(np.array(coefs_exburnin)[:,0], values2=np.array(coefs_exburnin)[:,1],ax = axs[0,0])
        az.plot_kde(np.array(coefs_exburnin)[:,2], values2=np.array(coefs_exburnin)[:,3],ax = axs[0,1],contour_kwargs={"colors":None,"levels":3}, contourf_kwargs={"levels":3},bw=2000)
        az.plot_kde(np.array(coefs_exburnin)[:,4], values2=np.array(coefs_exburnin)[:,5],ax = axs[0,2],contour_kwargs={"colors":None,"levels":4}, contourf_kwargs={"levels":4},bw=40)
        az.plot_kde(np.array(coefs_exburnin)[:,6], values2=np.array(coefs_exburnin)[:,7],ax = axs[0,3],contour_kwargs={"colors":None,"levels":5}, contourf_kwargs={"levels":5},bw=60)
        az.plot_kde(np.array(coefs_exburnin)[:,8], values2=np.array(coefs_exburnin)[:,9],ax = axs[1,0],contour_kwargs={"colors":None,"levels":6}, contourf_kwargs={"levels":6},bw=8)
        az.plot_kde(np.array(coefs_exburnin)[:,10], values2=np.array(coefs_exburnin)[:,11],ax = axs[1,1])
        az.plot_kde(np.array(coefs_exburnin)[:,12], values2=np.array(coefs_exburnin)[:,13],ax = axs[1,2],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,14], values2=np.array(coefs_exburnin)[:,15],ax = axs[1,3],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,16], values2=np.array(coefs_exburnin)[:,17],ax = axs[2,0],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,18], values2=np.array(coefs_exburnin)[:,19],ax = axs[2,1],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,20], values2=np.array(coefs_exburnin)[:,21],ax = axs[2,2])
        az.plot_kde(np.array(coefs_exburnin)[:,22], values2=np.array(coefs_exburnin)[:,23],ax = axs[2,3],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,24], values2=np.array(coefs_exburnin)[:,25],ax = axs[3,0],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,26], values2=np.array(coefs_exburnin)[:,27],ax = axs[3,1],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,28], values2=np.array(coefs_exburnin)[:,29],ax = axs[3,2],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,30], values2=np.array(coefs_exburnin)[:,31],ax = axs[3,3])

        
        fig, axs = plt.subplots(4,4)
        az.plot_kde(np.array(coefs_exburnin)[:,62], values2=np.array(coefs_exburnin)[:,63],ax = axs[0,0])
        az.plot_kde(np.array(coefs_exburnin)[:,32], values2=np.array(coefs_exburnin)[:,33],ax = axs[0,1],contour_kwargs={"colors":None,"levels":3}, contourf_kwargs={"levels":3},bw=2000)
        az.plot_kde(np.array(coefs_exburnin)[:,34], values2=np.array(coefs_exburnin)[:,35],ax = axs[0,2],contour_kwargs={"colors":None,"levels":4}, contourf_kwargs={"levels":4},bw=40)
        az.plot_kde(np.array(coefs_exburnin)[:,36], values2=np.array(coefs_exburnin)[:,37],ax = axs[0,3],contour_kwargs={"colors":None,"levels":5}, contourf_kwargs={"levels":5},bw=60)
        az.plot_kde(np.array(coefs_exburnin)[:,38], values2=np.array(coefs_exburnin)[:,39],ax = axs[1,0],contour_kwargs={"colors":None,"levels":6}, contourf_kwargs={"levels":6},bw=8)
        az.plot_kde(np.array(coefs_exburnin)[:,40], values2=np.array(coefs_exburnin)[:,41],ax = axs[1,1])
        az.plot_kde(np.array(coefs_exburnin)[:,42], values2=np.array(coefs_exburnin)[:,43],ax = axs[1,2],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,44], values2=np.array(coefs_exburnin)[:,45],ax = axs[1,3],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,46], values2=np.array(coefs_exburnin)[:,47],ax = axs[2,0],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,48], values2=np.array(coefs_exburnin)[:,49],ax = axs[2,1],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,50], values2=np.array(coefs_exburnin)[:,51],ax = axs[2,2])
        az.plot_kde(np.array(coefs_exburnin)[:,52], values2=np.array(coefs_exburnin)[:,53],ax = axs[2,3],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,54], values2=np.array(coefs_exburnin)[:,55],ax = axs[3,0],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,56], values2=np.array(coefs_exburnin)[:,57],ax = axs[3,1],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,58], values2=np.array(coefs_exburnin)[:,59],ax = axs[3,2],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,60], values2=np.array(coefs_exburnin)[:,61],ax = axs[3,3])




        terms = [1,11,12,-1]

        fig, axs = plt.subplots(4,4)
#        az.plot_kde(np.array(coefs_exburnin)[:,terms[0]], values2=np.array(coefs_exburnin)[:,terms[0]],ax = axs[0,0])
        az.plot_kde(np.array(coefs_exburnin)[:,terms[0]], values2=np.array(coefs_exburnin)[:,terms[1]],ax = axs[0,1],contour_kwargs={"colors":None,"levels":3}, contourf_kwargs={"levels":3},bw=2000)
        az.plot_kde(np.array(coefs_exburnin)[:,terms[0]], values2=np.array(coefs_exburnin)[:,terms[2]],ax = axs[0,2],contour_kwargs={"colors":None,"levels":4}, contourf_kwargs={"levels":4},bw=40)
        az.plot_kde(np.array(coefs_exburnin)[:,terms[0]], values2=np.array(coefs_exburnin)[:,terms[3]],ax = axs[0,3],contour_kwargs={"colors":None,"levels":5}, contourf_kwargs={"levels":5},bw=60)
        az.plot_kde(np.array(coefs_exburnin)[:,terms[1]], values2=np.array(coefs_exburnin)[:,terms[0]],ax = axs[1,0],contour_kwargs={"colors":None,"levels":6}, contourf_kwargs={"levels":6},bw=8)
#        az.plot_kde(np.array(coefs_exburnin)[:,terms[1]], values2=np.array(coefs_exburnin)[:,terms[1]],ax = axs[1,1])
        az.plot_kde(np.array(coefs_exburnin)[:,terms[1]], values2=np.array(coefs_exburnin)[:,terms[2]],ax = axs[1,2],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,terms[1]], values2=np.array(coefs_exburnin)[:,terms[3]],ax = axs[1,3],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,terms[2]], values2=np.array(coefs_exburnin)[:,terms[0]],ax = axs[2,0],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,terms[2]], values2=np.array(coefs_exburnin)[:,terms[1]],ax = axs[2,1],contour_kwargs={"colors":None})
#        az.plot_kde(np.array(coefs_exburnin)[:,terms[2]], values2=np.array(coefs_exburnin)[:,terms[2]],ax = axs[2,2])
        az.plot_kde(np.array(coefs_exburnin)[:,terms[2]], values2=np.array(coefs_exburnin)[:,terms[3]],ax = axs[2,3],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,terms[3]], values2=np.array(coefs_exburnin)[:,terms[0]],ax = axs[3,0],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,terms[3]], values2=np.array(coefs_exburnin)[:,terms[1]],ax = axs[3,1],contour_kwargs={"colors":None})
        az.plot_kde(np.array(coefs_exburnin)[:,terms[3]], values2=np.array(coefs_exburnin)[:,terms[2]],ax = axs[3,2],contour_kwargs={"colors":None})
#        az.plot_kde(np.array(coefs_exburnin)[:,terms[3]], values2=np.array(coefs_exburnin)[:,terms[3]],ax = axs[3,3])

#        fig, axs = plt.subplots(4,4)
        
#        fig, axs = plt.subplots(1,1)
#        az.plot_kde(np.array(sig_eps_exburnin))
        '''
        '''
        axs[0,0].scatter(np.array(coefs_exburnin)[:,terms[0]],np.array(coefs_exburnin)[:,terms[0]],alpha=0.01)
        axs[0,1].scatter(np.array(coefs_exburnin)[:,terms[0]],np.array(coefs_exburnin)[:,terms[1]],alpha=0.01)
        axs[0,2].scatter(np.array(coefs_exburnin)[:,terms[0]],np.array(coefs_exburnin)[:,terms[2]],alpha=0.01)
        axs[0,3].scatter(np.array(coefs_exburnin)[:,terms[0]],np.array(coefs_exburnin)[:,terms[3]],alpha=0.01)
        axs[1,0].scatter(np.array(coefs_exburnin)[:,terms[1]],np.array(coefs_exburnin)[:,terms[0]],alpha=0.01)
        axs[1,1].scatter(np.array(coefs_exburnin)[:,terms[1]],np.array(coefs_exburnin)[:,terms[1]],alpha=0.01)
        axs[1,2].scatter(np.array(coefs_exburnin)[:,terms[1]],np.array(coefs_exburnin)[:,terms[2]],alpha=0.01)
        axs[1,3].scatter(np.array(coefs_exburnin)[:,terms[1]],np.array(coefs_exburnin)[:,terms[3]],alpha=0.01)
        axs[2,0].scatter(np.array(coefs_exburnin)[:,terms[2]],np.array(coefs_exburnin)[:,terms[0]],alpha=0.01)
        axs[2,1].scatter(np.array(coefs_exburnin)[:,terms[2]],np.array(coefs_exburnin)[:,terms[1]],alpha=0.01)
        axs[2,2].scatter(np.array(coefs_exburnin)[:,terms[2]],np.array(coefs_exburnin)[:,terms[2]],alpha=0.01)
        axs[2,3].scatter(np.array(coefs_exburnin)[:,terms[2]],np.array(coefs_exburnin)[:,terms[3]],alpha=0.01)
        axs[3,0].scatter(np.array(coefs_exburnin)[:,terms[3]],np.array(coefs_exburnin)[:,terms[0]],alpha=0.01)
        axs[3,1].scatter(np.array(coefs_exburnin)[:,terms[3]],np.array(coefs_exburnin)[:,terms[1]],alpha=0.01)
        axs[3,2].scatter(np.array(coefs_exburnin)[:,terms[3]],np.array(coefs_exburnin)[:,terms[2]],alpha=0.01)
        axs[3,3].scatter(np.array(coefs_exburnin)[:,terms[3]],np.array(coefs_exburnin)[:,terms[3]],alpha=0.01)
        '''        
        
        
        cheb_tt_nn = np.zeros([66,nn_res[0],nn_res[1]])
        cheb_xx_nn = np.zeros([66,nn_res[0],nn_res[1]])

        for i in range(66):
            cheb_tt_nn[i] = self.chebt.P(tt_nn,self.indices_t[i],tensor=False)
            cheb_xx_nn[i] = self.chebx.P(xx_nn,self.indices_x[i],tensor=False)
        
        
        # plots biot numbers
#        n_thinned = n_samples
        
#        thin_dist = math.floor(len(sig_eps_exburnin)/n_thinned)
        thin_inds = np.linspace(0,len(sig_eps_exburnin)-1,n_thinned).astype(int)
        
#        math.floor(len(sig_eps_exburnin)/n_thinned)
        
        
        coefs_post = coefs_exburnin[thin_inds]
        sig_eps_post = sig_eps_exburnin[thin_inds]
        
        coefs_est = np.mean(coefs_post,axis=0)
#        biot_coefs_est = sum(coefs_est[i].reshape([-1,1,1])*np.reshape(cheb_xx_nn[i]*cheb_tt_nn[i],[1,nn_res[0],nn_res[1]]) for i in range(self.indices_t.shape[0]))
        #chebx = cheb_poly([xlim[0]-0.1,xlim[0]+0.1])
        #chebt = cheb_poly([xlim[0]-0.4,xlim[0]+0.4])
        



        
        post_Bi_samples = sum(coefs_post[:,i].reshape([-1,1,1])*np.reshape(cheb_xx_nn[i]*cheb_tt_nn[i],[1,nn_res[0],nn_res[1]]) for i in range(self.indices_t.shape[0]))
        if self.positive == True:
            post_Bi_samples = np.maximum(post_Bi_samples, 0*np.zeros_like(post_Bi_samples))
#        post_Bi_samples = np.minimum(post_Bi_samples, 200*np.ones_like(post_Bi_samples))

    
        post_Bi_mean = np.mean(post_Bi_samples,axis = 0)
        post_Bi_var = np.sqrt(np.var(post_Bi_samples,axis=0))

        #print('the max inferred function is', np.max(post_Bi_mean) )


        if self.kind=='exp biot' or self.kind=='direct biot':          
            FD_solver = FD_biot(self.xlim[0],self.xlim[1],self.tlim[1],self.L,self.R,self.I,res_x-1,res_t-1,0.5,cheb=True,indices_t=self.indices_t,indices_x=self.indices_x,bi=post_Bi_mean/self.flux_scale,c1=self.c1,c2=self.c2,c3=self.c3,T0=self.tlim[0])
        else:
            FD_solver = FD_flux(self.xlim[0],self.xlim[1],self.tlim[1],self.L,self.R,self.I,res_x-1,res_t-1,0.5,cheb=True,indices_t=self.indices_t,indices_x=self.indices_x,bi=post_Bi_mean/self.flux_scale,c1=self.c1,c2=self.c2,c3=self.c3,T0=self.tlim[0])
            
        FD_solution = FD_solver.solve(coefs_est)
        if self.kind=='exp biot' or self.kind=='exp flux':
            NN_solution = self.sess.run(tf.exp(self.u_eval(tt_nn, xx_nn, coefs_est*np.ones([tt_nn.size,1])))).reshape(nn_res)
        else:
            NN_solution = self.sess.run(self.u_eval(tt_nn, xx_nn, coefs_est*np.ones([tt_nn.size,1]))).reshape(nn_res)
        
        if self.kind=='exp biot' or self.kind=='direct biot':
            if self.csv_name == None:
                mi = self.flux_scale*np.min([np.min(post_Bi_mean*NN_solution),np.min(self.true_biot*self.true_sol)])
                ma = self.flux_scale*np.max([np.max(post_Bi_mean*NN_solution),np.max(self.true_biot*self.true_sol)])
            else:
                mi = self.flux_scale*np.min(post_Bi_mean*NN_solution)
                ma = self.flux_scale*np.max(post_Bi_mean*NN_solution)                    
        else:
            if self.csv_name == None:
                mi = self.flux_scale*np.min([np.min(post_Bi_mean),np.min(self.true_biot)])
                ma = self.flux_scale*np.max([np.max(post_Bi_mean),np.max(self.true_biot)])
            else:
                mi = self.flux_scale*np.min(post_Bi_mean)
                ma = self.flux_scale*np.max(post_Bi_mean)                    

        
        miv = np.min(self.flux_scale*post_Bi_var)
        mav = np.max(self.flux_scale*post_Bi_var)
        
        if self.csv_name == None:                
            fig, axs = plt.subplots(3, 1)
            
            if self.csv_name == None:                
                tt_biot = np.linspace(self.tlim[0],self.tlim[1],self.true_sol.shape[0])
                xx_biot = np.linspace(self.xlim[0],self.xlim[1],self.true_sol.shape[1])
                tt_biot,xx_biot = np.meshgrid(tt_biot,xx_biot)
                if self.kind=='exp biot' or self.kind=='direct biot':
                    im = axs[0].pcolormesh(tt_biot, xx_biot, self.flux_scale*(self.true_biot*self.true_sol).T, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                else:
                    im = axs[0].pcolormesh(tt_biot, xx_biot, self.flux_scale*(self.true_biot).T, cmap=cm.coolwarm, vmin=mi, vmax=ma)
    
            if self.kind=='exp biot' or self.kind=='direct biot':
                miv = np.min(self.flux_scale*post_Bi_var*NN_solution)
                mav = np.max(self.flux_scale*post_Bi_var*NN_solution)
                im = axs[1].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_mean*NN_solution, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                fig.colorbar(im, ax=[axs[0],axs[1]])
                im = axs[2].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_var*NN_solution, cmap=cm.coolwarm, vmin=0, vmax=mav)
                fig.colorbar(im, ax=axs[2])
            else:
                miv = np.min(self.flux_scale*post_Bi_var)
                mav = np.max(self.flux_scale*post_Bi_var)
                im = axs[1].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_mean, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                ax1.pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_mean, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                fig.colorbar(im, ax=[axs[0],axs[1]])
                im = axs[2].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_var, cmap=cm.coolwarm, vmin=0, vmax=mav)
                fig.colorbar(im, ax=axs[2])
    
    
            plt.show()
            im = axs[0].title.set_text('True flux')
            im = axs[1].title.set_text('Inferred flux')
            im = axs[2].title.set_text('Uncertainty')
        else:
            fig, axs = plt.subplots(2, 1)
            
            if self.kind=='exp biot' or self.kind=='direct biot':
                miv = np.min(self.flux_scale*post_Bi_var*NN_solution)
                mav = np.max(self.flux_scale*post_Bi_var*NN_solution)
                im = axs[0].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_mean*NN_solution, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                fig.colorbar(im, ax=[axs[0],axs[1]])
                im = axs[1].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_var*NN_solution, cmap=cm.coolwarm, vmin=0, vmax=mav)
                fig.colorbar(im, ax=axs[1])
            else:
                miv = np.min(self.flux_scale*post_Bi_var)
                mav = np.max(self.flux_scale*post_Bi_var)
                im = axs[0].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_mean, cmap=cm.coolwarm, vmin=mi, vmax=ma)
                fig.colorbar(im, ax=axs[0])
                im = axs[1].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_var, cmap=cm.coolwarm, vmin=0, vmax=mav)
                fig.colorbar(im, ax=axs[1])
    
    
            plt.show()
            im = axs[0].title.set_text('Inferred flux')
            im = axs[1].title.set_text('Uncertainty')

        #axs[0].colorbar()
        
        
        
        
        
        
        if self.kind=='exp biot' or self.kind=='direct biot':
            fig, axs = plt.subplots(3, 1)
            if self.csv_name == None:
                mi = self.flux_scale*np.min([np.min(post_Bi_mean),np.min(self.true_biot)])
                ma = self.flux_scale*np.max([np.max(post_Bi_mean),np.max(self.true_biot)])
                im = axs[0].pcolormesh(tt_biot, xx_biot, self.flux_scale*(self.true_biot).T, cmap=cm.coolwarm, vmin=mi, vmax=ma)        
            else:
                mi = self.flux_scale*np.min(post_Bi_mean)
                ma = self.flux_scale*np.max(post_Bi_mean)                    
            im = axs[1].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_mean, cmap=cm.coolwarm, vmin=mi, vmax=ma)
            ax1.pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_mean, cmap=cm.coolwarm, vmin=mi, vmax=ma)
            fig.colorbar(im, ax=[axs[0],axs[1]])
            im = axs[2].pcolormesh(tt_nn, xx_nn, self.flux_scale*post_Bi_var, cmap=cm.coolwarm, vmin=0, vmax=np.max(self.flux_scale*post_Bi_var))
            fig.colorbar(im, ax=axs[2])
            plt.show()
            im = axs[0].title.set_text('True biot')
            im = axs[1].title.set_text('Inferred biot')
            im = axs[2].title.set_text('Uncertainty')
        
        #coefs = np.mean(coefs_post,axis=0)
        
        # Plot 1 dimensional spatial cross sections
        t_cross = (self.tlim[1]-self.tlim[0])*np.array([0.,0.2,0.4,0.6,0.8,0.99])
        plti = [0,0,0,1,1,1]
        pltj = [0,1,2,0,1,2]
        
        nx_1d = res_x
        xx_1d = np.linspace(self.xlim[0],self.xlim[1],res_x)
        if self.csv_name == None:
            xx_1d2 = np.linspace(self.xlim[0],self.xlim[1],self.true_biot.shape[1])
        
        '''
        fig, axs = plt.subplots(2,3)
        
        for j in range(6):
            post_Bi_samples_1dx = sum(coefs_post[:,i].reshape([-1,1])*np.reshape(chebx.P(xx_1d,indices_x[i],tensor=False)*chebt.P(t_cross[j],indices_t[i],tensor=False),[1,nx_1d]) for i in range(indices_t.shape[0]))
            post_Bi_samples_1dx = np.maximum(post_Bi_samples_1dx, np.zeros_like(post_Bi_samples_1dx))
            post_Bi_samples_1dx = np.minimum(post_Bi_samples_1dx, 200*np.ones_like(post_Bi_samples_1dx))
            post_Bi_mean_1dx = np.mean(post_Bi_samples_1dx,axis = 0)
            post_Bi_sd = np.sqrt(np.var(post_Bi_samples_1dx,axis=0))
            funci1 = post_Bi_mean_1dx + 1.645*post_Bi_sd
            funci0 = post_Bi_mean_1dx - 1.645*post_Bi_sd
            funci0 = np.maximum(funci0, 0*np.ones_like(funci0))
            biot_1d = biot[round(t_cross[j]/3*(nt_FD-1) + 1)]
            axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_mean_1dx)
            axs[plti[j],pltj[j]].fill_between(xx_1d, funci1, funci0, facecolor='blue', alpha=0.3)
            axs[plti[j],pltj[j]].plot(xx_1d,biot_1d,label=r"True $Bi(x)$", color='tomato')
        
        
        '''
        fig, axs = plt.subplots(2,3)
        
        if self.kind=='exp biot' or self.kind=='direct biot':
            for j in range(6):
                sol_1dx = self.sess.run(self.u_eval(t_cross[j]*np.ones_like(xx_1d),xx_1d,coefs_est*(np.ones_like(xx_1d).reshape([-1,1])))).T
                post_Bi_samples_1dx = self.flux_scale*sol_1dx*sum(coefs_post[:,i].reshape([-1,1])*np.reshape(self.chebx.P(xx_1d,self.indices_x[i],tensor=False)*self.chebt.P(t_cross[j],self.indices_t[i],tensor=False),[1,nx_1d]) for i in range(self.indices_t.shape[0]))
                if self.positive==True:
                    post_Bi_samples_1dx = np.maximum(post_Bi_samples_1dx, np.zeros_like(post_Bi_samples_1dx))
                #post_Bi_samples_1dx = np.minimum(post_Bi_samples_1dx, 200*np.ones_like(post_Bi_samples_1dx))
                post_Bi_median_1dx = np.mean(post_Bi_samples_1dx,axis = 0)
#                post_Bi_var = np.sqrt(np.var(post_Bi_samples_1dx,axis=0))
                ma = self.flux_scale*np.max(post_Bi_mean*NN_solution + 1.645*post_Bi_var)
                mi = self.flux_scale*np.min(post_Bi_mean*NN_solution - 1.645*post_Bi_var) 
                funci1 = np.percentile(post_Bi_samples_1dx,97.5,axis=0)# post_Bi_mean_1dx + post_Bi_var
                funci0 = np.percentile(post_Bi_samples_1dx,2.5,axis=0)# post_Bi_mean_1dx - post_Bi_var
                if self.csv_name == None:
                    biot_1d = self.true_biot[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]*self.true_sol[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]
                axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_median_1dx)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[0],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.3)],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.6)],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                axs[plti[j],pltj[j]].fill_between(xx_1d, funci1, funci0, facecolor='blue', alpha=0.3)
                axs[plti[j],pltj[j]].title.set_text('Inferred flux at t =' + str(int(t_cross[j])))
                axs[plti[j],pltj[j]].set_ylim(mi,ma)
                if self.csv_name == None:
                    axs[plti[j],pltj[j]].plot(xx_1d2,biot_1d,label=r"True flux", color='tomato')
            fig, axs = plt.subplots(2,3)
            for j in range(6):
                sol_1dx = self.sess.run(self.u_eval(t_cross[j]*np.ones_like(xx_1d),xx_1d,coefs_est*(np.ones_like(xx_1d).reshape([-1,1])))).T
                post_Bi_samples_1dx = self.flux_scale*sum(coefs_post[:,i].reshape([-1,1])*np.reshape(self.chebx.P(xx_1d,self.indices_x[i],tensor=False)*self.chebt.P(t_cross[j],self.indices_t[i],tensor=False),[1,nx_1d]) for i in range(self.indices_t.shape[0]))
                if self.positive==True:
                    post_Bi_samples_1dx = np.maximum(post_Bi_samples_1dx, np.zeros_like(post_Bi_samples_1dx))
                #post_Bi_samples_1dx = np.minimum(post_Bi_samples_1dx, 200*np.ones_like(post_Bi_samples_1dx))
                post_Bi_median_1dx = np.mean(post_Bi_samples_1dx,axis = 0)
#                post_Bi_var = np.sqrt(np.var(post_Bi_samples_1dx,axis=0))
                ma = self.flux_scale*np.max(post_Bi_mean + 1.645*post_Bi_var)
                mi = self.flux_scale*np.min(post_Bi_mean - 1.645*post_Bi_var) 
                funci1 = np.percentile(post_Bi_samples_1dx,97.5,axis=0)# post_Bi_mean_1dx + post_Bi_var
                funci0 = np.percentile(post_Bi_samples_1dx,2.5,axis=0)# post_Bi_mean_1dx - post_Bi_var
                if self.csv_name == None:
                    biot_1d = self.true_biot[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]
                axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_median_1dx)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[0],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.3)],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.6)],label=r"True $Bi(x)$", color='green',alpha = 0.5)
                axs[plti[j],pltj[j]].fill_between(xx_1d, funci1, funci0, facecolor='blue', alpha=0.3)
                axs[plti[j],pltj[j]].title.set_text('Inferred biot at t =' + str(int(t_cross[j])))
                axs[plti[j],pltj[j]].set_ylim(mi,ma)
                if self.csv_name == None:
                    axs[plti[j],pltj[j]].plot(xx_1d2,biot_1d,label=r"True $Bi(x)$", color='tomato')
        else:
            for j in range(6):
                post_Bi_samples_1dx = self.flux_scale*sum(coefs_post[:,i].reshape([-1,1])*np.reshape(self.chebx.P(xx_1d,self.indices_x[i],tensor=False)*self.chebt.P(t_cross[j],self.indices_t[i],tensor=False),[1,nx_1d]) for i in range(self.indices_t.shape[0]))
                if self.positive==True:
                    post_Bi_samples_1dx = np.maximum(post_Bi_samples_1dx, np.zeros_like(post_Bi_samples_1dx))
                #post_Bi_samples_1dx = np.minimum(post_Bi_samples_1dx, 200*np.ones_like(post_Bi_samples_1dx))
                post_Bi_median_1dx = np.mean(post_Bi_samples_1dx,axis = 0)
 #               post_Bi_var = np.sqrt(np.var(post_Bi_samples_1dx,axis=0))
                ma = self.flux_scale*np.max(post_Bi_mean + 1.645*post_Bi_var) 
                mi = self.flux_scale*np.min(post_Bi_mean - 1.645*post_Bi_var) 
                funci1 = np.percentile(post_Bi_samples_1dx,97.5,axis=0)# post_Bi_mean_1dx + post_Bi_var
                funci0 = np.percentile(post_Bi_samples_1dx,2.5,axis=0)# post_Bi_mean_1dx - post_Bi_var
                if self.csv_name == None:
                    biot_1d = self.true_biot[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]
                axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_median_1dx)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[0],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.3)],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                #axs[plti[j],pltj[j]].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.6)],label=r"True $Bi(x)$", color='green',alpha = 0.1)
                axs[plti[j],pltj[j]].fill_between(xx_1d, funci1, funci0, facecolor='blue', alpha=0.3)
                axs[plti[j],pltj[j]].title.set_text('Inferred flux at t =' + str(int(t_cross[j])))
                axs[plti[j],pltj[j]].set_ylim(mi,ma)
                if self.csv_name == None:
                    axs[plti[j],pltj[j]].plot(xx_1d2,biot_1d,label=r"True $Bi(x)$", color='tomato')
        

        t_cross = (self.tlim[1]-self.tlim[0])*np.array([0.2,0.4,0.6,0.8])
        axes = [ax2,ax3,ax4,ax5]

        for j in range(4):
            #sol_1dx = self.sess.run(self.u_eval(t_cross[j]*np.ones_like(xx_1d),xx_1d,coefs_est*(np.ones_like(xx_1d).reshape([-1,1])))).T
            post_Bi_samples_1dx = self.flux_scale*sum(coefs_post[:,i].reshape([-1,1])*np.reshape(self.chebx.P(xx_1d,self.indices_x[i],tensor=False)*self.chebt.P(t_cross[j],self.indices_t[i],tensor=False),[1,nx_1d]) for i in range(self.indices_t.shape[0]))
            if self.positive==True:
                post_Bi_samples_1dx = np.maximum(post_Bi_samples_1dx, np.zeros_like(post_Bi_samples_1dx))
            #post_Bi_samples_1dx = np.minimum(post_Bi_samples_1dx, 200*np.ones_like(post_Bi_samples_1dx))
            post_Bi_median_1dx = np.mean(post_Bi_samples_1dx,axis = 0)
#                post_Bi_var = np.sqrt(np.var(post_Bi_samples_1dx,axis=0))
            ma = self.flux_scale*np.max(post_Bi_mean + 1.645*post_Bi_var)
            mi = self.flux_scale*np.min(post_Bi_mean - 1.645*post_Bi_var) 
            funci1 = np.percentile(post_Bi_samples_1dx,97.5,axis=0)# post_Bi_mean_1dx + post_Bi_var
            funci0 = np.percentile(post_Bi_samples_1dx,2.5,axis=0)# post_Bi_mean_1dx - post_Bi_var
            if self.csv_name == None:
                biot_1d = self.true_biot[round(t_cross[j]/(self.tlim[1]-self.tlim[0])*(self.true_biot.shape[0]-1))]
            axes[j].plot(xx_1d,post_Bi_median_1dx)
            axes[j].plot(xx_1d,post_Bi_samples_1dx[0],label=r"True $Bi(x)$", color='green',alpha = 0.5)
            axes[j].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.3)],label=r"True $Bi(x)$", color='green',alpha = 0.5)
            axes[j].plot(xx_1d,post_Bi_samples_1dx[round(n_thinned*0.6)],label=r"True $Bi(x)$", color='green',alpha = 0.5)
            #axs[plti[j],pltj[j]].fill_between(xx_1d, funci1, funci0, facecolor='blue', alpha=0.3)
            if self.kind=='exp biot' or self.kind=='direct biot':
                axes[j].set_ylim(mi,ma)
                if self.csv_name == None:
                    axes[j].plot(xx_1d2,biot_1d,label=r"True $Bi(x)$", color='tomato')
            else:
                axes[j].set_ylim(mi,ma)
                if self.csv_name == None:
                    axs[j].plot(xx_1d2,biot_1d,label=r"True flux", color='tomato')




        # Plot 1d temporal cross sections
        x_cross = [0.4,0.5,0.5,0.7,0.8,0.9];
        plti = [0,0,0,1,1,1]
        pltj = [0,1,2,0,1,2]
        
        nt_1d = res_t
        tt_1d = np.linspace(self.tlim[0],self.tlim[1],res_t)
        
#        fig, axs = plt.subplots(2,3)
        
        
        '''
        for j in range(6):
            post_Bi_samples_1dt = sum(coefs_post[:,i].reshape([-1,1])*np.reshape(self.chebx.P(tt_1d,self.indices_x[i],tensor=False)*self.chebt.P(tt_1d,self.indices_t[i],tensor=False),[1,nt_1d]) for i in range(self.indices_t.shape[0]))
            #post_Bi_samples_1dt = np.maximum(post_Bi_samples_1dt, np.zeros_like(post_Bi_samples_1dt))
            #post_Bi_samples_1dt = np.minimum(post_Bi_samples_1dt, 200*np.ones_like(post_Bi_samples_1dt))
            post_Bi_mean_1dt = np.mean(post_Bi_samples_1dt,axis = 0)
            post_Bi_sd = np.sqrt(np.var(post_Bi_samples_1dt,axis=0))
            funci1 = post_Bi_mean_1dt + 1.645*post_Bi_sd
            funci0 = post_Bi_mean_1dt - 1.645*post_Bi_sd
            biot_1d = self.true_biot[:,round(x_cross[j]/3*(2000-1) + 1)]
            axs[plti[j],pltj[j]].plot(tt_1d,post_Bi_mean_1dt)
            axs[plti[j],pltj[j]].fill_between(tt_1d, funci1, funci0, facecolor='blue', alpha=0.3)
            axs[plti[j],pltj[j]].plot(tt_1d,biot_1d,label=r"True $Bi(x)$", color='tomato')
        
        '''
        
        '''
        fig, axs = plt.subplots(2,3)
        
        for j in range(6):
            post_Bi_samples_1dt = sum(coefs_post[:,i].reshape([-1,1])*np.reshape(chebx.P(x_cross[j],indices_x[i],tensor=False)*chebt.P(tt_1d,indices_t[i],tensor=False),[1,nt_1d]) for i in range(indices_t.shape[0]))
            post_Bi_samples_1dt = np.maximum(post_Bi_samples_1dt, np.zeros_like(post_Bi_samples_1dt))
            post_Bi_samples_1dt = np.minimum(post_Bi_samples_1dt, 200*np.ones_like(post_Bi_samples_1dt))
            post_Bi_median_1dt = np.median(post_Bi_samples_1dt,axis = 0)
            post_Bi_var = np.sqrt(np.var(post_Bi_samples_1dt,axis=0))
            funci1 = np.percentile(post_Bi_samples_1dt,97.5,axis=0)# post_Bi_mean_1dt + post_Bi_var
            funci0 = np.percentile(post_Bi_samples_1dt,0.025,axis=0)# post_Bi_mean_1dt - post_Bi_var
            biot_1d = biot[:,round(x_cross[j]/3*(nt_FD-1) + 1)]
            axs[plti[j],pltj[j]].plot(tt_1d,post_Bi_median_1dt)
            axs[plti[j],pltj[j]].fill_between(tt_1d, funci1, funci0, facecolor='blue', alpha=0.3)
            axs[plti[j],pltj[j]].plot(tt_1d,biot_1d,label=r"True $Bi(x)$", color='tomato')
        '''
        
        
        
        # plots solutions
        n_thinned = 100
        
        thin_dist = math.floor(coefs_exburnin.shape[0]/n_thinned)
        
        
        coefs_post = coefs_exburnin[0:burnin:thin_dist]
        sig_eps_post = sig_eps_exburnin[0:burnin:thin_dist]
        
        if self.csv_name==None:
            True_solution = self.true_sol
        
        mi = self.data.dat_scale*np.min([np.min(FD_solution),np.min(NN_solution)])
        ma = self.data.dat_scale*np.max([np.max(FD_solution),np.max(NN_solution)])
        
        ax3.get_yaxis().set_visible(False)
        ax4.get_yaxis().set_visible(False)
        ax5.get_yaxis().set_visible(False)


        NN_solution = griddata(np.hstack([xx_nn.reshape([-1,1]),tt_nn.reshape([-1,1])]), NN_solution.reshape([-1,1]), (xx, tt), method='linear').reshape([res_t,res_x])

        fig, axs = plt.subplots(3, 1)
        
        im = axs[0].scatter(self.t_dat, self.x_dat, s=50, c=self.data.dat_scale*self.z_dat, vmin=mi, vmax=ma,  cmap=cm.coolwarm)
        im = axs[1].pcolormesh(tt, xx, self.data.dat_scale*NN_solution, cmap=cm.coolwarm, vmin=mi, vmax=ma)
        fig.colorbar(im, ax=[axs[0],axs[1],axs[2]])
        im = axs[2].axhline(y=0., color='r', linestyle='-')
        im = axs[2].scatter(self.t_dat,resid,alpha=0.5,s=15)
#        im = axs[2].pcolormesh(tt, xx, self.data.dat_scale*FD_solution[:,:round(res_t*time_cutoff)].T, cmap=cm.coolwarm, vmin=mi, vmax=ma)
#        im = axs[2].pcolormesh(tt, xx, FD_solution[:,:round(res_t*time_cutoff)].T - NN_solution, cmap=cm.coolwarm, vmin=-0.005, vmax=0.005)
#        fig.colorbar(im, ax=[axs[2]])
        
        plt.show()
        
#        mi2 = np.min([np.min(FD_solution-True_solution),np.min(NN_solution.T-True_solution[:,:round(res_t*time_cutoff)])])
#        ma2 = np.max([np.max(FD_solution-True_solution),np.max(NN_solution.T-True_solution[:,:round(res_t*time_cutoff)])])
        
#        fig, axs = plt.subplots(2, 1)
        
#        axs[0].pcolormesh(tt_plt, xx_plt, FD_solution[:,:round(nt_FD*time_cutoff)].T - U[:,:round(nt_FD*time_cutoff)].T, cmap=cm.coolwarm, vmin=mi2, vmax=ma2)
#        im = axs[1].pcolormesh(tt_plt, xx_plt, NN_solution - U[:,:round(nt_FD*time_cutoff)].T, cmap=cm.coolwarm, vmin=mi2, vmax=ma2)
        
#        fig.colorbar(im, ax=axs.ravel().tolist())
#        plt.show()
        
        mi3 = np.min(FD_solution.T - NN_solution)
        ma3 = np.max(FD_solution.T - NN_solution)
        
        fig,ax = plt.subplots(1)
        im = plt.pcolormesh(tt, xx, FD_solution.T - NN_solution, cmap=cm.coolwarm, vmin=mi3, vmax=ma3)
        fig.colorbar(im)

        t_cross = [0,0.2,0.4,0.6,0.8,0.99]

        fig, axs = plt.subplots(2,3)

        mas = np.max(NN_solution)*self.data.dat_scale
        mis = np.min(NN_solution)*self.data.dat_scale
                
        for j in range(6):
            t_loc = self.t_dat[np.argmin(np.abs(t_cross[j]-self.t_dat))]
            tdat_1dx = self.t_dat[self.t_dat==t_loc]
            xdat_1dx = self.x_dat[self.t_dat==t_loc]
            zdat_1dx = self.z_dat[self.t_dat==t_loc]*self.data.dat_scale
            sol_1dx = self.sess.run(self.u_eval(tdat_1dx[0]*np.ones_like(xx_1d),xx_1d,coefs_est*(np.ones_like(xx_1d).reshape([-1,1]))))*self.data.dat_scale
            axs[plti[j],pltj[j]].plot(xx_1d,sol_1dx)
            axs[plti[j],pltj[j]].scatter(xdat_1dx,zdat_1dx,label=r"True $Bi(x)$", color='red',alpha = 0.4)
            axs[plti[j],pltj[j]].title.set_text('PDE solution at t =' + str(int(t_cross[j])))
            axs[plti[j],pltj[j]].set_ylim(mis-5,mas+5)
