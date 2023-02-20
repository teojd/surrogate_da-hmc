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
from plot_estimates import plot_estimates
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from scipy.interpolate import griddata
import arviz as az
tfd = tfp.distributions

from BC import BC
from data import data
from biot import biot 
from dense_net import dense_net
from cheb_poly import cheb_poly
from FD_biot import FD_biot
from FD_flux import FD_flux


class bayes_estimator(plot_estimates):
    def __init__(self, kind = 'direct biot'):
        self.data = data()
        self.biot = self.data.biot
        self.bc   = BC()
        self.kind = kind
        if (kind not in ['direct flux','exp flux','direct biot','exp biot']):
            print('PDE kind not supported. Select one of: direct flux , exp flux, direct biot, exp biot')

    def read_data(self,csv_name=None,step=None,c1=362319.,c2=1.,c3=1.,smooth_lbc=46.,smooth_rbc=260.,smooth_ic=100.,x_multiplier=1.):
        self.positive = False
        self.dat = self.data.read_formatted(csv_name,step,x_multiplier)
        self.L, self.R, self.I = self.bc.getBC(self.dat,smooth_lbc=100000,smooth_rbc=100000,smooth_ic=100000)
        self.true_biot=None
        self.csv_name = csv_name
        self.t_dat = self.dat[:,0].reshape([-1,1])
        self.x_dat = self.dat[:,1].reshape([-1,1])
        self.z_dat = self.dat[:,2].reshape([-1,1])
        self.tlim = self.data.tlim
        self.xlim = self.data.xlim        
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

            
    def generate_data(self,L=None,R=None,I=None,c1=362319.,c2=1.,c3=1.,tlim=[0.,3600.],xlim=[0.3,1.],std_biot=20,rho_t=1200,rho_x=0.25,nu_t=1.5,nu_x=2.5,kernel_t='rbf',kernel_x='rbf',ndat_x=10,ndat_t=20,positive=False):
        self.positive = positive        
        if L==None:
            def L(t,return_domain=False):
                if return_domain==False:
                    return 0.3*np.ones_like(t)#0.3*t/t
                else:
                    return [tlim[0],tlim[1]]
                
        if R==None:
            def R(t,return_domain=False):
                if return_domain==False:
                    return np.ones_like(t)#t/t
                else:
                    return [tlim[0],tlim[1]]

        if I==None:
            def I(x,return_domain=False):
                if return_domain==False:
                    return x#2.5*x**2 - 2.25*x + 0.75 
                else:
                    return [xlim[0],xlim[1]]

        self.L = L
        self.R = R
        self.I = I
        if (self.kind == 'direct flux' or self.kind == 'exp flux'):
            generated = self.data.generate_data(ndat_t,ndat_x,tlim=tlim,xlim=xlim,std=std_biot,rho_t=rho_t,rho_x=rho_x,nu_t=nu_t,nu_x=nu_x,c1=c1,c2=c2,c3=c3,L=L,R=R,I=I,full=True,kind = 'flux')
        else:
            generated = self.data.generate_data(ndat_t,ndat_x,tlim=tlim,xlim=xlim,std=std_biot,rho_t=rho_t,rho_x=rho_x,nu_t=nu_t,nu_x=nu_x,c1=c1,c2=c2,c3=c3,L=L,R=R,I=I,full=True,kind = 'biot')

        self.std_biot = std_biot
        self.dat = generated[0]
        self.true_sol = generated[1]
        self.true_biot = generated[2]
        self.mesh =  generated[2]

        self.t_dat = self.dat[:,0].reshape([-1,1])
        self.x_dat = self.dat[:,1].reshape([-1,1])
        self.z_dat = self.dat[:,2].reshape([-1,1])
        self.tlim = self.data.tlim
        self.xlim = self.data.xlim
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.csv_name = None
            

                        
                      


    def update_bc(self,smooth_lbc=100000,smooth_rbc=100000,smooth_ic=100000):
        self.L, self.R, self.I = self.bc.getBC(self.dat,smooth_lbc=smooth_lbc,smooth_rbc=smooth_rbc,smooth_ic=smooth_ic)
        self.bc.plotBC(self.L, self.R, self.I,self.dat,bound_dat=True)

    def compile_map(self,layers = [68, 256, 256, 256, 256, 1],res_w=1.,prior_sd=40.,prior_mean=0.,prior_cov = 'prior_cov.csv'):
        self.net  = dense_net(layers)
        self.prior_var = np.float32((prior_sd**2))

        cheb_deg = 10
        self.n_terms= int((cheb_deg+1)*(cheb_deg+2)/2)
        t_degrees = np.linspace(0,cheb_deg,cheb_deg+1).astype(int)
        x_degrees = np.linspace(0,cheb_deg,cheb_deg+1).astype(int)
        t_degrees, x_degrees = np.meshgrid(t_degrees,x_degrees)
        t_degrees = t_degrees.reshape([-1])
        x_degrees = x_degrees.reshape([-1])
        self.indices_t = t_degrees[t_degrees+x_degrees<=cheb_deg] 
        self.indices_x = x_degrees[t_degrees+x_degrees<=cheb_deg] 
        self.indices = np.vstack([self.indices_t,self.indices_x]).T
        

        self.prior_covariance = np.array(pd.read_csv(prior_cov,header=None)).astype(np.float32)*self.prior_var
        self.prior_mean = np.zeros([self.n_terms]).astype(np.float32)

        scale = self.indices_t + self.indices_x+1
        scale = 2**(self.indices_t +self.indices_x)
        self.scale = np.maximum(scale/8,1.)
        
        self.nx_nn = 10
        self.nt_nn = 80

        self.size_a = 50 # no. of samples from each boundary (initial,left,right)
        self.size_x = self.nx_nn*self.nt_nn # no. of samples inside domain
              
        
        # parameter domain
        self.coefflim = [-80.,80]
        
        self.lr   = tf.placeholder(dtype = "float32", shape = [])
        self.loss_pl   = tf.placeholder(dtype = "float32", shape = [])
        self.map_pl = tf.placeholder(dtype = "float32", shape = [])
        self.x_tf = tf.placeholder(dtype = "float32", shape = [self.size_x,1])
        self.t_tf = tf.placeholder(dtype = "float32", shape = [self.size_x,1])
        self.noise_internal_tf   = tf.placeholder(dtype = "float32", shape = [self.size_x,66])
        self.noise_bound_tf   = tf.placeholder(dtype = "float32", shape = [self.size_a,66])
        
        
        self.pars_tf = tf.Variable(0.1*tf.ones([self.n_terms+1],dtype = "float32"))

        
#        self.xx_tf = tf.linspace(self.xlim[0],self.xlim[1],self.nx_nn)
#        self.tt_tf = tf.linspace(self.tlim[0],self.tlim[1],self.nt_nn)
        
#        self.x_tf, self.t_tf = tf.meshgrid(self.xx_tf,self.tt_tf)
#        self.x_tf = tf.cast(tf.reshape(self.x_tf,[-1,1]),tf.float32)
#        self.t_tf = tf.cast(tf.reshape(self.t_tf,[-1,1]),tf.float32)
        
        
        self.na = self.size_a
        self.xa_tf = np.reshape(self.xlim[0]*np.ones(self.na),[-1,1])
        self.ta_tf = np.reshape(np.linspace(self.tlim[0],self.tlim[1],self.na),[-1,1])
        
        self.nb = self.size_a
        self.xb_tf = np.reshape(self.xlim[1]*np.ones(self.nb),[-1,1])
        self.tb_tf = np.reshape(np.linspace(self.tlim[0],self.tlim[1],self.nb),[-1,1])
        
        self.n0 = self.size_a
        self.x0_tf = np.reshape(np.linspace(self.xlim[0],self.xlim[1],self.n0),[-1,1])
        self.t0_tf = np.reshape(self.tlim[0]*np.ones(self.n0),[-1,1])
        
        

        
        
        #evaluate differential operator terms on sample points
        self.pars_u = self.transform_pars(self.pars_tf[:-1])*tf.ones([self.nx_nn*self.nt_nn,1])+self.transform_pars(self.noise_internal_tf)
        
        self.u  = self.u_eval(self.t_tf,self.x_tf,self.pars_u)
        self.u_t = tf.gradients(self.u,self.t_tf)[0]
        self.u_x = tf.gradients(self.u,self.x_tf)[0]
        self.u_xx = tf.gradients(self.u_x,self.x_tf)[0]
        
        #evaluate network at boundary value sample points
        self.pars_a = self.transform_pars(self.pars_tf[:-1])*tf.ones([self.n0,1])+self.transform_pars(self.noise_bound_tf)
        
        self.u_0 = self.u_eval(self.t0_tf,self.x0_tf,self.pars_a)
        self.u_a = self.u_eval(self.ta_tf,self.xa_tf,self.pars_a)
        self.u_b = self.u_eval(self.tb_tf,self.xb_tf,self.pars_a)
        
        self.prior_covariance = np.array(pd.read_csv(prior_cov,header=None)).astype(np.float32)*self.prior_var
        self.prior_mean = np.zeros([self.n_terms]).astype(np.float32)

        #for simulated data use a mean of zero but for real data allow user specification
        if self.csv_name==None:
            self.prior_mean[0] = 0
        else:
            self.prior_mean[0] = prior_mean
            
        self.chebt = cheb_poly(self.tlim)
        self.chebx = cheb_poly(self.xlim)

        if self.csv_name==None:
            self.flux_scale = 1.
        else:
            self.flux_scale = self.data.dat_scale*0.016*6.6/2

        self.prior_density_tf = self.prior(self.pars_tf) 
        self.log_likelihood_tf = self.log_likelihood_nn(self.pars_tf,self.t_dat,self.x_dat,self.z_dat)

        
        
        self.Bi_tf = sum(tf.reshape(self.pars_u[:,i],[-1,1])*self.chebx.P(self.x_tf,self.indices_x[i])*self.chebt.P(self.t_tf,self.indices_t[i]) for i in range(self.indices_t.shape[0])) 
        
        
        if self.positive == True:
            self.Bi_tf = tf.math.maximum(self.Bi_tf,-0*tf.ones_like(self.Bi_tf))
        #self.Bi_tf = tf.math.minimum(self.Bi_tf,200*tf.ones_like(self.Bi_tf))

        #self.Bi_tf = tf.math.maximum(self.Bi_tf,-60*tf.ones_like(self.Bi_tf))


        # construct loss function
#        self.resid = ( (self.c2*self.u_xx + self.c3*self.u_x/self.x_tf - self.Bi_tf*self.u) - self.c1*self.u_t)**2
        if self.kind=='exp biot' or self.kind=='exp flux':
            self.bc_a = 1.0*(tf.exp(self.u_a)-self.L(self.ta_tf))**2
            self.bc_b = 1.0*(tf.exp(self.u_b)-self.R(self.tb_tf))**2
            self.ic   = 1.0*(tf.exp(self.u_0)-self.I(self.x0_tf))**2
        else:
            self.bc_a = 1.0*(self.u_a-self.L(self.ta_tf))**2
            self.bc_b = 1.0*(self.u_b-self.R(self.tb_tf))**2
            self.ic   = 1.0*(self.u_0-self.I(self.x0_tf))**2
            
        if self.kind=='exp biot':
            self.resid = res_w*(((self.c2*(self.u_xx+self.u_x**2) + self.c3*self.u_x/self.x_tf - self.Bi_tf) - self.c1*self.u_t)**2)/self.c1
        elif self.kind=='exp flux':
            self.resid = res_w*(((self.c2*(self.u_xx+self.u_x**2) + self.c3*self.u_x/self.x_tf - self.Bi_tf/tf.exp(self.u)) - self.c1*self.u_t)**2)/self.c1
        elif self.kind=='direct biot':
            self.resid = res_w*(((self.c2*self.u_xx + self.c3*self.u_x/self.x_tf - self.Bi_tf*self.u) - self.c1*self.u_t)**2)/self.c1 #biot
        elif self.kind=='direct flux':
            self.resid = res_w*(((self.c2*self.u_xx + self.c3*self.u_x/self.x_tf - self.Bi_tf) - self.c1*self.u_t)**2)/self.c1 #biot
        
        self.loss_PDE = tf.reduce_mean(self.resid) + tf.reduce_mean(self.bc_a) + tf.reduce_mean(self.bc_b) + tf.reduce_mean(self.ic) 
        
        self.loss_map = - (self.prior_density_tf + self.log_likelihood_tf)
        
        # prints total value of loss function to screen, as well as individual loss terms
        
        self.printout  = [tf.reduce_mean(self.resid) + tf.reduce_mean(self.bc_a) + tf.reduce_mean(self.bc_b) + tf.reduce_mean(self.ic),tf.reduce_mean(self.resid), tf.reduce_mean(self.bc_a), tf.reduce_mean(self.bc_b), tf.reduce_mean(self.ic),(self.prior_density_tf + self.log_likelihood_tf)]
        
        # get gradients of loss and clip by global norm
        optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_PDE,var_list=self.net.parameters))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))
        
        #optimizer_map = tf.contrib.opt.ScipyOptimizerInterface(loss_map, method='newton') 
        optimizer_map = tf.train.AdamOptimizer(learning_rate = self.lr)
        gradients_map, variables_map = zip(*optimizer.compute_gradients(self.loss_map,var_list=self.pars_tf))
        gradients_map, _ = tf.clip_by_global_norm(gradients_map, 1.)
        self.train_op_map = optimizer_map.apply_gradients(zip(gradients_map, variables_map))


        self.optimizer_newton = tf.contrib.opt.ScipyOptimizerInterface(self.loss_map,  method='BFGS', var_list = [self.pars_tf],options = {'maxiter': 1})

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        self.writer = tf.summary.FileWriter('./summaries', self.sess.graph)
        self.tf_loss_summary = tf.summary.scalar('loss', self.loss_pl)
        self.tf_map_summary = tf.summary.scalar('map', self.map_pl)
        self.it = -10
        

            

    def load_network(self,network_filename):
        parameters = np.load(network_filename)        
        for i in range(len(parameters.files)):
            self.net.parameters[i].load(parameters[parameters.files[i]],self.sess)

    def u_eval(self,t,x,coefs):
        t = tf.cast(tf.reshape(t,[-1,1]),tf.float32)
        x = tf.cast(tf.reshape(x,[-1,1]),tf.float32)
        coefs = tf.cast(tf.reshape(coefs,[-1,self.n_terms]),tf.float32)
    
        coefs_trans = (2*coefs-(self.coefflim[1]+self.coefflim[0]))/(self.coefflim[1]-self.coefflim[0])
        t_trans = (2*t-(self.tlim[1]+self.tlim[0]))/(self.tlim[1]-self.tlim[0])
        x_trans = (2*x-(self.xlim[1]+self.xlim[0]))/(self.xlim[1]-self.xlim[0])
    
        X = tf.concat([t_trans,x_trans,coefs_trans],axis=1)
        return self.net.evaluate(X)#*(x-self.xlim[1]) + self.R(t)

    def transform_sig(self,x):
            return x/10000
        
    def transform_pars(self,pars):
            return pars/self.scale
        
    def train(self,iterations,lr_train,lr_map,radius=1.,optimiser='standard',thres = np.inf): # main training loop
        for i in range(iterations):
            noise_internal_np = np.random.normal(size = [self.size_x,66])*radius
            noise_bound_np = np.random.normal(size = [self.size_a,66])*radius
            x_sample = np.random.uniform(self.xlim[0],self.xlim[1],size = [self.size_x,1])
            t_sample = np.random.uniform(self.tlim[0],self.tlim[1],size = [self.size_x,1])
            tf_dict = {self.lr:lr_train, self.noise_internal_tf:noise_internal_np , self.noise_bound_tf:noise_bound_np, \
                       self.x_tf:x_sample, self.t_tf:t_sample}
            self.sess.run(self.train_op,tf_dict)
            while self.sess.run(self.printout,tf_dict)[0]>thres:
                self.sess.run(self.train_op,tf_dict)
            if optimiser == 'newton':
                self.optimizer_newton.minimize(self.sess,feed_dict = tf_dict)  
            else:
                tf_dict = {self.lr:lr_map, self.noise_internal_tf:noise_internal_np , self.noise_bound_tf:noise_bound_np, \
                       self.x_tf:x_sample, self.t_tf:t_sample}
                self.sess.run(self.train_op_map,tf_dict)
            if i % 10 ==0:
                loss_eval = self.sess.run(self.printout,tf_dict)
                tf_dict = {self.lr:lr_map, self.loss_pl: loss_eval[0] , self.map_pl:loss_eval[-1], self.noise_internal_tf:noise_internal_np , self.noise_bound_tf:noise_bound_np, \
                       self.x_tf:x_sample, self.t_tf:t_sample}
                summary = self.sess.run(self.tf_loss_summary,tf_dict)
                summary2 = self.sess.run(self.tf_map_summary,tf_dict)
                self.it = self.it+10
                self.writer.add_summary(summary, self.it)            
                self.writer.add_summary(summary2, self.it)            
                print('iteration: ',i, '.  lr train: ', lr_train, 'lr map: ', lr_map, '. radius:', radius, '.  loss: ', "{:.7f}".format(loss_eval[0]), '. map:', "{:.3f}".format(loss_eval[-1]))


    def train_on_laplace(self,iterations,lr_train): # main training loop
        map_hess_untransformed = self.sess.run(tf.hessians(-self.loss_map, self.pars_tf)[0])
        self.inv_neg_hess_untransformed = np.linalg.inv(-map_hess_untransformed)
        chol_fac = np.linalg.cholesky(self.inv_neg_hess_untransformed)
        for i in range(iterations):
            map_cov_samples_untransformed = np.matmul(chol_fac,np.random.normal(size = [67,self.size_x + self.size_a])).T
            noise_internal_np = map_cov_samples_untransformed[:self.size_x,:66]
            noise_bound_np = map_cov_samples_untransformed[self.size_x:,:66]
            x_sample = np.random.uniform(self.xlim[0],self.xlim[1],size = [self.size_x,1])
            t_sample = np.random.uniform(self.tlim[0],self.tlim[1],size = [self.size_x,1])
            tf_dict = {self.lr:lr_train, self.noise_internal_tf:noise_internal_np , self.noise_bound_tf:noise_bound_np, \
                       self.x_tf:x_sample, self.t_tf:t_sample}
            self.sess.run(self.train_op,tf_dict)
            if i % 10 ==0:
                loss_eval = self.sess.run(self.printout,tf_dict)
                tf_dict = {self.loss_pl: loss_eval[0] , self.noise_internal_tf:noise_internal_np , self.noise_bound_tf:noise_bound_np, \
                       self.x_tf:x_sample, self.t_tf:t_sample}
                summary = self.sess.run(self.tf_loss_summary,tf_dict)
                self.it = self.it+10
                self.writer.add_summary(summary, self.it)            
                print('iteration: ',i, '.  lr train: ', lr_train, '.  loss: ', "{:.7f}".format(loss_eval[0]))
        


    def train_surrogate(self,iterations,lr_train,coeff_lim = [-80,80],save_filename = 'network_parameters'): # main training loop
        untransformed_map_pars = self.sess.run(self.pars_tf)[:-1]
        for i in range(iterations):
            untransformed_coefficient_sample = np.random.uniform(coeff_lim[0],coeff_lim[1],size=[self.size_x + self.size_a,66])
            untransformed_pars_internal_np = untransformed_coefficient_sample[:self.size_x,:] - untransformed_map_pars
            untransformed_pars_bound_np = untransformed_coefficient_sample[self.size_x:,:] - untransformed_map_pars
            x_sample = np.random.uniform(self.xlim[0],self.xlim[1],size = [self.size_x,1])
            t_sample = np.random.uniform(self.tlim[0],self.tlim[1],size = [self.size_x,1])
            tf_dict = {self.lr:lr_train, self.noise_internal_tf:untransformed_pars_internal_np , self.noise_bound_tf:untransformed_pars_bound_np, \
                       self.x_tf:x_sample, self.t_tf:t_sample}
            self.sess.run(self.train_op,tf_dict)
            if i % 10 ==0:
                loss_eval = self.sess.run(self.printout,tf_dict)
                tf_dict = {self.loss_pl: loss_eval[0] , self.noise_internal_tf:untransformed_pars_internal_np , self.noise_bound_tf:untransformed_pars_bound_np, \
                       self.x_tf:x_sample, self.t_tf:t_sample}
                summary = self.sess.run(self.tf_loss_summary,tf_dict)
                self.it = self.it+10
                self.writer.add_summary(summary, self.it)            
                print('iteration: ',i, '.  lr train: ', lr_train, '.  loss: ', "{:.7f}".format(loss_eval[0]))
            if i % 1000 ==0:
                parameters = self.sess.run(self.net.parameters)
                if np.isnan(loss_eval[0]):
                    break
                np.savez(save_filename,parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6],\
                                  parameters[7], parameters[8], parameters[9])


    def train_on_sample(self,lr_train,sample): # main training loop
        untransformed_map_pars = self.sess.run(self.pars_tf)[:-1]
        untransformed_coefficient_sample = sample[:,:-1]
        std = 0/self.scale
        inds = np.random.choice(untransformed_coefficient_sample.shape[0],size = self.size_x + self.size_a) #np.round((1-1/(1+np.random.choice(post.shape[0],size = [size_a-cut])**0.2))*post.shape[0]).astype(np.int)
        untransformed_coefficient_sample = untransformed_coefficient_sample[inds] + np.random.normal(loc=0,scale=std,size=[self.size_x + self.size_a,66]) 
        untransformed_pars_internal_np = untransformed_coefficient_sample[:self.size_x,:] - untransformed_map_pars
        untransformed_pars_bound_np = untransformed_coefficient_sample[self.size_x:self.size_x+self.size_a,:] - untransformed_map_pars
        x_sample = np.random.uniform(self.xlim[0],self.xlim[1],size = [self.size_x,1])
        t_sample = np.random.uniform(self.tlim[0],self.tlim[1],size = [self.size_x,1])
        tf_dict = {self.lr:lr_train, self.noise_internal_tf:untransformed_pars_internal_np , self.noise_bound_tf:untransformed_pars_bound_np, \
                       self.x_tf:x_sample, self.t_tf:t_sample}
        self.sess.run(self.train_op,tf_dict)
        loss_eval = self.sess.run(self.printout,tf_dict)
        tf_dict = {self.loss_pl: loss_eval[0] , self.noise_internal_tf:untransformed_pars_internal_np , self.noise_bound_tf:untransformed_pars_bound_np, \
                       self.x_tf:x_sample, self.t_tf:t_sample}
        return loss_eval[0]


        
    def PDE_prior(self,coefs): #function to evaluate prior pdf of given parameters
        pri = tfd.MultivariateNormalFullCovariance(loc=self.prior_mean, covariance_matrix=self.prior_covariance).log_prob(coefs)
        return pri
       
        
        
    def sigeps_prior(self,sig_eps):
        return tfd.Gamma(concentration=5.5,rate=1000.).log_prob(sig_eps)
        #return tfd.Gamma(concentration=1000,rate=1500000.).log_prob(sig_eps)
    
    def prior(self,pars):
        coefs = self.transform_pars(pars[:-1])
        sig_eps = self.transform_sig(pars[-1])
        return self.PDE_prior(coefs) + self.sigeps_prior(sig_eps)
    
    def log_likelihood_nn(self,pars,t_dat,x_dat,z_dat):
        n_dat = t_dat.shape[0]
        coefs  = self.transform_pars(pars[:-1])*tf.ones([n_dat,1])
        sig_eps = self.transform_sig(pars[-1])
        if self.kind=='exp biot' or self.kind=='exp flux':
            PDE_pred = tf.exp(self.u_eval(t_dat,x_dat,coefs))
        else:
            PDE_pred = self.u_eval(t_dat,x_dat,coefs)
        dat_term = 0.5*tf.reduce_sum((tf.cast(tf.constant(z_dat),tf.float32)-PDE_pred)**2)/sig_eps
        log_lik = -0.5*n_dat*(tf.log(2* np.pi)+tf.log(sig_eps)) - dat_term 
        return log_lik
    
    def log_likelihood_FD(self,pars,t_dat,x_dat,z_dat,FD):
        n_dat = t_dat.shape[0]
        coefs  = self.transform_pars(pars[:-1])
        sig_eps = self.transform_sig(pars[-1])
        sol = FD.solve(coefs).T
        PDE_pred = sp.interpolate.RectBivariateSpline(self.FD.tmesh,self.FD.xmesh, sol).ev(t_dat.reshape([-1,1]),x_dat.reshape([-1,1]))
        log_lik = -0.5*np.sum((z_dat-PDE_pred)**2)/sig_eps - 0.5*n_dat*(np.log(2* np.pi)+np.log(sig_eps))
        return log_lik
   
        
        
    def grad_target_dens(self,pars,t_dat,x_dat,z_dat):
        prior_density = self.prior(pars) 
        log_likelihood = self.log_likelihood_nn(pars,t_dat,x_dat,z_dat)
        return tf.gradients(prior_density + log_likelihood,pars)[0]
    

    def make_mass(self,covariances):
        inv_mass_mat = covariances
        mass_matrix = tf.linalg.inv(covariances)
        log_det_mass_mat = tf.constant(0.)  #tf.linalg.logdet(inv_mass_mat) # tf.constant(0.)  
        return mass_matrix,inv_mass_mat,log_det_mass_mat         

    def log_momentum_pdf(self,momentum,log_det_mass_mat,mass_matrix,inv_mass_mat):
        return -0.5*np.sum(momentum.reshape([-1,1])*np.matmul(inv_mass_mat,momentum.reshape([-1,1]))) -0.5*np.log(2* np.pi)*mass_matrix.shape[0] - 0.5*log_det_mass_mat  


    def compile_hamiltonian_proposal(self,step_no):
    
    
        def hamil_steps(step_no,step_size,pars,momentum,inv_mass_mat):
            dUdq = -self.grad_target_dens(pars,self.t_dat,self.x_dat,self.z_dat)
            momentum = momentum - 0.5*step_size*dUdq  # half step
            for _ in range(step_no - 1):
                pars = pars + step_size*tf.reshape(tf.matmul(inv_mass_mat,tf.reshape(momentum,[-1,1])),[-1])  # whole step
                dUdq = -self.grad_target_dens(pars,self.t_dat,self.x_dat,self.z_dat)
                momentum = momentum - step_size*dUdq  # whole step
            pars = pars + step_size * momentum
            dUdq = -self.grad_target_dens(pars,self.t_dat,self.x_dat,self.z_dat)
            momentum = momentum - 0.5*step_size*dUdq # half step
            return pars, -momentum
        
        
        pars_prop, momentum_prop = hamil_steps(step_no,self.step_size_pl,self.pars_mcmc,self.momentum_mcmc,self.inv_mass_mat_tf)
        
        def hamiltonian_propose_point(pars0,step_no,step_size,vars_np): #could be compiled entirely? probably not worth the time though
            mass_matrix,inv_mass_mat,log_det_mass_mat = self.sess.run([self.mass_matrix_tf,self.inv_mass_mat_tf,self.log_det_mass_mat_tf], {self.pars_mcmc:pars0,self.variances:vars_np})
            
            # decompose the mass matrix in advance to speed up this step
            momentum0 = np.random.multivariate_normal(np.zeros_like(mass_matrix[0]),mass_matrix).astype(np.float32) #np.ones([67]) #
            
            #test time difference with pars1 = pars0 # did this, 8 secs per 1000 its, not worth further optimisation
            pars1,momentum1 = self.sess.run([pars_prop,momentum_prop],{self.step_no_pl:step_no,self.step_size_pl:step_size,self.pars_mcmc:pars0,self.momentum_mcmc:momentum0,self.variances:vars_np})
            
            log_lik0,prior_density0 = self.sess.run([self.log_likelihood_mcmc,self.prior_density_mcmc],{self.pars_mcmc:pars0})
            log_lik1,prior_density1 = self.sess.run([self.log_likelihood_mcmc,self.prior_density_mcmc],{self.pars_mcmc:pars1})
            
            factor0 = log_lik0 + prior_density0 + self.log_momentum_pdf(momentum0,log_det_mass_mat,mass_matrix,inv_mass_mat)
            factor1 = log_lik1 + prior_density1 + self.log_momentum_pdf(momentum1,log_det_mass_mat,mass_matrix,inv_mass_mat)
     #       print(log_lik0 , prior_density0 , self.log_momentum_pdf(momentum0,log_det_mass_mat,mass_matrix,inv_mass_mat))
      #      print(log_lik1 , prior_density1 , self.log_momentum_pdf(momentum1,log_det_mass_mat,mass_matrix,inv_mass_mat))
            return pars1, factor0, factor1, log_lik0, log_lik1
        
        return hamiltonian_propose_point
    

    def metropolis_propose_point(self,pars0,vars_np,step_size):
        pars1 = pars0 + step_size*np.random.multivariate_normal(np.zeros_like(pars0),vars_np)
        log_lik0,prior_density0 = self.sess.run([self.log_likelihood_mcmc,self.prior_density_mcmc],{self.pars_mcmc:pars0})
        log_lik1,prior_density1 = self.sess.run([self.log_likelihood_mcmc,self.prior_density_mcmc],{self.pars_mcmc:pars1})
        factor0 = log_lik0 + prior_density0
        factor1 = log_lik1 + prior_density1
        return pars1, factor0, factor1, log_lik0, log_lik1


    def mala_propose_point(self,pars0,vars_np,step_size):
        grad_targ0 = self.sess.run(self.mala_grad,{self.pars_mcmc:pars0})
        pars1 = np.random.multivariate_normal(pars0 + step_size*np.matmul(grad_targ0,vars_np),2*step_size*vars_np)
        grad_targ1 = self.sess.run(self.mala_grad,{self.pars_mcmc:pars1})
        log_lik0,prior_density0 = self.sess.run([self.log_likelihood_mcmc,self.prior_density_mcmc],{self.pars_mcmc:pars0})
        log_lik1,prior_density1 = self.sess.run([self.log_likelihood_mcmc,self.prior_density_mcmc],{self.pars_mcmc:pars1})
        factor0 = log_lik0 + prior_density0 + sp.stats.multivariate_normal.logpdf(pars1,pars0 + step_size*np.matmul(grad_targ0,vars_np),2*step_size*vars_np)
        factor1 = log_lik1 + prior_density1 + sp.stats.multivariate_normal.logpdf(pars0,pars1 + step_size*np.matmul(grad_targ1,vars_np),2*step_size*vars_np)
        return pars1, factor0, factor1, log_lik0, log_lik1

    
    def accept(self,factor0,factor1):
        #if factor1==factor0:
        #    return False
        #else:
        accept_p = np.random.uniform(0,1)
        return ( accept_p < np.exp(factor1-factor0) )

 

    def sample(self,iterations = 100000,step_no = 30,cov_adapt_thres=1000,warmup=np.inf,refine_thres=np.inf,DA_thres=np.inf,loss_thres=np.inf,learning_rate = 0.0003,prop_type='hmc',target_accept = 0.65):
        train_prop = 0.5
        self.log_lik0_fd = None
        
        
        if  step_no != self.step_no0:
            self.hamiltonian_propose_point = self.compile_hamiltonian_proposal(step_no) #compile a function to generate hamiltonian proposals for specified step no
            self.step_no0 = step_no
            
        
        loss_printout = np.inf
        self.pars0 = np.float32(np.hstack([self.coefs,self.sig_eps]))
        
        
        for i in range(len(self.sig_eps_sample),iterations):
            if i==warmup:
                self.step_size = np.mean(np.array(self.step_size_vec[300:]))
                if DA_thres==warmup and prop_type=='hmc':
                    self.step_size = self.step_size*0.7
                t0=time.time()
            if i==0: #initially use the rescaled prior as the mass matrix
                cov = (self.prior_covariance*self.scale).T*self.scale*(self.prior_var)
                eps_var =  1000000.0
                cov = np.hstack([cov,np.zeros([66,1])])
                eps_row = np.hstack([np.zeros([1,66]),[[eps_var]]])
                self.vars_np = np.vstack([cov,eps_row])
                self.step_size_vec = [self.step_size]

            elif i % 100 == 0 and i>=cov_adapt_thres and i<warmup: #Update the mass matrix every 100th iteration until hyperpars are fixed
                self.vars_np = 100000*np.cov(np.hstack([np.array(self.coefs_sample_untransformed[round(i*0.6):]),np.array(self.sig_eps_sample_untransformed[round(i*0.6):]).reshape([-1,1])]).T)
        #   vars_np represents an estimate of the posterior covariance of the parameter. Used in mass matrix since optimal mass is the inverse of the posterior covariance
        
            self.pars0 = np.float32(np.hstack([self.coefs,self.sig_eps]))
            if prop_type=='hmc':
                self.pars1, factor0, factor1, log_lik0_nn, log_lik1_nn =  self.hamiltonian_propose_point(self.pars0,step_no,self.step_size,self.vars_np) #sample proposal points
            elif prop_type=='metropolis':
                self.pars1, factor0, factor1, log_lik0_nn, log_lik1_nn =  self.metropolis_propose_point(self.pars0,self.vars_np,self.step_size) #sample proposal points
            elif prop_type=='mala':
                self.pars1, factor0, factor1, log_lik0_nn, log_lik1_nn =  self.mala_propose_point(self.pars0,self.vars_np,self.step_size) #sample proposal points
            coefs1  = self.pars1[:66]
            sig_eps1 = self.pars1[-1]

            if i % 10 == 9: #Show printout every 10th iteration
                self.acceptance_rate = np.sum(self.accepted[-100:])/len(self.sig_eps_sample[-100:])
                print(i+1 , 'loss=',"{:.2e}".format(loss_printout), 'last 100 accept=', "{:.2f}".format(self.acceptance_rate), 'coef0=', "{:.3f}".format(np.mean(np.array(self.coefs_sample)[:,0])),"{:.3f}".format(np.array(self.coefs_sample)[i-1,0]),'sig_eps=', "{:.5f}".format(np.mean(self.sig_eps_sample)),"{:.6f}".format(self.sig_eps_sample[i-1]),'step size = ',"{:.6f}".format(self.step_size))

            if i % 10 == 9 and i<warmup: #Update the step size every 10th iteration until hyperpars are fixed
                rate_check = np.sum(self.accepted[-40:])/len(self.sig_eps_sample[-40:])
                self.step_size = self.step_size*(1+0.5*(rate_check-target_accept)**2)**np.sign(rate_check-target_accept)
                self.step_size_vec.append(self.step_size)
                  
            if (self.accept(factor0,factor1)): #choose whether to accept metropolis proposal for hyperparameters, store sample and update chain
                if i<DA_thres: #Delayed acceptance threshold
                    self.coefs     = coefs1
                    self.sig_eps   = sig_eps1
                    self.accepted.append(1.)
                else:
                    if self.log_lik0_fd==None:
                        self.log_lik0_fd = self.log_likelihood_FD(self.pars0,self.t_dat,self.x_dat,self.z_dat,self.FD)
                    self.log_lik1_fd = self.log_likelihood_FD(self.pars1,self.t_dat,self.x_dat,self.z_dat,self.FD)
                    accept_p = np.exp(self.log_lik1_fd + log_lik0_nn - self.log_lik0_fd - log_lik1_nn) 
#                    print([log_lik0_nn,self.log_lik0_fd,log_lik1_nn,self.log_lik1_fd,accept_p])
                    if (np.random.uniform(0,1) < accept_p):
                        self.coefs     = coefs1
                        self.sig_eps = sig_eps1            
                        self.accepted.append(1.)
                        self.log_lik0_fd = self.log_lik1_fd
                    else:
                        self.accepted.append(0.)
            else:
                self.accepted.append(0.)

            if i>refine_thres and i<warmup: #Refine network every other iteration
                    self.sample_choice = np.hstack([np.array(self.coefs_sample_untransformed[round(i*(1-train_prop)):]),np.zeros([i-round(i*(1-train_prop)),1])])
                    loss_printout = self.train_on_sample(learning_rate,self.sample_choice)

            if i % round(warmup/2) == 1 and i>refine_thres and i<warmup: #Refine network every other iteration
                it = 0  
                self.t0 = time.time()
                while loss_printout>loss_thres:    
                    self.sample_choice = np.hstack([np.array(self.coefs_sample_untransformed[round(i*(1-train_prop)):]),np.zeros([i-round(i*(1-train_prop)),1])])
                    if it%5000 < 4500:
                        loss_printout = self.train_on_sample(learning_rate,self.sample_choice)
                    else:
                        loss_printout = self.train_on_sample(0.000001,self.sample_choice)
                    it = it+1
                    if it % 10 == 0:
                        print('it = ', it,', loss = ',loss_printout)
                self.t1 = time.time()
                    
            if i==warmup and i>refine_thres:
                for j in range(1000):    
                    self.sample_choice = np.hstack([np.array(self.coefs_sample_untransformed[round(i*(1-train_prop)):]),np.zeros([i-round(i*(1-train_prop)),1])])
                    loss_printout = self.train_on_sample(0.00001,self.sample_choice)
                print('Final loss = ',loss_printout)


 #           if i==DA_thres*1.5:
 #               for j in range(50000):
 #                   self.sample_choice = np.hstack([np.array(self.coefs_sample_untransformed[round(i*(1-train_prop)):]),np.zeros([i-round(i*(1-train_prop)),1])])
 #                   loss_printout = self.train_on_sample(min(0.00005,loss_printout*8),self.sample_choice)
 #                   if j % 10 == 9:
 #                       print(i+1 , 'loss=',"{:.2e}".format(loss_printout), 'last 100 accept=', "{:.2f}".format(self.acceptance_rate), 'coef0=', "{:.3f}".format(np.mean(np.array(self.coefs_sample)[:,0])),"{:.3f}".format(np.array(self.coefs_sample)[i-1,0]),'sig_eps=', "{:.5f}".format(np.mean(self.sig_eps_sample)),"{:.6f}".format(self.sig_eps_sample[i-1]),'step size = ',"{:.6f}".format(self.step_size))

                    
            #add new sample to lists
            self.coefs_sample.append(self.transform_pars(self.coefs))
            self.sig_eps_sample.append(self.transform_sig(self.sig_eps))

            self.coefs_sample_untransformed.append(self.coefs)
            self.sig_eps_sample_untransformed.append(self.sig_eps)
            
        
        t1 = time.time()
        
        self.sample_time = t1-t0

       
    def compile_mcmc(self,fd_mesh=[61,41]):        


        self.pars_mcmc = tf.placeholder(dtype = "float32", shape = [67]) 
        self.momentum_mcmc = tf.placeholder(dtype = "float32", shape = [67])
        self.variances = tf.placeholder(dtype = "float32", shape = [67,67]) 
        self.step_size_pl = tf.placeholder(dtype = "float32", shape = [])
        self.step_no_pl = tf.placeholder(dtype = "float32", shape = [])        

        self.prior_density_mcmc = self.prior(self.pars_mcmc) 
        self.log_likelihood_mcmc = self.log_likelihood_nn(self.pars_mcmc,self.t_dat,self.x_dat,self.z_dat)
     
        self.mala_grad = self.grad_target_dens(self.pars_mcmc,self.t_dat,self.x_dat,self.z_dat)

           
        self.map_pars_untransformed_mcmc = self.sess.run(self.pars_tf)
        
        self.map_hess_untransformed = self.sess.run(tf.hessians(-self.loss_map, self.pars_tf)[0])
        self.map_cov = np.linalg.inv(-self.map_hess_untransformed)
            
        
#        self.mass_matrix_tf,self.inv_mass_mat_tf,self.log_det_mass_mat_tf = self.make_mass(tf.constant(self.map_cov))     
        self.mass_matrix_tf,self.inv_mass_mat_tf,self.log_det_mass_mat_tf = self.make_mass(self.variances)
        
        self.coefs = self.map_pars_untransformed_mcmc[:-1]
        self.sig_eps = self.map_pars_untransformed_mcmc[-1] 

#        self.coefs = np.ones([66])*0.
#        self.sig_eps = 50.5 
        
        
        #lists to store posterior samples

        self.coefs_sample_untransformed = []   
        self.sig_eps_sample_untransformed = [] 
        self.accepted = []

        self.coefs_sample = []   
        self.sig_eps_sample = [] 
        
        self.n_accepted = 0
        
        #set iterations here
        self.step_size=0.00001

        self.step_no0 = 0
        self.loss_printout=0.
        
        self.nt_FD = fd_mesh[0]
        self.nx_FD = fd_mesh[1]

        if self.kind=='exp biot' or self.kind=='direct biot':
            self.FD = FD_biot(self.xlim[0],self.xlim[1],self.tlim[1],self.L,self.R,self.I,self.nx_FD-1,self.nt_FD-1,0.5,self.indices_t,self.indices_x,c1=self.c1,T0=self.tlim[0])
        else:
            self.FD = FD_flux(self.xlim[0],self.xlim[1],self.tlim[1],self.L,self.R,self.I,self.nx_FD-1,self.nt_FD-1,0.5,self.indices_t,self.indices_x,c1=self.c1,T0=self.tlim[0])


#        x_pts = np.linspace(self.xlim[0],self.xlim[1],self.nx_FD)
#        t_pts = np.linspace(self.tlim[0],self.tlim[1],self.nt_FD)

        x_pts = self.FD.xmesh
        t_pts = self.FD.tmesh


        self.xx, self.tt = np.meshgrid(x_pts,t_pts)


            
        
    def save_sample(self,filename):
        post_coefs = np.array(self.coefs_sample)[:50000]
        post_sig = np.array(self.sig_eps_sample)[:50000]
        np.save(filename, post_coefs)
        np.save('sig'+filename, post_sig)
        
    
    
          