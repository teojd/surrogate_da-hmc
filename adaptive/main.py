#%% simulated data example
import numpy as np
import tensorflow as tf
from bayes_estimator import bayes_estimator
import time as time



#%% Initialise the class 
seed = 0

np.random.seed(seed)
tf.set_random_seed(seed)

kind = 'direct biot'
d = bayes_estimator(kind)
d.generate_data(c1=35000.,c2=1.,c3=1.,tlim=[0.0,3600.],xlim=[0.3,1.],rho_t=900.,rho_x=0.3,std_biot=15)
d.compile_map(prior_sd=100,layers = [68, 256, 256, 256, 256, 1])


#%% Generate MAP estimate


t0=time.time()
d.train(iterations = 5000,  lr_train = 0.003,   lr_map = 0.1,    radius = 15.0,thres = 0.1)
d.train(iterations = 10000,  lr_train = 0.003,   lr_map = 0.1,    radius = 10.0,thres = 0.05)
d.train(iterations = 5000, lr_train = 0.001,  lr_map = 0.01,   radius = 5.0,thres = 0.01)
d.train(iterations = 5000, lr_train = 0.001,  lr_map = 0.003,   radius = 2.0)
d.train(iterations = 5000, lr_train = 0.0004,  lr_map = 0.003,   radius = 0.5)
d.train(iterations = 1000,  lr_train = 0.0001, lr_map = 0.001, radius = 0.1)
t1=time.time()





#d.plot_map_est(res_t=501,res_x=501)
#d.L1_error(res_t=1001,res_x=1001)

#print('time taken (secs):',t1-t0)
    

#%% Train surrogate model over the laplace estimate
t2=time.time()
#d.train_on_laplace(iterations = 10000,lr_train = 0.001)
d.train_on_laplace(iterations = 20000,lr_train = 0.0004)
d.train_on_laplace(iterations = 10000,lr_train = 0.00015)
#d.train_on_laplace(iterations = 5000,lr_train = 0.00005)
t3=time.time()


#this saves the surrogate as 'laplace_surrogate.npz'
d.train_surrogate(iterations = 2000, lr_train = 0.00000,coeff_lim = [-80,80],save_filename = 'laplace_surrogate.npz')


print('map estimation time (secs):',t1-t0)
print('Laplace train time (secs):',t3-t2)


#%% HMC sampling
d.compile_mcmc(fd_mesh=[401,401])

d.sample(iterations = 20000,refine_thres=1000,step_no = 40,cov_adapt_thres=1000,warmup=10000,loss_thres=0.0000015,learning_rate = 0.00015,prop_type='hmc',target_accept = 0.65)

d.save_sample('adapt_HMC_sample'+str(seed)+'.npy')


#d.plot_mcmc_est(res_t=2000,res_x=1000,burnin_prop = 0.5,n_thinned=2000)

print('hmc sampling time    (secs):',d.sample_time)

import arviz as az
hmc_ess = np.zeros([np.array(d.coefs_sample).shape[1]])

for i in range(hmc_ess.shape[0]):
    hmc_ess[i] = az.ess(np.array(d.coefs_sample)[10000:,i])

hmc_time = d.sample_time
hmc_cost = hmc_time/np.mean(hmc_ess)


print('hmc sampling time (secs):',hmc_time,'. ESS:', np.mean(hmc_ess), '. Cost:',hmc_cost)



#%% MALA sampling
d.compile_mcmc(fd_mesh=[401,401])

#MALA
d.sample(iterations = 20000,refine_thres=200000,cov_adapt_thres=1000,warmup=10000,loss_thres=0.0000015,learning_rate = 0.00015,prop_type='mala',target_accept = 0.574)

d.save_sample('adapt_MALA_sample'+str(seed)+'.npy')


#d.plot_mcmc_est(res_t=2000,res_x=1000,burnin_prop = 0.5,n_thinned=2000)

mala_ess = np.zeros([np.array(d.coefs_sample).shape[1]])

for i in range(mala_ess.shape[0]):
    mala_ess[i] = az.ess(np.array(d.coefs_sample)[10000:,i])

mala_time = d.sample_time
mala_cost = mala_time/np.mean(mala_ess)

print('mala sampling time (secs):',mala_time,'. ESS:', np.mean(mala_ess), '. Cost:',mala_cost)


#%% MCMC sampling
d.compile_mcmc(fd_mesh=[401,401])

#RWMH
d.sample(iterations = 20000,refine_thres=200000,cov_adapt_thres=5000,warmup=10000,loss_thres=0.0000015,learning_rate = 0.0002,prop_type='metropolis',target_accept = 0.5)

d.save_sample('adapt_RWMH_sample'+str(seed)+'.npy')


#d.plot_mcmc_est(res_t=2000,res_x=1000,burnin_prop = 0.5,n_thinned=2000)

met_ess = np.zeros([np.array(d.coefs_sample).shape[1]])

for i in range(met_ess.shape[0]):
    met_ess[i] = az.ess(np.array(d.coefs_sample)[10000:,i])
    
met_time = d.sample_time
met_cost = met_time/np.mean(met_ess)

print('mcmc sampling time (secs):',met_time,'. ESS:', np.mean(met_ess), '. Cost:',met_cost)



#%% DA-HMC sampling
d.compile_mcmc(fd_mesh=[401,401])

d.sample(iterations = 20000,step_no = 40,refine_thres=1000000,cov_adapt_thres=1000,DA_thres=10000,warmup=10000,loss_thres=0.0000015,learning_rate = 0.0002,prop_type='hmc',target_accept = 0.65)

d.save_sample('adapt_DA-HMC_sample'+str(seed)+'.npy')


#d.plot_mcmc_est(res_t=2000,res_x=400,burnin_prop = 0.5,n_thinned=2000)

DA_hmc_ess = np.zeros([np.array(d.coefs_sample).shape[1]])

for i in range(DA_hmc_ess.shape[0]):
    DA_hmc_ess[i] = az.ess(np.array(d.coefs_sample)[10000:,i])

DA_hmc_time = d.sample_time
DA_hmc_cost = DA_hmc_time/np.mean(DA_hmc_ess)


print('DA hmc sampling time (secs):',DA_hmc_time,'. ESS:', np.mean(DA_hmc_ess), '. Cost:',DA_hmc_cost)

print('hmc sampling time (secs):',hmc_time,'. ESS:', np.mean(hmc_ess), '. Cost:',hmc_cost)
print('mala sampling time (secs):',mala_time,'. ESS:', np.mean(mala_ess), '. Cost:',mala_cost)
print('mcmc sampling time (secs):',met_time,'. ESS:', np.mean(met_ess), '. Cost:',met_cost)


#%% DA-MALA sampling
d.compile_mcmc(fd_mesh=[401,401])

#MALA
d.sample(iterations = 20000,refine_thres=200000,cov_adapt_thres=1000,DA_thres=10000,warmup=10000,loss_thres=0.0000015,learning_rate = 0.00015,prop_type='mala',target_accept = 0.574)

d.save_sample('adapt_DA-MALA_sample'+str(seed)+'.npy')


#d.plot_mcmc_est(res_t=2000,res_x=1000,burnin_prop = 0.5,n_thinned=2000)

DA_mala_ess = np.zeros([np.array(d.coefs_sample).shape[1]])

for i in range(DA_mala_ess.shape[0]):
    DA_mala_ess[i] = az.ess(np.array(d.coefs_sample)[10000:,i])

DA_mala_time = d.sample_time
DA_mala_cost = DA_mala_time/np.mean(DA_mala_ess)

print('DA mala sampling time (secs):',DA_mala_time,'. ESS:', np.mean(DA_mala_ess), '. Cost:',DA_mala_cost)


print('DA hmc sampling time (secs):',DA_hmc_time,'. ESS:', np.mean(DA_hmc_ess), '. Cost:',DA_hmc_cost)
print('hmc sampling time (secs):',hmc_time,'. ESS:', np.mean(hmc_ess), '. Cost:',hmc_cost)
print('mala sampling time (secs):',mala_time,'. ESS:', np.mean(mala_ess), '. Cost:',mala_cost)
print('mcmc sampling time (secs):',met_time,'. ESS:', np.mean(met_ess), '. Cost:',met_cost)


#%% DA-MCMC sampling
d.compile_mcmc(fd_mesh=[401,401])

#RWMH
d.sample(iterations = 20000,refine_thres=200000,cov_adapt_thres=5000,DA_thres=10000,warmup=10000,loss_thres=0.0000015,learning_rate = 0.00015,prop_type='metropolis',target_accept = 0.5)

d.save_sample('adapt_DA-RWMH_sample'+str(seed)+'.npy')


#d.plot_mcmc_est(res_t=2000,res_x=1000,burnin_prop = 0.5,n_thinned=2000)

DA_met_ess = np.zeros([np.array(d.coefs_sample).shape[1]])

for i in range(DA_met_ess.shape[0]):
    DA_met_ess[i] = az.ess(np.array(d.coefs_sample)[10000:,i])
    
DA_met_time = d.sample_time
DA_met_cost = DA_met_time/np.mean(DA_met_ess)

#print('DA mcmc sampling time (secs):',DA_met_time,'. ESS:', np.mean(DA_met_ess), '. Cost:',DA_met_cost)

#%%
print('DA hmc sampling time (secs):',DA_hmc_time,'. ESS:', np.mean(DA_hmc_ess), '. Cost:',DA_hmc_cost)
print('DA mala sampling time (secs):',DA_mala_time,'. ESS:', np.mean(DA_mala_ess), '. Cost:',DA_mala_cost)
print('DA mcmc sampling time (secs):',DA_met_time,'. ESS:', np.mean(DA_met_ess), '. Cost:',DA_met_cost)

print('hmc sampling time (secs):',hmc_time,'. ESS:', np.mean(hmc_ess), '. Cost:',hmc_cost)
print('mala sampling time (secs):',mala_time,'. ESS:', np.mean(mala_ess), '. Cost:',mala_cost)
print('mcmc sampling time (secs):',met_time,'. ESS:', np.mean(met_ess), '. Cost:',met_cost)

esss = np.vstack([np.mean(DA_hmc_ess),np.mean(DA_mala_ess),np.mean(DA_met_ess),np.mean(hmc_ess),np.mean(mala_ess),np.mean(met_ess)])

np.savetxt("ess"+str(seed)+".csv", esss, delimiter=",")
    