#%% Train a general surrogate, simulate some data from the FD model, then run 10,000 warm-up and 10,000 burn in iterations with each
#   sampler on this data with the trained surrogate. Once complete, prints the sampling time, effective sample size, and cost for each method
 
import numpy as np
import tensorflow as tf
from bayes_estimator import bayes_estimator
import time as time



#%% Initialise the class 
'''
seed = 0

np.random.seed(seed)
tf.set_random_seed(seed)

kind = 'direct biot'
d = bayes_estimator(kind)
d.generate_data(c1=35000.,c2=1.,c3=1.,tlim=[0.0,3600.],xlim=[0.3,1.],rho_t=1200.,rho_x=0.3,std_biot=10,positive=True)
d.compile_map(prior_sd=100,layers = [68, 256, 256, 256, 256, 1])


#%% #Train a general surrogate

t2 = time.time()
d.train_surrogate(iterations = 100000, lr_train = 0.001,coeff_lim = [-80,80])
d.train_surrogate(iterations = 500000, lr_train = 0.0002,coeff_lim = [-80,80])
d.train_surrogate(iterations = 1500000, lr_train = 0.00005,coeff_lim = [-80,80])
d.train_surrogate(iterations = 100000, lr_train = 0.00001,coeff_lim = [-80,80])
t3 = time.time()

'''

#%% alternatively load pre trained general surrogate

seed = 0

np.random.seed(seed)
tf.set_random_seed(seed)

kind = 'direct biot'
d = bayes_estimator(kind)
d.generate_data(c1=35000.,c2=1.,c3=1.,tlim=[0.0,3600.],xlim=[0.3,1.],rho_t=900.,rho_x=0.3,std_biot=15)
d.compile_map(prior_sd=100,layers = [68, 256, 256, 256, 256, 1])
d.load_network('network_parameters-Simulation.npz')



#%% Find MAP approximation to use as initial MCMC point 
#(this only optimises the biot number, it does not train the network)
d.train(iterations = 500, lr_train = 0.0,  lr_map = 0.1,   radius = 1.5)
d.train(iterations = 500, lr_train = 0.0,  lr_map = 0.05,   radius = 1.5)
d.train(iterations = 500, lr_train = 0.0,  lr_map = 0.001,   radius = 1.5)
#d.plot_map_est(res_t=501,res_x=501)





#%% HMC sampling
d.compile_mcmc(fd_mesh=[401,401])


d.sample(iterations = 20000,step_no = 40,warmup=10000,prop_type='hmc',target_accept = 0.65)

d.save_sample('gen_HMC_sample'+str(seed)+'.npy')



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
d.sample(iterations = 20000,warmup=10000,prop_type='mala',target_accept = 0.574)

d.save_sample('gen_MALA_sample'+str(seed)+'.npy')


#d.plot_mcmc_est(res_t=2000,res_x=1000,burnin_prop = 0.5,n_thinned=2000)

print('mala sampling time    (secs):',d.sample_time)

import arviz as az
mala_ess = np.zeros([np.array(d.coefs_sample).shape[1]])

for i in range(mala_ess.shape[0]):
    mala_ess[i] = az.ess(np.array(d.coefs_sample)[10000:,i])

mala_time = d.sample_time
mala_cost = mala_time/np.mean(mala_ess)

print('mala sampling time (secs):',mala_time,'. ESS:', np.mean(mala_ess), '. Cost:',mala_cost)


#%% MCMC sampling
d.compile_mcmc(fd_mesh=[401,401])

#RWMH
d.sample(iterations = 20000,warmup=10000,prop_type='metropolis',target_accept = 0.5)

d.save_sample('gen_RWMH_sample'+str(seed)+'.npy')


#d.plot_mcmc_est(res_t=2000,res_x=1000,burnin_prop = 0.5,n_thinned=2000)

print('mcmc sampling time    (secs):',d.sample_time)


import arviz as az
met_ess = np.zeros([np.array(d.coefs_sample).shape[1]])

for i in range(met_ess.shape[0]):
    met_ess[i] = az.ess(np.array(d.coefs_sample)[10000:,i])
    
met_time = d.sample_time
met_cost = met_time/np.mean(met_ess)

print('mcmc sampling time (secs):',met_time,'. ESS:', np.mean(met_ess), '. Cost:',met_cost)



#%% DA-HMC sampling
d.compile_mcmc(fd_mesh=[401,401])

d.sample(iterations = 20000,step_no = 40,DA_thres=10000,warmup=10000,prop_type='hmc',target_accept = 0.65)

d.save_sample('gen_DA-HMC_sample'+str(seed)+'.npy')


#d.plot_mcmc_est(res_t=2000,res_x=1000,burnin_prop = 0.5,n_thinned=2000)

print('hmc sampling time    (secs):',d.sample_time)

import arviz as az
DA_hmc_ess = np.zeros([np.array(d.coefs_sample).shape[1]])

for i in range(hmc_ess.shape[0]):
    DA_hmc_ess[i] = az.ess(np.array(d.coefs_sample)[10000:,i])

DA_hmc_time = d.sample_time
DA_hmc_cost = DA_hmc_time/np.mean(DA_hmc_ess)


print('DA hmc sampling time (secs):',DA_hmc_time,'. ESS:', np.mean(DA_hmc_ess), '. Cost:',DA_hmc_cost)

#%% DA-MALA sampling
d.compile_mcmc(fd_mesh=[401,401])

#MALA
d.sample(iterations = 20000,DA_thres=10000,warmup=10000,prop_type='mala',target_accept = 0.574)


d.save_sample('gen_DA-MALA_sample'+str(seed)+'.npy')


#d.plot_mcmc_est(res_t=2000,res_x=1000,burnin_prop = 0.5,n_thinned=2000)

print('DA mala sampling time    (secs):',d.sample_time)

import arviz as az
DA_mala_ess = np.zeros([np.array(d.coefs_sample).shape[1]])

for i in range(DA_mala_ess.shape[0]):
    DA_mala_ess[i] = az.ess(np.array(d.coefs_sample)[10000:,i])

DA_mala_time = d.sample_time
DA_mala_cost = DA_mala_time/np.mean(DA_mala_ess)

print('DA mala sampling time (secs):',DA_mala_time,'. ESS:', np.mean(DA_mala_ess), '. Cost:',DA_mala_cost)


#%% DA-MCMC sampling
d.compile_mcmc(fd_mesh=[401,401])

#RWMH
d.sample(iterations = 20000,DA_thres=10000,warmup=10000,prop_type='metropolis',target_accept = 0.5)

d.save_sample('gen_DA-RWMH_sample'+str(seed)+'.npy')


#d.plot_mcmc_est(res_t=2000,res_x=1000,burnin_prop = 0.5,n_thinned=2000)

print('mcmc sampling time    (secs):',d.sample_time)


import arviz as az
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
    
    
