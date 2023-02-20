# surrogate_da-hmc

This GitHub repository contains the code for the paper "Deep surrogate accelerated delayed-acceptance HMC for Bayesian inference of spatio-temporal heat fluxes in rotating disc systems."

## Description

The code in this repository generates data from the forward model using a random biot number and performs MCMC sampling with and without delayed-acceptance. Random walk Metropolis-Hastings (RWMH), Metropolis-adjusted Langevin (MALA), and Hamiltonian Monte Carlo (HMC) are included as samplers in each case.

The two subdirectories in the repository relate to the surrogate type that is used, between the general and adaptive surrogate: 
* In the code in the 'adaptive' directory, a deep learning surrogate is randomly initiated and adaptively trained from scratch using the generated data as described in the paper. Each of the 6 MCMC schemes is then carried out.
* In the code in the 'general' directory, a pre-trained general surrogate is loaded and used to carry out each of the MCMC schemes. The pre-trained general surrogate is provided as training one from scratch is time consuming, though this can be done by uncommenting the indicated section of the main script if desired.

In both cases the scripts will store the MCMC chains and tensorboard summaries in the working directory.

## Getting Started



### Dependencies
All computations were done using Python 3.7.9 with the following module versions. Other versions may work but have not been tested:

```
Package                       Version   
----------------------------- -------------------
arviz                         0.12.1
matplotlib                    3.3.3                                                                                     
numpy                         1.19.2                                                                                    
pandas                        1.1.4                                                                                     
scipy                         1.5.2                                                                                     
tensorboard                   1.15.0                                                                                    
tensorflow                    1.15.0                                                                                    
tensorflow-probability        0.8.0 
```
A more comprehensive list of modules can be found in ```requirements.txt```

### Installing and running

* Download these files to your machine and execute the main.py file in the directory of the desired experiment. This will train or load in the desired surrogate and carry out sampling.

## Directory Structure

This repo is organised as follows:

```
surrogate_da-hmc
├── adaptive
│   ├── main.py  # Main script to carry out adaptive surrogate experiments
│   ├── bayes_estimator.py # Utility code used within main script for training and sampling
│   ├── BC.py # Code to handle boundary conditions
│   ├── biot.py # Code to generate ground truth parameters
│   ├── cheb_poly.py # Code to manage Chebyshev polynomials 
│   ├── data.py # Code to produce/read data for the inverse problem
│   ├── dense_net.py # Code to implement a fully-connected feed-forward neural network
│   ├── FD_biot.py # Code for the finite difference solver used in delayed acceptance
│   ├── FD_flux.py # Alternative finite difference code used in the case when the flux is desired
│   ├── plot_estimates.py # Plotting routines
│   ├── prior_cov.csv # covariance matrix for the prior
│   ├── summaries # Folder where tensorboard summaries will be stored
├── general
│   ├── main.py  # Main script to carry out general surrogate experiments
│   ├── bayes_estimator.py # Utility code used within main script for training and sampling
│   ├── BC.py # Code to handle boundary conditions
│   ├── biot.py # Code to generate ground truth parameters
│   ├── cheb_poly.py # Code to manage Chebyshev polynomials 
│   ├── data.py # Code to produce/read data for the inverse problem
│   ├── dense_net.py # Code to implement a fully-connected feed-forward neural network
│   ├── FD_biot.py # Code for the finite difference solver used in delayed acceptance
│   ├── FD_flux.py # Alternative finite difference code used in the case when the flux is desired
│   ├── plot_estimates.py # Plotting routines
│   ├── network_parameters-Simulation.npz # parameters of the pre-trained general surrogate
│   ├── prior_cov.csv # covariance matrix for the prior
│   ├── summaries # Folder where tensorboard summaries will be stored
```



