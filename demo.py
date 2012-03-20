#!/usr/bin/env python
from __future__ import division
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 8

import hsmm
from basic_distributions.observations import gaussian
from basic_distributions.durations import poisson
from util.text import progprint_xrange

save_images = True
#hsmm.use_eigen()

#### Data generation
# Set parameters
N = 4
T = 500
obs_dim = 2
# Set duration parameters to be (10,20,30,40)
durparams = [10.*(idx+1) for idx in xrange(N)]
# Set observation hyperparameters (which control the random mean and covariance
# matrices for each state)
obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'lmbda_0':np.eye(obs_dim),
                'kappa_0':0.15,
                'nu_0':obs_dim+2}

# Construct the true observation and duration distributions
true_obs_distns = [gaussian(**obs_hypparams) for state in xrange(N)]
true_dur_distns = [poisson(lmbda=param) for param in durparams]

# Build the true HSMM model
truemodel = hsmm.hsmm(T,true_obs_distns,true_dur_distns)

# Sample data from the true model
data, labels = truemodel.generate()

# Plot the truth
truemodel.plot(data)
plt.gcf().suptitle('True HSMM')
if save_images:
    plt.savefig('truth.png')

#### Posterior inference
# Set the weak limit truncation level. This is essentially the maximum 
# number of states that can be learned
Nmax = 10

# Construct the observation and duration distribution objects, which set
# priors over parameters and then infer parameter values.
obs_distns = [gaussian(**obs_hypparams) for state in xrange(Nmax)]
dur_distns = [poisson() for state in xrange(Nmax)]

# Build the HSMM model that will represent the posterior 
posteriormodel = hsmm.hsmm(T,obs_distns,dur_distns)

# Resample the model 100 times, printing a dot at each iteration
# and plotting every so often
plot_every = 25
for idx in progprint_xrange(101):
    if (idx % plot_every) == 0:
        posteriormodel.plot(data)
        plt.gcf().suptitle('inferred HSMM after %d iterations (arbitrary colors)' % idx)
        if save_images:
            plt.savefig('posterior_sample_%d.png' % idx)

    posteriormodel.resample(data)

plt.show()
