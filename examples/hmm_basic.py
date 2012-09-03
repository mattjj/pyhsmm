#!/usr/bin/env python
from __future__ import division
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 8

import pyhsmm
from pyhsmm.util.text import progprint_xrange

# pyhsmm.use_eigen() # using Eigen will usually make inference faster
save_images = False

print \
'''
This demo shows why HDP-HMMs fail without some kind of temporal regularization
(in the form of a sticky bias or duration modeling): without setting the number
of states to be the correct number a priori, lots of extra states are usually
intsantiated.
'''

#### Data generation
# Set parameters
N = 4
T = 500
obs_dim = 2

# Set observation hyperparameters (which control the randomly-sampled mean and
# covariance matrices for each state)
obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.15,
                'nu_0':obs_dim+2}

# Construct the true observation and duration distributions
true_obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(N)]

# Build the true HSMM model
truemodel = pyhsmm.models.HMM(alpha=4.,gamma=4.,
                              obs_distns=true_obs_distns)

# Sample data from the true model
data, labels = truemodel.generate(T)

# Plot the truth
plt.figure()
truemodel.plot()
plt.gcf().suptitle('True HMM')
if save_images:
    plt.savefig('truth.png')

#### Posterior inference
# Set the weak limit truncation level. This is essentially the maximum number of
# states that can be learned
Nmax = 10

# Construct the observation and duration distribution objects, which set priors
# over parameters and then infer parameter values.
obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(Nmax)]

# Build the HMM model that will represent the posterior
posteriormodel = pyhsmm.models.HMM(alpha=6.,gamma=6.,
                                   obs_distns=obs_distns)
posteriormodel.add_data(data)

# Resample the model 100 times, printing a dot at each iteration and plotting
# every so often
plt.figure()
plot_every = 25
for idx in progprint_xrange(101):
    if (idx % plot_every) == 0:
        posteriormodel.plot()
        plt.gcf().suptitle('inferred HMM after %d iterations' % idx)
        if save_images:
            plt.savefig('posterior_sample_%d.png' % idx)

    posteriormodel.resample_model()

plt.show()
