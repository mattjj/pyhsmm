#!/usr/bin/env python
from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from matplotlib import pyplot as plt

import pyhsmm
from pyhsmm.util.text import progprint_xrange

save_images = False

### data generation
N = 4
T = 500
obs_dim = 2

# Set observation hyperparameters (which control the randomly-sampled mean and
# covariance matrices for each state)
obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.15,
                'nu_0':obs_dim+2}

dur_hypparams = {'alpha_0':5,
                 'beta_0':1/8}

# Construct the true observation and duration distributions
true_obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(N)]
true_dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in xrange(N)]

# Build the true HSMM model
truemodel = pyhsmm.models.hsmm(alpha=4.,gamma=4.,
                        obs_distns=true_obs_distns,
                        dur_distns=true_dur_distns)

# Sample data from the true model
data, labels = truemodel.generate(T)

# Plot the truth
plt.figure()
truemodel.plot()
plt.gcf().suptitle('True HSMM')

# !!! get the changepoints !!!
# TODO estimate the changepoints instead
temp = np.concatenate(((0,),truemodel.states_list[0].durations.cumsum()))
changepoints = zip(temp[:-1],temp[1:])
changepoints[-1] = (changepoints[-1][0],T) # because last duration might be censored
print 'segments:'
print changepoints

### build the posterior
Nmax = 10

# Construct the observation and duration distribution objects, which set priors
# over parameters and then infer parameter values.
obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(Nmax)]
dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in xrange(N)]

# build new hsmm_possiblechangepoints model that will represent the posterior
posteriormodel = pyhsmm.models.hsmm_possiblechangepoints(alpha=6.,gamma=6.,
        obs_distns=obs_distns,dur_distns=dur_distns,trunc=70)
posteriormodel.add_data(data,changepoints)


# Resample the model 100 times, printing a dot at each iteration and plotting
# every so often
plot_every = 10
fig = plt.figure()
for idx in progprint_xrange(101):
    if (idx % plot_every) == 0:
        plt.gcf().clf()
        posteriormodel.plot()
        plt.gcf().suptitle('inferred HSMM after %d iterations (arbitrary colors)' % idx)
        plt.draw()
        if save_images:
            plt.savefig('posterior_sample_%d.png' % idx)

    posteriormodel.resample()

plt.show()
