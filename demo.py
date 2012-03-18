# Imports

import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for us
from matplotlib import pyplot as plt

import hsmm, observations, durations, stats_util
from text_util import progprint_xrange

#### Data generation
# Set parameters
N = 4
T = 500
obs_dim = 2
# Set duration parameters to be (10,20,30,40)
durparams = [10.*(idx+1) for idx in xrange(N)]
# Set observation hyperparameters (which control the random mean and covariance
# matrices for each state)
obs_hypparams = (np.zeros(obs_dim),np.eye(obs_dim),0.25,obs_dim+2)

# Construct the true observation and duration distributions
truth_obs_distns = [observations.gaussian(*stats_util.sample_niw(*obs_hypparams)) for state in xrange(N)]
truth_dur_distns = [durations.poisson(lmbda=param) for param in durparams]

# Build the true HSMM model
truthmodel = hsmm.hsmm(T,truth_obs_distns,truth_dur_distns)

# Sample data from the true model
data, labels = truthmodel.generate()

#### Posterior inference
# Set the weak limit truncation level. This is essentially the maximum 
# number of states that can be learned
Nmax = 10

# Construct the observation and duration distribution objects, which set
# priors over parameters and then infer parameter values.
obs_distns = [observations.gaussian(*stats_util.sample_niw(*obs_hypparams)) for state in xrange(Nmax)]
dur_distns = [durations.poisson() for state in xrange(Nmax)]

# Build the HSMM model that will represent the posterior 
posteriormodel = hsmm.hsmm(T,obs_distns,dur_distns)

# Resample the model 100 times, printing a dot at each iteration
plot_every = 25
for idx in progprint_xrange(100):
    if idx != 0 and (idx % plot_every) == 0:
        posteriormodel.plot(data)
        plt.title('HSMM after %d iterations' % idx)

    posteriormodel.resample(data)
