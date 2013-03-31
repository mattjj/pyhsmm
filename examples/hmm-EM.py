from __future__ import division
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 8

import pyhsmm
from pyhsmm.util.text import progprint_xrange

save_images = False

#### Data generation
# Set parameters
N = 4
T = 500
obs_dim = 2

# Set observation hyperparameters (which control the randomly-sampled mean and
# covariance matrices for each state)
obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.1,
                'nu_0':obs_dim+2}

# Construct the true observation and duration distributions
true_obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(N)]

# Build the true HMM model
truemodel = pyhsmm.models.StickyHMM(
        kappa=200.,alpha=50.,gamma=50.,init_state_concentration=50., # big numbers for uniform transitions
        obs_distns=true_obs_distns)

# Sample data from the true model
data, labels = truemodel.generate(T)

# Plot the truth
plt.figure()
truemodel.plot()
plt.gcf().suptitle('True HMM')
if save_images:
    plt.savefig('truth.png')

#### EM

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(N)]

# Build the HMM model that will represent the fitmodel
fitmodel = pyhsmm.models.HMM(
        alpha=50.,gamma=50.,init_state_concentration=50., # these are only used for initialization
        obs_distns=obs_distns)
fitmodel.add_data(data)

print 'Gibbs sampling for initialization'

for idx in progprint_xrange(25):
    fitmodel.resample_model()

plt.figure()
fitmodel.plot()
plt.gcf().suptitle('Gibbs-sampled initialization')

print 'EM'

for idx in progprint_xrange(100):
    # instead of blindly running iterations, we could also quit on convergence
    fitmodel.EM_step()

plt.figure()
fitmodel.plot()
plt.gcf().suptitle('EM fit')
plt.show()

