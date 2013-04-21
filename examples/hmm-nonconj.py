from __future__ import division
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 8

import pyhsmm
from pyhsmm.util.text import progprint_xrange

#####################
#  data generation  #
#####################

# Set parameters
N = 4
T = 1000
obs_dim = 2

obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.1,
                'nu_0':obs_dim+5}

dur_hypparams = {'alpha_0':2*30,
                 'beta_0':2}

true_obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(N)]
true_dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(N)]

truemodel = pyhsmm.models.HSMM(alpha=6.,gamma=6.,init_state_concentration=6.,
                              obs_distns=true_obs_distns,
                              dur_distns=true_dur_distns)

data, labels = truemodel.generate(T)

plt.figure()
truemodel.plot()
plt.gcf().suptitle('True model')


#########################
#  posterior inference  #
#########################

# Set the weak limit truncation level
Nmax = 25

### Sticky-HDP-HMM

obs_distns = [pyhsmm.distributions.GaussianNonConj(
            np.zeros(2), # mean hyperparameter for mus
            3*np.eye(2), # covariance hyperparameter for mus
            obs_dim+5,   # pseudocounts hyperparameter for sigmas
            np.eye(2),   # mean hyperparameter for sigmas
            ) for state in xrange(Nmax)]
posteriormodel = pyhsmm.models.StickyHMMEigen(kappa=50.,alpha=6.,gamma=6.,init_state_concentration=6.,
                                   obs_distns=obs_distns)
posteriormodel.add_data(data)

for idx in progprint_xrange(100):
    posteriormodel.resample_model()

plt.figure()
posteriormodel.plot()
plt.gcf().suptitle('Sticky HDP-HMM sampled model after 100 iterations')

plt.show()
