from __future__ import division
from builtins import range
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
np.random.seed(0)

from matplotlib import pyplot as plt
import matplotlib
import os
matplotlib.rcParams['font.size'] = 8

from pyhsmm import models
from pyhsmm.basic import distributions
from pyhsmm.util.text import progprint_xrange

print('''
This demo shows how HDP-HMMs can fail when the underlying data has state
persistence without some kind of temporal regularization (in the form of a
sticky bias or duration modeling): without setting the number of states to be
the correct number a priori, lots of extra states can be intsantiated.

BUT the effect is much more relevant on real data (when the data doesn't exactly
fit the model). Maybe this demo should use multinomial emissions...
''')
###############
#  load data  #
###############

data = np.loadtxt(os.path.join(os.path.dirname(__file__),'example-data.txt'))[:2500]
T = data.shape[0]

#########################
#  posterior inference  #
#########################

# Set the weak limit truncation level
Nmax = 25

# and some hyperparameters
obs_dim = data.shape[1]
obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.25,
                'nu_0':obs_dim+2}

### HDP-HMM without the sticky bias
obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
posteriormodel = models.WeakLimitHDPHMM(alpha=6.,gamma=6.,
                                               init_state_concentration=1.,
                                   obs_distns=obs_distns)
posteriormodel.add_data(data)

for idx in progprint_xrange(100):
    posteriormodel.resample_model()

posteriormodel.plot()
plt.gcf().suptitle('HDP-HMM sampled model after 100 iterations')

### HDP-HMM with "sticky" initialization
obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
posteriormodel = models.WeakLimitHDPHMM(alpha=6.,gamma=6.,
                                               init_state_concentration=1.,
                                   obs_distns=obs_distns)

# Start with a "sticky" state sequence
z_init = np.random.randint(0, Nmax, size=(T//5)).repeat(5)
posteriormodel.add_data(data, stateseq=z_init)

# Initialize the parameters of the model, holding the stateseq fixed
for _ in progprint_xrange(10):
    posteriormodel.resample_parameters()

for idx in progprint_xrange(100):
    posteriormodel.resample_model()

posteriormodel.plot()
plt.gcf().suptitle('HDP-HMM (sticky initialization) sampled model after 100 iterations')

### Sticky-HDP-HMM

obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
posteriormodel = models.WeakLimitStickyHDPHMM(
        kappa=50.,alpha=6.,gamma=6.,init_state_concentration=1.,
        obs_distns=obs_distns)
posteriormodel.add_data(data)

for idx in progprint_xrange(100):
    posteriormodel.resample_model()

posteriormodel.plot()
plt.gcf().suptitle('Sticky HDP-HMM sampled model after 100 iterations')

plt.show()
