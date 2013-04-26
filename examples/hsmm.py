from __future__ import division
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import copy

import pyhsmm
from pyhsmm.util.text import progprint_xrange

SAVE_FIGURES = False

print \
'''
This demo shows the HDP-HSMM in action. Its iterations are slower than those for
the (Sticky-)HDP-HMM, but explicit duration modeling can be a big advantage for
conditioning the prior or for discovering structure in data.
'''

#####################
#  data generation  #
#####################

N = 4
T = 500
obs_dim = 2

obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.3,
                'nu_0':obs_dim+5}

dur_hypparams = {'alpha_0':2*30,
                 'beta_0':2}

# Construct the true observation and duration distributions
true_obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(N)]
true_dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(N)]

# Build the true HSMM model
truemodel = pyhsmm.models.HSMM(
        alpha=6.,gamma=6.,
        init_state_concentration=10.,
        obs_distns=true_obs_distns,
        dur_distns=true_dur_distns)

# Sample data from the true model
data, labels = truemodel.generate(T)

# Plot the truth
plt.figure()
truemodel.plot()
plt.gcf().suptitle('True HSMM')

#########################
#  posterior inference  #
#########################

# Set the weak limit truncation level
Nmax = 25

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

posteriormodel = pyhsmm.models.HSMMEigen(
        alpha=6.,gamma=6., # these can matter; better to sample over them (concentration-resampling.py)
        init_state_concentration=6., # pretty inconsequential
        obs_distns=obs_distns,
        dur_distns=dur_distns,
        trunc=60) # duration truncation speeds things up when it's possible
posteriormodel.add_data(data)

models = []
for idx in progprint_xrange(150):
    posteriormodel.resample_model()
    if (idx+1) % 10 == 0:
        models.append(copy.deepcopy(posteriormodel))

fig = plt.figure()
for idx, model in enumerate(models):
    plt.clf()
    model.plot()
    plt.gcf().suptitle('HDP-HSMM sampled after %d iterations' % (10*(idx+1)))
    if SAVE_FIGURES:
        plt.savefig('iter_%.3d.png' % (10*(idx+1)))

plt.show()
