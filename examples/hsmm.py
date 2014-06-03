from __future__ import division
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import copy, os

import pyhsmm
from pyhsmm.util.text import progprint_xrange

SAVE_FIGURES = False

print \
'''
This demo shows the HDP-HSMM in action. Its iterations are slower than those for
the (Sticky-)HDP-HMM, but explicit duration modeling can be a big advantage for
conditioning the prior or for discovering structure in data.
'''

###############
#  load data  #
###############

T = 1000
data = np.loadtxt(os.path.join(os.path.dirname(__file__),'example-data.txt'))[:T]

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
dur_hypparams = {'alpha_0':2*30,
                 'beta_0':2}

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
        alpha=6.,gamma=6., # these can matter; see concentration-resampling.py
        init_state_concentration=6., # pretty inconsequential
        obs_distns=obs_distns,
        dur_distns=dur_distns)
posteriormodel.add_data(data,trunc=60) # duration truncation speeds things up when it's possible

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
