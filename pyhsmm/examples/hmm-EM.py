from __future__ import division
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import matplotlib
import os
matplotlib.rcParams['font.size'] = 8

import pyhsmm
from pyhsmm.util.text import progprint_xrange

save_images = False

#### load data

data = np.loadtxt(os.path.join(os.path.dirname(__file__),'example-data.txt'))

#### EM

N = 4
obs_dim = data.shape[1]
obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.25,
                'nu_0':obs_dim+2}

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(N)]

# Build the HMM model that will represent the fitmodel
fitmodel = pyhsmm.models.HMM(
        alpha=50.,init_state_concentration=50., # these are only used for initialization
        obs_distns=obs_distns)
fitmodel.add_data(data)

print 'Gibbs sampling for initialization'

for idx in progprint_xrange(25):
    fitmodel.resample_model()

plt.figure()
fitmodel.plot()
plt.gcf().suptitle('Gibbs-sampled initialization')

print 'EM'

likes = fitmodel.EM_fit()

plt.figure()
fitmodel.plot()
plt.gcf().suptitle('EM fit')

plt.figure()
plt.plot(likes)
plt.gcf().suptitle('log likelihoods during EM')

plt.show()

