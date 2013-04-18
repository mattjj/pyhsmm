from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import pyhsmm
from pyhsmm.util.stats import cov
from pyhsmm.util.text import progprint_xrange

###############
#  load data  #
###############

data = np.loadtxt('example-data.txt')
T, obs_dim = data.shape

##################
#  set up model  #
##################

Nmax = 20

obs_distns = \
        [pyhsmm.distributions.Gaussian(
            mu_0=data.mean(0),
            sigma_0=0.5*cov(data),
            kappa_0=0.5,
            nu_0=obs_dim+3) for state in range(Nmax)]

dur_distns = \
        [pyhsmm.distributions.NegativeBinomialDuration(7*100,1./100,50*10,50*1)
                for state in range(Nmax)]

model = pyhsmm.models.HSMMGeoApproximation(
        init_state_concentration=Nmax, # doesn't matter for one chain
        alpha=6.,gamma=6.,
        obs_distns=obs_distns,
        dur_distns=dur_distns,
        trunc=150) # NOTE: blows up this isn't long enough wrt the NegativeBinomial parameters
model.add_data(data)

##############
#  resample  #
##############

for itr in progprint_xrange(25):
    model.resample_model()

##########
#  plot  #
##########

plt.figure()
model.plot()

plt.show()
