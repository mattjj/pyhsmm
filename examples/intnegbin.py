from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import pyhsmm
from pyhsmm.util.stats import cov
from pyhsmm.util.text import progprint_xrange

###############
#  load data  #
###############

data = np.loadtxt('example-data.txt')[:2500]
T,obs_dim = data.shape

##################
#  set up model  #
##################

Nmax = 50

obs_distns = \
        [pyhsmm.distributions.Gaussian(
            mu_0=data.mean(0),
            sigma_0=0.5*cov(data),
            kappa_0=0.5,
            nu_0=obs_dim+3) for state in range(Nmax)]

dur_distns = \
        [pyhsmm.distributions.NegativeBinomialIntegerRVariantDuration(
            np.r_[0,0,0,0,0,1.,1.,1.], # discrete distribution uniform over {6,7,8}
            9,1, # average geometric success probability 1/(9+1)
            ) for state in range(Nmax)]

model  = pyhsmm.models.HSMMIntNegBin(
        init_state_concentration=10.,
        alpha=6.,gamma=6.,
        obs_distns=obs_distns,
        dur_distns=dur_distns)
model.add_data(data)

##############
#  resample  #
##############

for itr in progprint_xrange(10):
    model.resample_model()

################
#  viterbi EM  #
################

for itr in progprint_xrange(25):
    model.Viterbi_EM_step()

##########
#  plot  #
##########

plt.figure()
model.plot()

plt.show()
