from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import pyhsmm
from pyhsmm.util.stats import cov
from pyhsmm.util.text import progprint_xrange

###############
#  load data  #
###############

# data = np.loadtxt('example-data.txt')[:2500]
# T,obs_dim = data.shape

N = 4
T = 1000
obs_dim = 2

obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.05,
                'nu_0':obs_dim+5}

dur_hypparams = {'alpha_0':2*20,
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
        [pyhsmm.distributions.NegativeBinomialIntegerRVariantDuration(
            np.r_[0,0,0,0,0,1.,1.,1.], # discrete distribution uniform over {6,7,8}
            9,1, # average geometric success probability 1/(9+1)
            ) for state in range(Nmax)]

model  = pyhsmm.models.HSMMIntNegBinVariant(
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

for itr in progprint_xrange(50):
    model.Viterbi_EM_step()

##########
#  plot  #
##########

plt.figure()
model.plot()

plt.show()
