from __future__ import division
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import copy, os

import pyhsmm
from pyhsmm.util.text import progprint_xrange

###################
#  generate data  #
###################

T = 1000
obs_dim = 2
N = 4

obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.25,
                'nu_0':obs_dim+2}
dur_hypparams = {'alpha_0':10*1,
                 'beta_0':10*100}

true_obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams)
        for state in range(N)]
true_dur_distns = [pyhsmm.distributions.GeometricDuration(**dur_hypparams)
        for state in range(N)]

truemodel = pyhsmm.models.GeoHSMM(
        alpha=6.,
        init_state_concentration=6.,
        obs_distns=true_obs_distns,
        dur_distns=true_dur_distns)

data, labels = truemodel.generate(T)

plt.figure()
truemodel.plot()



temp = np.concatenate(((0,),truemodel.states_list[0].durations.cumsum()))
changepoints = zip(temp[:-1],temp[1:])
changepoints[-1] = (changepoints[-1][0],T) # because last duration might be censored



#########################
#  posterior inference  #
#########################

Nmax = 25

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
dur_distns = [pyhsmm.distributions.GeometricDuration(**dur_hypparams) for state in range(Nmax)]

posteriormodel = pyhsmm.models.GeoHSMMPossibleChangepoints(
        alpha=6.,
        init_state_concentration=6.,
        obs_distns=obs_distns,
        dur_distns=dur_distns)
posteriormodel.add_data(data,changepoints=changepoints)

for idx in progprint_xrange(50):
    posteriormodel.resample_model()

plt.figure()
posteriormodel.plot()

plt.show()
