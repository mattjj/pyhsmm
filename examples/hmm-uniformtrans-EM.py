from __future__ import division
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 8

import pyhsmm
import pyhsmm.internals.transitions as transitions
pyhsmm.internals.states.use_eigen() # makes HMMs faster, message passing done in C++ with Eigen
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

true_obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(N)]
true_trans_distn = transitions.UniformTransitionsFixedSelfTrans(
        pi=pyhsmm.distributions.Multinomial(alpha_0=10.*N,K=N),
        lmbda=0.9)

truemodel = pyhsmm.models.HMM(
        init_state_concentration=6.,
        obs_distns=true_obs_distns,
        trans_distn=true_trans_distn)

data, labels = truemodel.generate(T)

plt.figure()
truemodel.plot()
plt.gcf().suptitle('True model')


#########################
#  posterior inference  #
#########################

Nfit = 4

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(Nfit)]
trans_distn = transitions.UniformTransitionsFixedSelfTrans(
        pi=pyhsmm.distributions.Multinomial(alpha_0=10.*Nfit,K=Nfit),
        lmbda=0.9)

fitmodel = pyhsmm.models.HMM(
        init_state_concentration=6.,
        obs_distns=obs_distns,
        trans_distn=trans_distn)

fitmodel.add_data(data)

print 'Gibbs initialization'
for idx in progprint_xrange(10):
    fitmodel.resample_model()

print 'EM'
for idx in progprint_xrange(100):
    fitmodel.EM_step()

plt.figure()
fitmodel.plot()
plt.gcf().suptitle('EM fit')

plt.figure()
plt.plot(trans_distn.pi.weights)

plt.show()

