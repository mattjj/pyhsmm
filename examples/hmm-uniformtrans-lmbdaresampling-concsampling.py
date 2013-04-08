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
                'kappa_0':0.05,
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

# Set the weak limit truncation level
Nmax = 25

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(Nmax)]
trans_distn = transitions.UniformTransitions(
        pi=pyhsmm.distributions.MultinomialConcentration(1.,2.,K=Nmax),
        lmbda_a_0=9.,lmbda_b_0=1.)

posteriormodel = pyhsmm.models.HMM(
        init_state_concentration=Nmax, # doesn't matter with one observation sequence
        obs_distns=obs_distns,
        trans_distn=trans_distn)

posteriormodel.add_data(data)

savedstuff = []
for idx in progprint_xrange(50):
    posteriormodel.resample_model()
    savedstuff.append(
            (
                trans_distn.pi.weights.copy(),
                trans_distn.lmbda,
                posteriormodel.states_list[0].stateseq.copy()
            ))

plt.figure()
savedstuff = savedstuff[10:]
cluster_numbers = [(np.bincount(s[2]) > T*0.05).sum() for s in savedstuff]
lmbda_values = [s[1] for s in savedstuff]
plt.plot(cluster_numbers,lmbda_values,'bx')
plt.xlim(np.min(cluster_numbers)-1,np.max(cluster_numbers)+1)
plt.gcf().suptitle('Samples')

plt.show()
