from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import copy

import pyhsmm

N = 50
T = 10000
obs_dim = 2

obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.01,
                'nu_0':obs_dim+5}

# Construct the true observation and duration distributions
true_obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(N)]
true_dur_distns = [pyhsmm.distributions.NegativeBinomialIntegerRVariantDuration(np.r_[1.,0.,1.,0.,1.,0.,1.],100.,5.)
                        for state in range(N)]

# Build the true HSMM model
truemodel = pyhsmm.models.HSMM(
        alpha=6.,gamma=6.,
        init_state_concentration=10.,
        obs_distns=true_obs_distns,
        dur_distns=true_dur_distns)

# Sample data from the true model
data, labels = truemodel.generate(T)

### posterior

usualmodel = pyhsmm.models.HSMM(
        init_state_concentration=10.,
        obs_distns=copy.deepcopy(true_obs_distns),
        dur_distns=copy.deepcopy(true_dur_distns),
        trans_distn=copy.deepcopy(truemodel.trans_distn))
usualmodel.add_data(data)

specialmodel = pyhsmm.models.HSMMIntNegBin(
        init_state_concentration=10., # pretty inconsequential
        obs_distns=copy.deepcopy(true_obs_distns),
        dur_distns=copy.deepcopy(true_dur_distns),
        trans_distn=copy.deepcopy(truemodel.trans_distn))
specialmodel.add_data(data)

specialbetalslow, specialbetalslow2 = specialmodel.states_list[0].messages_backwards_python()
specialbetal, specialsuperbetal = specialmodel.states_list[0].messages_backwards()

plt.figure()
plt.plot(specialbetalslow[:,0],'bx',label='slow 1')
# plt.plot(specialbetalslow2[:,0],'r+-',label='slow 2')
plt.plot(specialbetal[:,0],'g+',label='good guys')
plt.legend(loc='best')

specialmodel.states_list[0].sample_forwards(specialbetal,specialsuperbetal)
plt.figure()
specialmodel.plot()

print truemodel.states_list[0].stateseq == specialmodel.states_list[0].stateseq

plt.show()

