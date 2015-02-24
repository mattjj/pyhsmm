from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from matplotlib import pyplot as plt

import pyhsmm
from pyhsmm.util.text import progprint_xrange

np.random.seed(0)

#####################
#  data generation  #
#####################

N = 4
T = 1000
obs_dim = 600

obs_hypparams = dict(
        mu_0=np.zeros(obs_dim),
        nus_0=1./10*np.ones(obs_dim),
        alphas_0=np.ones(obs_dim),
        betas_0=np.ones(obs_dim),
        )

dur_hypparams = dict(alpha_0=2*30,beta_0=2)

true_obs_distns = [pyhsmm.distributions.DiagonalGaussian(**obs_hypparams) for state in range(N)]
true_dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(N)]

truemodel = pyhsmm.models.HSMM(alpha=6.,init_state_concentration=6.,
                               obs_distns=true_obs_distns,
                               dur_distns=true_dur_distns)

data, labels = truemodel.generate(T)

plt.figure()
truemodel.plot()
plt.gcf().suptitle('True HSMM')

# !!! get the changepoints !!!
# NOTE: usually these would be estimated by some external process; here I'm
# totally cheating and just getting them from the truth
temp = np.concatenate(((0,),truemodel.states_list[0].durations.cumsum()))
changepoints = zip(temp[:-1],temp[1:])
changepoints[-1] = (changepoints[-1][0],T) # because last duration might be censored
print 'segments:'
print changepoints

#########################
#  posterior inference  #
#########################

Nmax = 25

obs_distns = [pyhsmm.basic.models.MixtureDistribution(
        alpha_0=1.,
        components=[pyhsmm.distributions.DiagonalGaussian(**obs_hypparams)
            for component in xrange(2)]) for state in xrange(Nmax)]

# obs_distns = [pyhsmm.distributions.DiagonalGaussian(**obs_hypparams) for state in xrange(Nmax)]

dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in xrange(Nmax)]

posteriormodel = pyhsmm.models.DiagGaussGMMHSMMPossibleChangepointsSeparateTrans(
        alpha=6.,init_state_concentration=6.,
        obs_distns=obs_distns,dur_distns=dur_distns)

# posteriormodel = pyhsmm.models.DiagGaussHSMMPossibleChangepointsSeparateTrans(
#         alpha=6.,init_state_concentration=6.,
#         obs_distns=obs_distns,dur_distns=dur_distns)

posteriormodel.add_data(data,changepoints,group_id=0,stateseq=labels)
posteriormodel.init_meanfield_from_sample()

plt.figure()
posteriormodel.plot()

def normalize(A):
    return A / A.sum(1)[:,None]
plt.matshow(truemodel.trans_distn.trans_matrix)
from pyhsmm.util.general import count_transitions
plt.matshow(count_transitions(truemodel.states_list[0].stateseq_norep))
plt.matshow(normalize(posteriormodel.trans_distns[0].exp_expected_log_trans_matrix)[:N,:N])

plt.show()

