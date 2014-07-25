from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from matplotlib import pyplot as plt

import pyhsmm
from pyhsmm.util.text import progprint_xrange

from pyhsmm.basic.pybasicbayes.parallel_tempering import ParallelTempering

#####################
#  data generation  #
#####################

N = 4
T = 1000
obs_dim = 2

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

data[100:120] = data[200:220] = np.nan

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

# obs_distns = [pyhsmm.basic.models.MixtureDistribution(
#         alpha_0=1.,
#         components=[pyhsmm.distributions.DiagonalGaussian(**obs_hypparams)
#             for component in xrange(2)]) for state in xrange(Nmax)]

obs_distns = [pyhsmm.distributions.DiagonalGaussian(**obs_hypparams) for state in xrange(Nmax)]

dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in xrange(Nmax)]

# posteriormodel = pyhsmm.models.DiagGaussGMMHSMMPossibleChangepointsSeparateTrans(
#         alpha=6.,init_state_concentration=6.,
#         obs_distns=obs_distns,dur_distns=dur_distns)

posteriormodel = pyhsmm.models.DiagGaussHSMMPossibleChangepointsSeparateTrans(
        alpha=6.,init_state_concentration=6.,
        obs_distns=obs_distns,dur_distns=dur_distns)

posteriormodel.add_data(data,changepoints,group_id=0)


pt = ParallelTempering(posteriormodel,[100.])
pt.run(100,10)
for (T1,T2), count in pt.swapcounts.items():
    print 'temperature pair (%0.2f, %0.2f) swapped %d times' % (T1,T2,count)
    print '(%0.3f%% of the time)' % ((count / pt.itercount) * 100)
    print

plt.figure()
pt.unit_temp_model.plot()

plt.show()

