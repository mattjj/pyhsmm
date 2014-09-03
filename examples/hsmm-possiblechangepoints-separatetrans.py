from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from matplotlib import pyplot as plt

import pyhsmm
from pyhsmm.util.text import progprint_xrange, progprint
from pyhsmm.util.general import sgd_passes

# TODO generate data from a separatetrans model

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

datas = [truemodel.generate(T)[0] for itr in range(2)]

plt.figure()
truemodel.plot()
plt.gcf().suptitle('True HSMM')


# !!! get the changepoints !!!
# NOTE: usually these would be estimated by some external process; here I'm
# totally cheating and just getting them from the truth
changepointss = []
for s in truemodel.states_list:
    temp = np.concatenate(((0,),s.durations.cumsum()))
    changepoints = zip(temp[:-1],temp[1:])
    changepoints[-1] = (changepoints[-1][0],T) # because last duration might be censored
    changepointss.append(changepoints)

#########################
#  posterior inference  #
#########################

Nmax = 10

obs_distns = [pyhsmm.distributions.DiagonalGaussian(**obs_hypparams) for state in xrange(Nmax)]
dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in xrange(Nmax)]

# posteriormodel = pyhsmm.models.WeakLimitHDPHSMMPossibleChangepointsSeparateTrans(
#         alpha=4.,gamma=4.,init_state_concentration=4.,
#         obs_distns=obs_distns,dur_distns=dur_distns)

posteriormodel = pyhsmm.models.HSMMPossibleChangepointsSeparateTrans(
        alpha=4.,init_state_concentration=4.,
        obs_distns=obs_distns,dur_distns=dur_distns)

### sampling

for idx, (data, changepoints) in enumerate(zip(datas,changepointss)):
    posteriormodel.add_data(data=data,changepoints=changepoints,group_id=idx)

for idx in progprint_xrange(100):
    posteriormodel.resample_model()

for s in posteriormodel.states_list:
    s.Viterbi()

plt.figure()
posteriormodel.plot()


### SVI

# sgdseq = sgd_passes(
#         tau=0,kappa=0.7,npasses=20,
#         datalist=zip(range(len(datas)),datas,changepointss))

# for (group_id,data,changepoints), rho_t in progprint(sgdseq):
#     posteriormodel.meanfield_sgdstep(
#             data,changepoints=changepoints,group_id=group_id,
#             minibatchfrac=1./len(datas),stepsize=rho_t)

# plt.figure()
# for idx, (data, changepoints) in enumerate(zip(datas,changepointss)):
#     posteriormodel.add_data(data,changepoints=changepoints,group_id=idx)
#     posteriormodel.states_list[-1].mf_Viterbi()
# posteriormodel.plot()

### mean field

# for idx, (data, changepoints) in enumerate(zip(datas,changepointss)):
#     posteriormodel.add_data(data=data,changepoints=changepoints,group_id=idx)

# scores = []
# for idx in progprint_xrange(50):
#     scores.append(posteriormodel.meanfield_coordinate_descent_step(joblib_jobs=1))

# plt.figure()
# plt.plot(scores)
# plt.figure()
# posteriormodel.plot()


plt.show()

