from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import pyhsmm
from pyhsmm.util.stats import cov
from pyhsmm.util.text import progprint_xrange


Nsuper = 3
Nsub = 5
T = 5000
obs_dim = 2

try:
    import brewer2mpl
    plt.set_cmap(brewer2mpl.get_map('Set1','qualitative',max(3,min(8,Nsuper))).mpl_colormap)
except:
    pass

obs_hypparams = dict(
        mu_0=np.zeros(obs_dim),
        sigma_0=np.eye(obs_dim),
        kappa_0=0.01,
        nu_0=obs_dim+10,
        )

dur_hypparams = dict(
        r_discrete_distn=np.r_[0,0,0,0,0,1.,1.,1.],
        alpha_0=20,
        beta_0=2,
        )

true_obs_distnss = [[pyhsmm.distributions.Gaussian(**obs_hypparams) for substate in xrange(Nsub)]
        for superstate in xrange(Nsuper)]

true_dur_distns = [pyhsmm.distributions.NegativeBinomialIntegerRVariantDuration(
    **dur_hypparams) for superstate in range(Nsuper)]

truemodel = pyhsmm.models.HSMMIntNegBinVariantSubHMMs(
        init_state_concentration=6,
        alpha=10.,gamma=10.,
        sub_alpha=10.,sub_gamma=10.,
        obs_distnss=true_obs_distnss,
        dur_distns=true_dur_distns)

data, _ = truemodel.generate(T)

truemodel.plot()
plt.gcf().suptitle('truth')


##################
#  set up model  #
##################

Nmaxsuper = 2*Nsuper
Nmaxsub = 2*Nsub

obs_distnss = \
        [[pyhsmm.distributions.Gaussian(**obs_hypparams)
            for substate in range(Nmaxsub)] for superstate in range(Nmaxsuper)]

dur_distns = \
        [pyhsmm.distributions.NegativeBinomialIntegerRVariantDuration(
            **dur_hypparams) for superstate in range(Nmaxsuper)]

model  = pyhsmm.models.HSMMIntNegBinVariantSubHMMs(
        init_state_concentration=6.,
        alpha=6.,gamma=6.,
        sub_alpha=6,sub_gamma=6,
        obs_distnss=obs_distnss,
        dur_distns=dur_distns)

model.add_data(data,left_censoring=True)

###############
#  inference  #
###############
for itr in progprint_xrange(50):
    model.resample_model()

plt.figure()
model.plot()
plt.gcf().suptitle('fit')

plt.show()

