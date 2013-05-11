from __future__ import division
import numpy as np

from pyhsmm.util.general import rle
from pyhsmm.util.text import progprint_xrange

import pyhsmm
from pyhsmm import distributions as d
from pyhsmm.basic.pybasicbayes.models import FrozenMixtureDistribution

###############
#  load data  #
###############

T = 1000

f = np.load('/Users/mattjj/Dropbox/Test Data/TMT_50p_mixtures_and_data.npz')
mus = f['mu']
sigmas = f['sigma']
data = f['data'][:T]

library_size, obs_dim = mus.shape
labels = f['labels'][:T]

# boost diagonal a bit to make it better behaved
for i in range(sigmas.shape[0]):
    sigmas[i] += np.eye(obs_dim)*1e-7

#####################################
#  build observation distributions  #
#####################################

component_library = \
        [d.Gaussian(
            mu=mu,sigma=sigma,
            mu_0=np.zeros(obs_dim),sigma_0=np.eye(obs_dim),nu_0=obs_dim+10,kappa_0=1., # dummies, not used
            ) for mu,sigma in zip(mus,sigmas)]

# frozen mixtures never change their component parameters so we can compute the
# likelihoods all at once in the front
all_likelihoods = FrozenMixtureDistribution.get_all_likelihoods(
        components=component_library,
        data=data)

# initialize to each state corresponding to just one gaussian component
init_weights = np.eye(library_size)

hsmm_obs_distns = [FrozenMixtureDistribution(
    all_likelihoods=all_likelihoods,
    all_data=data, # for plotting
    components=component_library,
    alpha_0=3.,
    weights=weights,
    ) for weights in init_weights]

####################
#  build HDP-HSMM  #
####################

dur_distns = [d.NegativeBinomialIntegerRVariantDuration(
    np.r_[0.,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
    alpha_0=5.,beta_0=5.) for state in range(library_size)]

model = pyhsmm.models.HSMMIntNegBinVariant(
        init_state_concentration=10.,
        alpha=6.,gamma=6.,
        obs_distns=hsmm_obs_distns,
        dur_distns=dur_distns)
model.trans_distn.max_likelihood([rle(labels)[0]])

#############
#  run it!  #
#############

# NOTE: data is a dummy, just indices being passed around because we precompute
# all likelihoods
model.add_data(np.arange(T))

for itr in progprint_xrange(25):
    model.resample_model()

# niter = 25
# W = np.zeros((niter,library_size,library_size))
# for itr in progprint_xrange(niter):
#     model.Viterbi_EM_step()
#     W[itr] = np.array([od.weights.weights for od in model.obs_distns])

