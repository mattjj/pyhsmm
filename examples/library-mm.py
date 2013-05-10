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

# dummy parameters, not used

T = 1000

f = np.load('/Users/mattjj/Dropbox/Test Data/TMT_50p_mixtures_and_data.npz')
mus = f['mu']
sigmas = f['sigma']
data = f['data'][:T]

library_size, obs_dim = mus.shape
labels = f['labels'][:T]

for i in range(sigmas.shape[0]):
    sigmas[i] += np.eye(obs_dim)*1e-6

#####################################
#  build observation distributions  #
#####################################

component_library = [d.Gaussian(
    mu=mu,sigma=sigma,
    mu_0=np.zeros(obs_dim),sigma_0=np.eye(obs_dim),nu_0=obs_dim+10,kappa_0=1., # dummies, not used
    ) for mu,sigma in zip(mus,sigmas)]

# frozen mixtures never change their component parameters so we can compute the
# likelihoods all at once in the front
likelihoods = FrozenMixtureDistribution.get_all_likelihoods(
        components=component_library,
        data=data)

# init_weights = np.ones((library_size,library_size))
init_weights = np.eye(library_size)

hsmm_obs_distns = [FrozenMixtureDistribution(
    likelihoods=likelihoods,
    alpha_0=3.,
    components=component_library,
    weights=weights,
    ) for weights in init_weights]

dur_distns = [d.NegativeBinomialIntegerRVariantDuration(np.ones(20.),alpha_0=5.,beta_0=5.)
        for state in range(library_size)]

####################
#  build HDP-HSMM  #
####################

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

# niter = 25
# W = np.zeros((niter,library_size,library_size))
# for itr in progprint_xrange(niter):
#     model.Viterbi_EM_step()
#     W[itr] = np.array([od.weights.weights for od in model.obs_distns])


# TODO why dont the weights look updated?
# TODO why doesn't it work good?
