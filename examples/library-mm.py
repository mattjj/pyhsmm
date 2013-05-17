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

f = np.load("/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/Test Data/TMT_50p_mixtures_and_data.npz")
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
    # NOTE: alpha_0 here controls the prior on the number of mixture components
    # within each GMM emission distribution. a lower alpha_0 makes it more
    # expensive to use more mixture components. it roughly corresponds to the
    # expected number of components, though it's much more flexible than that.
    alpha_0=6.,
    # NOTE: an alternative to setting alpha_0 directly is to put a gamma prior
    # over it and sample it. to do that, comment out the alpha_0 line above and
    # use the line below to set a gamma prior. see the call to plt.plot below
    # to get a sense for how these parameters set the relative probabilities of
    # different alpha_0.
    # a_0=1.,b_0=1./6,
    weights=weights,
    ) for weights in init_weights]

## NOTE: run this block to visualize a gamma prior
# from matplotlib import pyplot as plt
# import scipy.stats as stats
# a_0=1.   # making a_0 >1 will change its shape into a blob
# b_0=1./5 # mean is a_0 / b_0
# t = np.arange(0.01,25,0.01)
# plt.figure() # plt.plot(t,stats.gamma.pdf(t,a_0,scale=1./b_0))
# plt.hist(np.random.gamma(a_0,1./b_0,size=1000),normed=True)

####################
#  build HDP-HSMM  #
####################

dur_distns = [d.NegativeBinomialIntegerRVariantDuration(
    np.r_[0.,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
    alpha_0=5.,beta_0=5.) for state in range(library_size)]

model = pyhsmm.models.HSMMIntNegBinVariant(
        init_state_concentration=10., # this parameter is irrelevant for us
        # NOTE: alpha and gamma are the concentration parameters for the HDP
        # over the HSMM transition matrix, and they work analogously to alpha_0
        # for the GMMs described above. roughly, gamma controls the total number
        # of states while alpha controls the diversity of the transition
        # distributions.
        # alpha=10.,gamma=10.,
        # NOTE: as with a_0 and b_0 for the GMMs described above, we can also
        # put gamma priors over alpha and gamma by commenting out the direct
        # alpha= and gamma= lines and using these instead
        alpha_a_0=1.,alpha_b_0=1./10,
        gamma_a_0=1.,gamma_b_0=1./10,
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

