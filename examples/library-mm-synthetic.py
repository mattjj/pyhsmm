from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

from pyhsmm.models import HSMMIntNegBinVariant
from pyhsmm.basic.models import MixtureDistribution, FrozenMixtureDistribution
from pyhsmm.basic.distributions import Gaussian, NegativeBinomialIntegerRVariantDuration
from pyhsmm.util.text import progprint_xrange

#############################
#  generate synthetic data  #
#############################

states_in_hsmm = 4
components_per_GMM = 4
component_hyperparameters = dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.025,nu_0=3)

GMMs = [MixtureDistribution(
    alpha_0=4.,
    components=[Gaussian(**component_hyperparameters) for i in range(components_per_GMM)])
    for state in range(states_in_hsmm)]

true_dur_distns = [NegativeBinomialIntegerRVariantDuration(np.r_[0.,0,0,1,1,1,1,1],alpha_0=5.,beta_0=5.)
        for state in range(states_in_hsmm)]

truemodel = HSMMIntNegBinVariant(
        init_state_concentration=10.,
        alpha=6.,gamma=6.,
        obs_distns=GMMs,
        dur_distns=true_dur_distns)

data, truelabels = truemodel.generate(1000)

#####################################
#  set up FrozenMixture components  #
#####################################

# list of all Gaussians
component_library = [c for m in GMMs for c in m.components]
library_size = len(component_library)

# get all likelihoods
all_likelihoods = FrozenMixtureDistribution.get_all_likelihoods(component_library,data)

# initialize weights to indicator on one component
init_weights = np.eye(library_size)

FrozenGMMs = [FrozenMixtureDistribution(
    all_likelihoods=all_likelihoods,
    all_data=data,
    components=component_library,
    alpha_0=4.,
    weights=row)
    for row in init_weights]

################
#  build HSMM  #
################

dur_distns = [NegativeBinomialIntegerRVariantDuration(np.r_[0.,0,0,1,1,1,1,1],alpha_0=5.,beta_0=5.)
        for state in range(library_size)]

model = HSMMIntNegBinVariant(
        init_state_concentration=10.,
        alpha=6.,gamma=6.,
        obs_distns=FrozenGMMs,
        dur_distns=dur_distns)

model.add_data(np.arange(data.shape[0]))

##################
#  infer things  #
##################

for i in progprint_xrange(50):
    model.resample_model()

plt.figure()
truemodel.plot()
plt.gcf().suptitle('truth')

plt.figure()
model.plot()
plt.gcf().suptitle('inferred')

plt.show()
