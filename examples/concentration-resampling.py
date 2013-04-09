from __future__ import division
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import scipy.stats as stats

import pyhsmm
from pyhsmm.util.text import progprint_xrange

#####################
#  data generation  #
#####################

N = 4
T = 1000
obs_dim = 2

obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.3,
                'nu_0':obs_dim+5}

dur_hypparams = {'alpha_0':2*30,
                 'beta_0':2}

# Construct the true observation and duration distributions
true_obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(N)]
true_dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(N)]

# Build the true HSMM model
truemodel = pyhsmm.models.HSMM(alpha=6.,gamma=6.,init_state_concentration=6.,
                               obs_distns=true_obs_distns,
                               dur_distns=true_dur_distns)

# Sample data from the true model
data, labels = truemodel.generate(T)

# Plot the truth
plt.figure()
truemodel.plot()
plt.gcf().suptitle('True HSMM')

#########################
#  posterior inference  #
#########################

Nmax = 25

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

posteriormodel = pyhsmm.models.HSMM(
        # NOTE: instead of passing in alpha_0 and gamma_0, we pass in parameters
        # for priors over those concentration parameters
        alpha_a_0=1.,alpha_b_0=1./4,
        gamma_a_0=1.,gamma_b_0=1./4,
        init_state_concentration=6.,
        obs_distns=obs_distns,
        dur_distns=dur_distns,trunc=70)
posteriormodel.add_data(data)

for idx in progprint_xrange(100):
    posteriormodel.resample_model()

plt.figure()
posteriormodel.plot()
plt.gcf().suptitle('Sampled after 100 iterations')

plt.figure()
t = np.linspace(0.01,30,1000)
plt.plot(t,stats.gamma.pdf(t,1.,scale=4.)) # NOTE: numpy/scipy scale is inverted compared to my scale
plt.title('Prior on concentration parameters')

plt.show()
