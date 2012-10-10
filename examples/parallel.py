from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import pyhsmm
import pyhsmm.basic.distributions as distns
import pyhsmm.parallel as parallel
from pyhsmm.util.text import progprint_xrange

# NOTE: before running this file, start some engines
# for local engines, you can do something like this
# ipcluster start --n=4

### data generation
N = 4
T = 500
obs_dim = 2

# Set observation hyperparameters (which control the randomly-sampled mean and
# covariance matrices for each state)
obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.15,
                'nu_0':obs_dim+2}

dur_hypparams = {'alpha_0':2*30,
                 'beta_0':2}

# Construct the true observation and duration distributions
true_obs_distns = [distns.Gaussian(**obs_hypparams) for state in xrange(N)]
true_dur_distns = [distns.PoissonDuration(**dur_hypparams) for state in xrange(N)]

# Build the true HSMM model
truemodel = pyhsmm.models.HSMM(alpha=4.,gamma=4.,init_state_concentration=6.,
                        obs_distns=true_obs_distns,
                        dur_distns=true_dur_distns)

# Sample data from the true model
data, labels = truemodel.generate(T)

# Plot the truth
plt.figure()
truemodel.plot()
plt.gcf().suptitle('True HSMM')

# !!! get the changepoints !!!
# TODO estimate the changepoints instead
temp = np.concatenate(((0,),truemodel.states_list[0].durations.cumsum()))
changepoints = zip(temp[:-1],temp[1:])
changepoints[-1] = (changepoints[-1][0],T) # because last duration might be censored

### set up parallel stuff

parallel.alldata = {0:data}
parallel.allchangepoints = {0:changepoints}

from IPython.parallel import Client
dv = Client()[:]
dv['alldata'] = parallel.alldata
dv['allchangepoints'] = parallel.allchangepoints

### build posterior model

posteriormodel = pyhsmm.models.HSMMPossibleChangepoints(
        alpha=6,gamma=6,init_state_concentration=6.,
        obs_distns=[distns.Gaussian(**obs_hypparams) for state in xrange(N)],
        dur_distns=[distns.PoissonDuration(**dur_hypparams) for state in xrange(N)])

for data_id in parallel.alldata.keys():
    posteriormodel.add_data_parallel(data_id)

for itr in progprint_xrange(50):
    posteriormodel.resample_model_parallel()

plt.figure()
posteriormodel.plot()
plt.gcf().suptitle('Sampled')

plt.show()
