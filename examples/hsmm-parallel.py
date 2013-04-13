from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import pyhsmm
import pyhsmm.basic.distributions as distns
import pyhsmm.parallel as parallel
from pyhsmm.util.text import progprint_xrange

# NOTE: before running this file, start some engines for local engines:
# ipcluster start --n=4

#####################
#  data generation  #
#####################

N = 4
T = 1000
obs_dim = 2

obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.1,
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
data0, _ = truemodel.generate(T)
data1, _ = truemodel.generate(T)
data2, _ = truemodel.generate(T)
data3, _ = truemodel.generate(T)

# Plot the truth
plt.figure()
truemodel.plot()
plt.gcf().suptitle('True HSMM')

########################
#  parallel inference  #
########################

### boilerplate parallel setup: this 'alldata' dict needs to be present on every engine
parallel.alldata = {0:data0, 1:data1, 2:data2, 3:data3}

from IPython.parallel import Client
dv = Client()[:]
dv['alldata'] = parallel.alldata

### building and running the model looks about the same as it usually does

posteriormodel = pyhsmm.models.HSMM(
        alpha=6,gamma=6,init_state_concentration=6.,
        obs_distns=[distns.Gaussian(**obs_hypparams) for state in xrange(N)],
        dur_distns=[distns.PoissonDuration(**dur_hypparams) for state in xrange(N)])

for data_id in parallel.alldata.keys():
    posteriormodel.add_data_parallel(data_id) # NOTE: add_data_parallel

for itr in progprint_xrange(100):
    posteriormodel.resample_model_parallel() # NOTE: resample_model_parallel

plt.figure()
posteriormodel.plot()
plt.gcf().suptitle('Sampled model after 100 iterations')

plt.show()
