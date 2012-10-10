from __future__ import division
import numpy as np

import pyhsmm
import pyhsmm.basic.distributions as distns
import pyhsmm.parallel as parallel
from pyhsmm.util.text import progprint_xrange

parallel.alldata = {0:np.zeros(10),1:np.zeros(15)}
parallel.allchangepoints = {0:[(0,5),(5,10)],1:[(0,3),(3,15)]}

from IPython.parallel import Client
dv = Client()[:]
dv['alldata'] = parallel.alldata
dv['allchangepoints'] = parallel.allchangepoints

posteriormodel = pyhsmm.models.HSMMPossibleChangepoints(
        alpha=6,gamma=6,init_state_concentration=6.,
        obs_distns=[distns.ScalarGaussianNIX(mu_0=0.,kappa_0=0.2,sigmasq_0=1,nu_0=10)
            for ss in range(5)],
        dur_distns=[distns.PoissonDuration(alpha_0=2*10,beta_0=2)
            for s in range(5)],
        )

for data_id in parallel.alldata.keys():
    posteriormodel.add_data_parallel(data_id)

for itr in progprint_xrange(20):
    posteriormodel.resample_model_parallel()
