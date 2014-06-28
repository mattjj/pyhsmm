from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

import pyhsmm
from pyhsmm.util.text import progprint_xrange
from pyhsmm.util.general import rle

N = 4
obs_hypparams = dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.05,nu_0=4)
T = 2000

def normalize(mat):
    return np.nan_to_num(mat / mat.sum(1)[:,None])

### generate data

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(N)]
truemodel = pyhsmm.models.HMM(alpha=N,init_state_concentration=1.,obs_distns=obs_distns)
truemodel.trans_distn.trans_matrix = normalize(np.eye(N) + np.diag(0.05*np.ones(N-1),1) + np.diag(0.05 * np.ones(N-1),-1))

data, labels = truemodel.generate(T)

# cheat to get the changepoints
_, durations = rle(labels)
temp = np.concatenate(((0,),durations.cumsum()))
changepoints = zip(temp[:-1],temp[1:])
changepoints[-1] = (changepoints[-1][0],T) # because last duration might be censored
print 'segments:'
print changepoints

### inference (using true obs_distns)

Nmax = 25
model = pyhsmm.models.HMMPossibleChangepoints(
        alpha=N/2.,init_state_concentration=1.,
        obs_distns=[pyhsmm.distributions.Gaussian(**obs_hypparams)
            for state in range(Nmax)])

model.add_data(data,changepoints=changepoints)

for itr in progprint_xrange(100):
    model.resample_model()

plt.figure()
truemodel.plot()
plt.figure()
model.plot()

plt.show()

