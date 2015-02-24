from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

from pyhsmm import models, distributions

np.seterr(invalid='raise')
obs_hypparams = dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.05,nu_0=5)

### generate data

num_modes = 3

true_obs_distns = [distributions.Gaussian(**obs_hypparams) for i in range(num_modes)]
data = np.concatenate([true_obs_distns[i % num_modes].rvs(25) for i in range(25)])

## inference!

hmm = models.HMM(
        obs_distns=[distributions.Gaussian(**obs_hypparams) for i in range(num_modes*3)],
        alpha=3.,init_state_concentration=1.)
hmm.add_data(data)
hmm.meanfield_coordinate_descent_step()
scores = [hmm.meanfield_coordinate_descent_step() for i in range(50)]
scores = np.array(scores)

plt.figure()
hmm.plot()

plt.figure()
plt.plot(scores)

def normalize(A):
    return A / A.sum(1)[:,None]
plt.matshow(normalize(hmm.trans_distn.exp_expected_log_trans_matrix))

plt.show()
