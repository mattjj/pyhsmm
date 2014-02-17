from __future__ import division
import numpy as np
from numpy import newaxis as na
from matplotlib import pyplot as plt

from pyhsmm import models, distributions
from pyhsmm.util.general import sgd_steps, hold_out
from pyhsmm.util.text import progprint_xrange, progprint

np.random.seed(0)

### load data

print 'loading data...'
alldata = np.array_split(np.loadtxt('svi_data2.txt'),250)
datas, heldout = hold_out(alldata,0.05)
T = sum(data.shape[0] for data in datas)
print '...done!'

### inference!

Nmax = 20
obs_hypparams = dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.2,nu_0=5)

hmm = models.DATruncHDPHMM(
        obs_distns=[distributions.Gaussian(**obs_hypparams) for i in range(Nmax)],
        alpha=10.,gamma=10.,init_state_concentration=1.)

scores = []
stepsizes = sgd_steps(tau=0,kappa=0.7,nsteps=len(datas))
for t, (data, rho_t) in progprint(enumerate(zip(datas,stepsizes))):
    hmm.meanfield_sgdstep(data, data.shape[0] / T, rho_t)

    if t % 10 == 0:
        scores.append(hmm.log_likelihood(heldout))

plt.plot(scores)
plt.show()
