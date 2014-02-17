from __future__ import division
import numpy as np
from numpy import newaxis as na
from matplotlib import pyplot as plt

from pyhsmm import models, distributions
from pyhsmm.util.text import progprint_xrange, progprint

import time

np.seterr(invalid='raise')
obs_hypparams = dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.2,nu_0=5)

### generate data

num_modes = 4
A = 25*np.eye(num_modes) + np.ones((num_modes,num_modes))
A /= A.sum(1)[:,na]

true_obs_distns = [distributions.Gaussian(**obs_hypparams) for i in range(num_modes)]
truehmm = models.HMM(
        obs_distns=true_obs_distns,
        alpha=20.,init_state_concentration=10.)
truehmm.trans_distn.trans_matrix = A

print 'generating data...'
datas = [truehmm.generate(2000,keep=False)[0] for itr in range(100)]
print '... done'

## inference!

def sgd_steps(tau,kappa,nsteps):
    assert 0.5 < kappa <= 1
    assert tau >= 0
    for t in xrange(nsteps):
        yield (t+tau)**(-kappa)

Nmax = num_modes*3

hmm = models.HMM(
        obs_distns=[distributions.Gaussian(**obs_hypparams) for i in range(Nmax)],
        alpha=3.,init_state_concentration=1.)
hmm.add_data(datas[0])

scores = [hmm.meanfield_coordinate_descent_step() for i in progprint_xrange(2)]

newscores = []
newscores.extend([hmm._meanfield_sgdstep_batch(rho_t)
    for rho_t in progprint(sgd_steps(tau=1,kappa=0.8,nsteps=100))])

plt.figure()
hmm.plot()

plt.figure()
plt.plot(scores,'b-')
plt.plot(range(len(scores),len(scores)+len(newscores)-1),newscores[:-1],'r-')

# def normalize(A):
#     return A / A.sum(1)[:,None]
# plt.matshow(normalize(hmm.trans_distn.exp_expected_log_trans_matrix))


plt.show()

