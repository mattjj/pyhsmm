from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

from pyhsmm import models, distributions
from pyhsmm.util.text import progprint_xrange, progprint

import time

np.seterr(invalid='raise')
obs_hypparams = dict(mu_0=np.zeros(12),sigma_0=np.eye(12),kappa_0=0.4,nu_0=15)

### generate data

num_modes = 30
true_obs_distns = [distributions.Gaussian(**obs_hypparams) for i in range(num_modes)]
truehmm = models.HMM(
        obs_distns=true_obs_distns,
        alpha=20.,init_state_concentration=10.)
datas = [truehmm.generate(2000,keep=False)[0] for itr in range(100)]
heldout = truehmm.generate(2000,keep=False)[0]

def sgd_steps(tau,kappa,nsteps):
    assert 0.5 < kappa <= 1
    assert tau >= 0
    for t in xrange(nsteps):
        yield (t+tau)**(-kappa)

## inference!


# # hmm = models.HMM(
# #         obs_distns=[distributions.Gaussian(**obs_hypparams) for i in range(num_modes*3)],
# #         alpha=3.,init_state_concentration=1.)
# hmm = models.DATruncHDPHMM(
#         obs_distns=[distributions.Gaussian(**obs_hypparams) for i in range(num_modes*2)],
#         alpha=20.,gamma=20.,init_state_concentration=10.)
# hmm.add_data(data)
# # scores = [hmm.meanfield_coordinate_descent_step() for i in progprint_xrange(2)]
# scores = []


# newscores = []
# newscores.extend([hmm._meanfield_sgdstep_batch(rho_t)
#     for rho_t in progprint(sgd_steps(tau=20,kappa=0.8,nsteps=100))])

# plt.figure()
# hmm.plot()

# plt.figure()
# plt.plot(scores,'b-')
# plt.plot(range(len(scores),len(scores)+len(newscores)-1),newscores[:-1],'r-')

# def normalize(A):
#     return A / A.sum(1)[:,None]
# plt.matshow(normalize(hmm.trans_distn.exp_expected_log_trans_matrix))


# TODO set alpha/gamma to be smaller after the first few iters?

plainhmm = models.DATruncHDPHMM(
        obs_distns=[distributions.Gaussian(**obs_hypparams) for i in range(num_modes*3)],
        alpha=50.,gamma=50.,init_state_concentration=10.)

nsamples = 10
likess = []
itertimes = []
for minibatch, rho_t in progprint(zip(datas*3,sgd_steps(tau=1,kappa=0.8,nsteps=3*len(datas)))):
    tic = time.time()
    plainhmm.meanfield_sgdstep(minibatch,1./len(datas),rho_t)
    itertimes.append(time.time() - tic)

    likes = []
    for itr in xrange(nsamples):
        plainhmm._resample_from_mf()
        likes.append(plainhmm.log_likelihood(heldout))
    likess.append(likes)

plt.figure()
plainhmm.plot_observations()

alldata = np.concatenate(datas).dot(plainhmm.obs_distns[0].__class__.plotting_subspace_basis.T)
plt.plot(alldata[:,0],alldata[:,1],'k.',alpha=0.1)
plainhmm.plot_observations()

plt.figure()
likess = np.asarray(likess)
plt.plot(np.cumsum(itertimes),likess.mean(1))




plainhmm = models.HMM(
        obs_distns=[distributions.Gaussian(**obs_hypparams) for i in range(num_modes*3)],
        alpha=20.,init_state_concentration=10.)

nsamples = 10
likess = []
itertimes = []
for minibatch, rho_t in progprint(zip(datas*4,sgd_steps(tau=1,kappa=0.8,nsteps=4*len(datas)))):
    tic = time.time()
    plainhmm.meanfield_sgdstep(minibatch,1./len(datas),rho_t)
    itertimes.append(time.time() - tic)

    likes = []
    for itr in xrange(nsamples):
        plainhmm._resample_from_mf()
        likes.append(plainhmm.log_likelihood(heldout))
    likess.append(likes)

likess = np.asarray(likess)
plt.plot(np.cumsum(itertimes),likess.mean(1))







batchhmm = models.HMM(
        obs_distns=[distributions.Gaussian(**obs_hypparams) for i in range(num_modes*3)],
        alpha=50.,init_state_concentration=10.)

for data in datas:
    batchhmm.add_data(data)

batchlikess = []
batchtimes = []
for itr in progprint_xrange(10):
    tic = time.time()
    batchhmm.meanfield_coordinate_descent_step()
    batchtimes.append(time.time() - tic)

    likes = []
    for itr in xrange(nsamples):
        batchhmm._resample_from_mf()
        likes.append(batchhmm.log_likelihood(heldout))
    batchlikess.append(likes)

batchlikess = np.asarray(batchlikess)
plt.plot(np.cumsum(batchtimes),batchlikess.mean(1))


plt.show()

# TODO get the timings by getting rid of the likes calculation (temporarily)

