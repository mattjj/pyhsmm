from __future__ import division
import numpy as np
from numpy import newaxis as na
from matplotlib import pyplot as plt
from os.path import join, dirname, isfile

from pyhsmm import models, distributions
from pyhsmm.util.general import sgd_passes, hold_out, get_file
from pyhsmm.util.text import progprint_xrange, progprint

np.random.seed(0)
datapath = str(join(dirname(__file__),'svi_data.gz'))

### load data

if not isfile(datapath):
    print 'download svi_data.gz data and put it in examples/'
    print 'https://github.com/mattjj/example_data'
    import sys; sys.exit(1)

print 'loading data...'
alldata = np.loadtxt(datapath)
allseqs = np.array_split(alldata,250)
datas, heldout = hold_out(allseqs,0.05)
training_size = sum(data.shape[0] for data in datas)
print '...done!'

print '%d total frames' % sum(data.shape[0] for data in alldata)
print 'split into %d training and %d test sequences' % (len(datas),len(heldout))

### inference!

Nmax = 20
obs_hypparams = dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.2,nu_0=5)

hmm = models.HMM(
        obs_distns=[distributions.Gaussian(**obs_hypparams) for i in range(Nmax)],
        alpha=10.,init_state_concentration=1.)

scores = []
sgdseq = sgd_passes(tau=0,kappa=0.7,datalist=datas)
for t, (data, rho_t) in progprint(enumerate(sgdseq)):
    hmm.meanfield_sgdstep(data, data.shape[0] / training_size, rho_t)

    if t % 10 == 0:
        scores.append(hmm.log_likelihood(heldout))

plt.figure()
plt.plot(scores)

plt.show()

