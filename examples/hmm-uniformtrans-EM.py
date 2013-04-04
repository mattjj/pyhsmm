from __future__ import division
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import copy

import pyhsmm
import pyhsmm.internals.transitions as transitions
pyhsmm.internals.states.use_eigen() # makes HMMs faster, message passing done in C++ with Eigen
from pyhsmm.util.text import progprint_xrange

#####################
#  data generation  #
#####################

# Set parameters
N = 4
T = 1000
obs_dim = 2

obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.1,
                'nu_0':obs_dim+5}

true_obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(N)]
true_trans_distn = transitions.UniformTransitionsFixedSelfTrans(
        pi=pyhsmm.distributions.Multinomial(alpha_0=10.*N,K=N),
        lmbda=0.9)

truemodel = pyhsmm.models.HMM(
        init_state_concentration=6.,
        obs_distns=true_obs_distns,
        trans_distn=true_trans_distn)

data, labels = truemodel.generate(T)

plt.figure()
truemodel.plot()
plt.gcf().suptitle('True model')


###################
#  model fitting  #
###################


lmbda = 0.9 # timescale
Ns = [3,4,5]
num_EM_attempts = 5

BICs = []
examplemodels = []
for Nfit in Ns:
    print ''
    print '### Fitting with %d states ###' % Nfit
    print ''

    obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(Nfit)]
    trans_distn = transitions.UniformTransitionsFixedSelfTrans(
            pi=pyhsmm.distributions.Multinomial(alpha_0=10.*Nfit,K=Nfit),
            lmbda=lmbda)

    fitmodel = pyhsmm.models.HMM(
            init_state_concentration=6.,
            obs_distns=obs_distns,
            trans_distn=trans_distn)

    fitmodel.add_data(data)

    theseBICs = []
    for i in range(num_EM_attempts):
        print 'Gibbs sampling initialization'
        for itr in progprint_xrange(50):
            fitmodel.resample_model()

        print 'EM fit'
        for itr in progprint_xrange(50):
            fitmodel.EM_step()

        theseBICs.append(fitmodel.BIC())
    examplemodels.append(copy.deepcopy(fitmodel))
    BICs.append(theseBICs)

plt.figure()
plt.errorbar(
        x=Ns,
        y=[np.mean(x) for x in BICs],
        yerr=[np.std(x) for x in BICs],
        )
plt.xlabel('num states')
plt.ylabel('BIC')

plt.figure()
examplemodels[np.argmin([np.min(x) for x in BICs])].plot()
plt.title('a decent model')

plt.show()

