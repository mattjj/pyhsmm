from __future__ import division
import numpy as np
from functools import wraps
from nose.plugins.attrib import attr

from pyhsmm import models, distributions

##########
#  util  #
##########

def runmultiple(n):
    def dec(fn):
        @wraps(fn)
        def wrapper():
            for i in range(n):
                yield fn
        return wrapper
    return dec

@attr('slow')
@runmultiple(1)
def discrete_geweke_test():
    Nstates = 2
    Nemissions = 2
    alpha = 3.
    init_state_concentration=3.
    T = 10
    num_iter = 10000
    num_checks = 10

    obs_distns = [distributions.Categorical(K=Nemissions,alpha_0=1.)
            for _ in range(Nstates)]

    hmm = models.HMM(
            alpha=alpha,init_state_concentration=init_state_concentration,
            obs_distns=obs_distns)

    # generate state sequences from the prior
    prior_stateseqs = []
    for itr in xrange(num_iter):
        hmm.resample_model() # sample parameters from the prior
        _, stateseq = hmm.generate(T,keep=False)
        prior_stateseqs.append(stateseq)
    assert len(hmm.states_list) == 0
    prior_stateseqs = np.array(prior_stateseqs)


    # generate state sequences using Gibbs
    hmm.generate(T,keep=True)
    s = hmm.states_list[0]

    gibbs_stateseqs = []
    for itr in xrange(num_iter):
        s.generate_obs() # resamples data given state sequence, obs params
        hmm.resample_model() # resamples everything else as usual
        gibbs_stateseqs.append(s.stateseq)
    gibbs_stateseqs = np.array(gibbs_stateseqs)

    # test that they look similar by checking probability of co-assignment
    time_indices = np.arange(T)
    for itr in xrange(num_checks):
        i,j = np.random.choice(time_indices,replace=False,size=2)
        prior_prob_of_coassignment = (prior_stateseqs[:,i] == prior_stateseqs[:,j]).mean()
        gibbs_prob_of_coassignment = (gibbs_stateseqs[:,i] == gibbs_stateseqs[:,j]).mean()

        assert np.isclose(
                prior_prob_of_coassignment,gibbs_prob_of_coassignment,
                rtol=0.025,atol=0.025,
                )

