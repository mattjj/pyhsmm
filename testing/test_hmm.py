from __future__ import division
import numpy as np
from numpy import newaxis as na
from inspect import getargspec
import itertools
from nose.plugins.attrib import attr

from pyhsmm import models as m, distributions as d

# a utility for writing test generators
def make_nose_tuple(func,**kwargs):
    return (func,) + tuple(kwargs[k] for k in getargspec(func).args)

######################
#  likelihood tests  #
######################

# this helper creates a pyhsmm.HMM instance and checks its likelihood value
# on data against target_val
def _likelihood_helper(obs_distns,trans_matrix,init_distn,data,target_val):
    hmm = m.HMM(
            alpha=6,gamma=6,init_state_concentration=1, # placeholders
            obs_distns=obs_distns)
    hmm.trans_distn.trans_matrix = trans_matrix
    hmm.init_state_distn.weights = init_distn
    hmm.add_data(data)
    assert np.isclose(hmm.log_likelihood(), target_val)

# this method computes HMM likelihood via exhaustive stateseq enumeration
def compute_likelihood_enumeration(obs_distns,trans_matrix,init_distn,data):
    N = len(obs_distns)
    T = len(data)

    Al = np.log(trans_matrix)
    aBl = np.hstack([o.log_likelihood(data)[:,na] for o in obs_distns])

    tot = -np.inf
    for stateseq in itertools.product(range(N),repeat=T):
        loglike = 0.
        loglike += np.log(init_distn[stateseq[0]])
        for a,b in zip(stateseq[:-1],stateseq[1:]):
            loglike += Al[a,b]
        for t,a in enumerate(stateseq):
            loglike += aBl[t,a]
        tot = np.logaddexp(tot,loglike)
    return tot

def random_model(nstates):
    init_distn = np.random.dirichlet(np.ones(nstates))
    trans_matrix = np.vstack([np.random.dirichlet(np.ones(nstates)) for i in range(nstates)])
    return dict(init_distn=init_distn,trans_matrix=trans_matrix)


def likelihood_hand_tests():
    yield make_nose_tuple(_likelihood_helper,
            obs_distns=[d.Categorical(weights=row) for row in np.eye(2)],
            trans_matrix=np.array([[0.9,0.1],[0.1,0.9]]),
            init_distn=np.array([1.,0.]),
            data=np.zeros(10,dtype=int),
            target_val=9*np.log(0.9))

    yield make_nose_tuple(_likelihood_helper,
            obs_distns=[d.Categorical(weights=row) for row in np.eye(2)],
            trans_matrix=np.array([[0.9,0.1],[0.2,0.8]]),
            init_distn=np.array([1.,0.]),
            data=np.tile((0,1),5),
            target_val=5*np.log(0.1) + 4*np.log(0.2))


@attr('slow')
def likelihood_exhaustive_tests():

    ### discrete data

    for i in range(2):
        model = random_model(2)
        obs_distns = [d.Categorical(K=3,alpha_0=1.),d.Categorical(K=3,alpha_0=1.)]
        stateseq = np.random.randint(2,size=10)
        data = np.array([obs_distns[a].rvs() for a in stateseq])
        target_val = compute_likelihood_enumeration(obs_distns=obs_distns,data=data,**model)
        yield make_nose_tuple(_likelihood_helper,target_val=target_val,data=data,
                obs_distns=obs_distns,**model)

    # Gaussian data

    for i in range(2):
        model = random_model(3)
        obs_distns = [
                d.Gaussian(mu=np.random.randn(2),sigma=np.eye(2)),
                d.Gaussian(mu=np.random.randn(2),sigma=np.eye(2)),
                d.Gaussian(mu=np.random.randn(2),sigma=np.eye(2))]
        stateseq = np.random.randint(3,size=10)
        data = np.vstack([obs_distns[a].rvs() for a in stateseq])
        target_val = compute_likelihood_enumeration(obs_distns=obs_distns,data=data,**model)
        yield make_nose_tuple(_likelihood_helper,target_val=target_val,data=data,
                obs_distns=obs_distns,**model)
