from __future__ import division
import numpy as np
from numpy import newaxis as na
from inspect import getargspec
from functools import wraps
import itertools
from nose.plugins.attrib import attr

from pyhsmm import models as m, distributions as d

##########
#  util  #
##########

def likelihood_check(obs_distns,trans_matrix,init_distn,data,target_val):
    for cls in [m.HMMPython, m.HMM]:
        hmm = cls(alpha=6.,init_state_concentration=1, # placeholders
                obs_distns=obs_distns)
        hmm.trans_distn.trans_matrix = trans_matrix
        hmm.init_state_distn.weights = init_distn
        hmm.add_data(data)

        # test default log_likelihood method

        assert np.isclose(target_val, hmm.log_likelihood())

        # manual tests of the several message passing methods

        states = hmm.states_list[-1]

        states.clear_caches()
        states.messages_forwards_normalized()
        assert np.isclose(target_val,states._normalizer)

        states.clear_caches()
        states.messages_forwards_log()
        assert np.isinf(target_val) or np.isclose(target_val,states._normalizer)

        states.clear_caches()
        states.messages_backwards_log()
        assert np.isinf(target_val) or np.isclose(target_val,states._normalizer)

        # test held-out vs in-model

        assert np.isclose(target_val, hmm.log_likelihood(data))

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

def runmultiple(n):
    def dec(fn):
        @wraps(fn)
        def wrapper():
            for i in range(n):
                yield fn
        return wrapper
    return dec

###########
#  tests  #
###########

@attr('hmm','likelihood','messages','basic')
def like_hand_test_1():
    likelihood_check(
        obs_distns=[d.Categorical(weights=row) for row in np.eye(2)],
        trans_matrix=np.eye(2),
        init_distn=np.array([1.,0.]),
        data=np.zeros(10,dtype=int),
        target_val=0.)

@attr('hmm','likelihood','messages','basic','robust')
def like_hand_test_2():
    likelihood_check(
        obs_distns=[d.Categorical(weights=row) for row in np.eye(2)],
        trans_matrix=np.eye(2),
        init_distn=np.array([0.,1.]),
        data=np.zeros(10,dtype=int),
        target_val=np.log(0.))

@attr('hmm','likelihood','messages','basic')
def like_hand_test_3():
    likelihood_check(
        obs_distns=[d.Categorical(weights=row) for row in np.eye(2)],
        trans_matrix=np.array([[0.,1.],[1.,0.]]),
        init_distn=np.array([1.,0.]),
        data=np.tile([0,1],5).astype(int),
        target_val=0.)

@attr('hmm','likelihood','messages','basic')
def like_hand_test_4():
    likelihood_check(
        obs_distns=[d.Categorical(weights=row) for row in np.eye(2)],
        trans_matrix=np.array([[0.,1.],[1.,0.]]),
        init_distn=np.array([1.,0.]),
        data=np.tile([0,1],5).astype(int),
        target_val=0.)

@attr('hmm','likelihood','messages','basic')
def like_hand_test_5():
    likelihood_check(
        obs_distns=[d.Categorical(weights=row) for row in np.eye(2)],
        trans_matrix=np.array([[0.9,0.1],[0.2,0.8]]),
        init_distn=np.array([1.,0.]),
        data=np.tile((0,1),5),
        target_val=5*np.log(0.1) + 4*np.log(0.2))

@attr('hmm','slow','likelihood','messages')
@runmultiple(3)
def discrete_exhaustive_test():
    model = random_model(2)
    obs_distns = [d.Categorical(K=3,alpha_0=1.),d.Categorical(K=3,alpha_0=1.)]
    stateseq = np.random.randint(2,size=10)
    data = np.array([obs_distns[a].rvs() for a in stateseq])
    target_val = compute_likelihood_enumeration(obs_distns=obs_distns,data=data,**model)
    likelihood_check(target_val=target_val,data=data,obs_distns=obs_distns,**model)

@attr('hmm','slow','likelihood','messages')
@runmultiple(3)
def gaussian_exhaustive_test():
    model = random_model(3)
    obs_distns = [
            d.Gaussian(mu=np.random.randn(2),sigma=np.eye(2)),
            d.Gaussian(mu=np.random.randn(2),sigma=np.eye(2)),
            d.Gaussian(mu=np.random.randn(2),sigma=np.eye(2))]
    stateseq = np.random.randint(3,size=10)
    data = np.vstack([obs_distns[a].rvs() for a in stateseq])
    target_val = compute_likelihood_enumeration(obs_distns=obs_distns,data=data,**model)
    likelihood_check(target_val=target_val,data=data,obs_distns=obs_distns,**model)

