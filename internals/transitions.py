from __future__ import division
import numpy as np
import scipy.stats as stats
from numpy.random import random
from numpy import newaxis as na
import operator

from ..util.general import rle

# TODO the code organization in this file needs improvement... could use a Dir
# distribution object?

class HDPHSMMTransitions(object):
    '''
    HDPHSMM transition distribution class.
    Uses a weak-limit HDP prior. Zeroed diagonal to forbid self-transitions.

    Hyperparameters follow the notation in Fox et al., except the definitions of
    alpha and gamma are reversed for no reason.
        alpha, gamma

    Parameters are the shared transition vector beta, the full transition matrix,
    and the matrix with the diagonal zeroed.
        beta, A
    '''

    def __init__(self,state_dim,alpha,gamma,beta=None,A=None,fullA=None):
        self.alpha = alpha
        self.gamma = gamma
        self.state_dim = state_dim
        if A is None or fullA is None or beta is None:
            self.resample()
        else:
            self.A = A
            self.beta = beta
            self.fullA = fullA

    def resample(self,stateseqs=[]):
        if type(stateseqs) != type([]):
            stateseqs = [stateseqs]

        states_noreps = map(operator.itemgetter(0),map(rle, stateseqs))

        if not any(len(states_norep) >= 2 for states_norep in states_noreps):
            # if there is no data we just sample from the prior
            self.beta = np.random.dirichlet((self.alpha / self.state_dim)*np.ones(self.state_dim))

            self.fullA = np.random.dirichlet(self.beta*self.gamma,size=self.state_dim)
            self.A = (1.-np.eye(self.state_dim)) * self.fullA
            self.A /= self.A.sum(1)[:,na]

            assert not np.isnan(self.beta).any()
            assert not np.isnan(self.fullA).any()
            assert (self.A.diagonal() == 0).all()
        else:
            data = np.zeros((self.state_dim,self.state_dim))
            for states_norep in states_noreps:
                for idx in xrange(len(states_norep)-1):
                    data[states_norep[idx],states_norep[idx+1]] += 1
            assert (data.diagonal() == 0).all()

            froms = np.sum(data,axis=1)
            self_transitions = [np.random.geometric(1-pi_ii,size=n) if n > 0 else 0
                    for pi_ii,n in zip(self.fullA.diagional(),froms)]
            augmented_data = data + np.diag(self_transitions)

            m = np.zeros((self.state_dim,self.state_dim))
            for (i,j), n in np.ndenumerate(augmented_data):
                m[i,j] = (np.random.rand(n) < self.alpha*self.beta[j] \
                        / (np.arange(n) + self.alpha*self.beta[j])).sum()
            self.m = m # save it for possible use in any child classes

            self.beta = np.random.dirichlet(self.alpha/self.state_dim + m.sum(0))
            self.fullA = np.random.gamma(self.gamma * self.beta + augmented_data)
            self.fullA /= self.fullA.sum(1)[:,na]
            self.A = self.fullA * (1.-np.eye(self.state_dim))
            self.A /= self.A.sum(1)[:,na]
            assert not np.isnan(self.A).any()

class HDPHMMTransitions(object):
    def __init__(self,state_dim,alpha,gamma,beta=None,A=None):
        self.state_dim = state_dim
        self.alpha = alpha
        self.gamma = gamma
        if A is None or beta is None:
            self.resample()
        else:
            self.A = A
            self.beta = beta

    def resample_beta(self,m):
        self.beta = stats.gamma.rvs(self.alpha / self.state_dim + np.sum(m,axis=0))
        self.beta /= np.sum(self.beta)
        assert not np.isnan(self.beta).any()

    def resample_A(self,data):
        self.A = stats.gamma.rvs(self.gamma * self.beta + data)
        self.A /= np.sum(self.A,axis=1)[:,na]
        assert not np.isnan(self.A).any()

    def resample(self,states_list=[]):
        # TODO these checks can be removed at some point
        assert type(states_list) == type([])
        for states_norep in states_list:
            assert type(states_norep) == type(np.array([]))

        # count all transitions
        data = np.zeros((self.state_dim,self.state_dim),dtype=np.int32)
        for states in states_list:
            if len(states) >= 2:
                for idx in xrange(len(states)-1):
                    data[states[idx],states[idx+1]] += 1

        m = np.zeros((self.state_dim,self.state_dim),dtype=np.int32)
        if not (0 == data).all():
            for (rowidx, colidx), val in np.ndenumerate(data):
                m[rowidx,colidx] = (np.random.rand(val) < self.alpha * self.beta[colidx]\
                        /(np.arange(val) + self.alpha*self.beta[colidx])).sum()

        self.resample_beta(m)
        self.resample_A(data)


class StickyHDPHMMTransitions(HDPHMMTransitions):
    # TODO resample kappa
    def __init__(self,kappa,*args,**kwargs):
        self.kappa = kappa
        super(StickyHDPHMMTransitions,self).__init__(*args,**kwargs)

    def resample_A(self,data):
        aug_data = data + np.diag(self.kappa * np.ones(data.shape[0]))
        super(StickyHDPHMMTransitions,self).resample_A(aug_data)

