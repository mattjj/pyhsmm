import numpy as np
import scipy.stats as stats
from numpy.random import random
from numpy import newaxis as na

# TODO make resample methods take lists of state sequences

class transitions(object):#{{{
    '''
    HSMM transition distribution class.
    Uses a weak-limit HDP prior. Zeroed diagonal to forbid self-transitions.

    Hyperparameters follow the notation in Fox et al., except the definitions of
    alpha and gamma are reversed.
    alpha, gamma

    Parameters are the shared transition vector beta, the full transition matrix,
    and the matrix with the diagonal zeroed.
    beta, fullA, A
    '''

    def __init__(self,state_dim,alpha=1000.,gamma=8.,beta=None,A=None,fullA=None,**kwargs):
        self.alpha = alpha
        self.gamma = gamma
        self.state_dim = state_dim
        if A is None or fullA is None or beta is None:
            self.resample()
        else:
            self.A = A
            self.beta = beta
            self.fullA = fullA

    def resample(self,states_noreps=[]):
        if type(states_noreps) != type([]):
            states_noreps = [states_noreps]
        for states_norep in states_noreps:
            assert type(states_norep) == type(np.array([]))
        if not any(len(states_norep) >= 2 for states_norep in states_noreps):
            # if there is no data we just sample forward
            self.beta = stats.gamma.rvs(self.alpha / self.state_dim, size=self.state_dim)
            self.beta /= np.sum(self.beta)

            self.fullA = stats.gamma.rvs(self.beta * self.gamma * np.ones((self.state_dim,1)))
            self.A = (1.-np.eye(self.state_dim)) * self.fullA
            self.fullA /= np.sum(self.fullA,axis=1)[:,na]
            self.A /= np.sum(self.A,axis=1)[:,na]

            assert not np.isnan(self.beta).any()
            assert not np.isnan(self.fullA).any()
            assert (self.A.diagonal() == 0).all()
        else:
            # make 2d array of transition counts
            data = np.zeros((self.state_dim,self.state_dim))
            for states_norep in states_noreps:
                for idx in xrange(len(states_norep)-1):
                    data[states_norep[idx],states_norep[idx+1]] += 1
            # we resample the children (A) then the mother (beta)
            # first, we complete the data using the current parameters
            # every time we transferred from a state, we had geometrically many 
            # self-transitions thrown away that we want to sample
            assert (data.diagonal() == 0).all()
            froms = np.sum(data,axis=1)
            self_transitions = np.array([np.sum(stats.geom.rvs(1.-self.fullA.diagonal()[idx],size=from_num)) if from_num > 0 else 0 for idx, from_num in enumerate(froms)])
            self_transitions[froms == 0] = 0 # really emphasized here!
            assert (self_transitions < 1e7).all(), 'maybe alpha is too low... code is not happy about that at the moment'
            augmented_data = data + np.diag(self_transitions)
            # then, compute m's and stuff
            m = np.zeros((self.state_dim,self.state_dim))
            for rowidx in xrange(self.state_dim):
                for colidx in xrange(self.state_dim):
                    n = 0.
                    for i in xrange(int(augmented_data[rowidx,colidx])):
                        m[rowidx,colidx] += random() < self.alpha * self.beta[colidx] / (n + self.alpha * self.beta[colidx])
                        n += 1.
            self.m = m # save it for possible use in any child classes

            # resample mother (beta)
            self.beta = stats.gamma.rvs(self.alpha / self.state_dim  + np.sum(m,axis=0))
            self.beta /= np.sum(self.beta)
            assert not np.isnan(self.beta).any()
            # resample children (fullA and A)
            self.fullA = stats.gamma.rvs(self.gamma * self.beta + augmented_data)
            self.fullA /= np.sum(self.fullA,axis=1)[:,na]
            self.A = self.fullA * (1.-np.eye(self.state_dim))
            self.A /= np.sum(self.A,axis=1)[:,na]
            assert not np.isnan(self.A).any()#}}}

class hdphmm_transitions(object):#{{{
    # TODO alpha/gamma remains switched wrt math notation
    def __init__(self,state_dim,alpha=4.,gamma=8.,beta=None,A=None):
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

    def resample(self,states=np.array([])):
        m = np.zeros((self.state_dim,self.state_dim))
        data = np.zeros((self.state_dim,self.state_dim))
        if len(states) >= 2:
            # count transitions
            for idx in xrange(len(states)-1):
                data[states[idx],states[idx+1]] += 1

            # sample m's (auxiliary variables which make beta's posterior
            # conjugate and indep. of data)
            for rowidx in xrange(self.state_dim):
                for colidx in xrange(self.state_dim):
                    for n in xrange(int(data[rowidx,colidx])):
                        m[rowidx,colidx] += random() < self.alpha * self.beta[colidx] / (n + self.alpha * self.beta[colidx])

        self.resample_beta(m)
        self.resample_A(data)#}}}

class sticky_hdphmm_transitions(hdphmm_transitions):#{{{
    def __init__(self,kappa,*args,**kwargs):
        self.kappa = kappa
        hdphmm_transitions.__init__(self,*args,**kwargs)

    def resample_A(self,data):
        aug_data = data + np.diag(self.kappa * np.ones(data.shape[0]))
        hdphmm_transitions.resample_A(self,aug_data)#}}}

class hmm_transitions(object): #{{{
    def __init__(self,state_dim,gamma=8.,self_trans=0.,A=None,**kwargs):
        self.state_dim = state_dim
        self.gamma = gamma
        self.self_trans = self_trans
        if A is None:
            self.resample()
        else:
            self.A = A

    def resample_A(self,data):
        self.A = stats.gamma.rvs(self.gamma/self.state_dim + data)
        self.A /= np.sum(self.A,axis=1)[:,na]
        assert not np.isnan(self.A).any()

    def resample(self,statess=[]):
        data = np.zeros((self.state_dim,self.state_dim))
        data += self.self_trans * np.eye(len(data))
        if len(statess) > 0:
            for states in statess:
                if len(states) >= 2:
                    for idx in xrange(len(states)-1):
                        data[states[idx],states[idx+1]] += 1

        self.resample_A(data)
#}}}

class ltr_hmm_transitions(hmm_transitions): # {{{
    '''upper triangle only'''
    def resample_A(self,data):
        self.A = stats.gamma.rvs(self.gamma/self.state_dim + data)
        self.A = np.triu(self.A)
        self.A /= np.sum(self.A,axis=1)[:,na]
        assert not np.isnan(self.A).any()
#}}}
