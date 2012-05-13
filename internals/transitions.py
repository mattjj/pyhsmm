from __future__ import division
import numpy as np
import scipy.stats as stats
from numpy.random import random
from numpy import newaxis as na

from pyhsmm.util.general import rle
from util.stats import sample_discrete

# TODO add concentration parameter resampling to the classes
# TODO add kappa resampling for sticky hdphmms

class concentration_parameter(object):
    '''
    implements Gamma(a,b) prior over symmetric Dirichlet / DP concentration
    parameter given multinomial data (integrating out the weights)
    '''
    def __init__(self,a,b,concentration=None):
        self.a = a
        self.b = b

        if concentration is not None:
            self.concentration = concentration
        else:
            self.resample()

    def resample(self,sample_numbers=None,total_num_distinct=None,niter=20):
        # num_samples can be a vector, one element for each multinomial
        # observation set from a different pi sample, and each element is
        # the number of draws in that multinomial set
        # see appendix A of http://www.cs.berkeley.edu/~jordan/papers/hdp.pdf
        # and appendix C of Emily Fox's PhD thesis
        # the notation of w's and s's follows from the HDP paper
        sample_numbers = np.array(sample_numbers)
        a,b = self.a, self.b
        if sample_numbers is None or total_num_distinct is None:
            self.concentration = stats.gamma.rvs(a,scale=1./b)
        else:
            # sample_numbers += 1e-10 # convenient in case any element is zero
            for itr in range(niter):
                wvec = stats.beta.rvs(self.concentration+1,sample_numbers)
                svec = np.array(stats.bernoulli.rvs(sample_numbers/(sample_numbers+self.concentration)))
                self.concentration = stats.gamma.rvs(a+total_num_distinct-svec.sum(),scale=1./(b-np.log(wvec).sum()))
                # note scipy.stats.gamma uses a scale parameter that is the
                # inverse of the scale parameter used in the reference papers.
                # that gets me EVERY TIME

    @classmethod
    def test(cls):
        truth = cls(1.,1.)

        alldata = []
        sizes = [1000]
        for size in sizes:
            weights = stats.gamma.rvs(truth.concentration/400.,size=400)
            weights /= weights.sum()
            alldata.append(sample_discrete(weights,size=size))

        infer = cls(1.,1.)
        print truth.concentration
        print ''
        blah = []
        for itr in range(500):
            infer.resample(sample_numbers=np.array(sizes),total_num_distinct=len(set(np.concatenate(alldata))))
            blah.append(infer.concentration)

        print np.mean(blah)

class hsmm_transitions(object):
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

    def __init__(self,state_dim,alpha,gamma,beta=None,A=None,fullA=None,**kwargs):
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

        states_noreps = map(lambda x: rle(x)[0], stateseqs)

        if not any(len(states_norep) >= 2 for states_norep in states_noreps):
            # if there is no data we just sample from the prior
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
            assert not np.isnan(self.A).any()

class hdphmm_transitions(object):
    # TODO alpha/gamma remains switched wrt math notation
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
        data = np.zeros((self.state_dim,self.state_dim))
        for states in states_list:
            if len(states) >= 2:
                for idx in xrange(len(states)-1):
                    data[states[idx],states[idx+1]] += 1

        m = np.zeros((self.state_dim,self.state_dim))
        if not (0 == data).all():
            # sample m's (auxiliary variables which make beta's posterior
            # conjugate and indep. of data)
            for rowidx in xrange(self.state_dim):
                for colidx in xrange(self.state_dim):
                    for n in xrange(int(data[rowidx,colidx])):
                        m[rowidx,colidx] += random() < self.alpha * self.beta[colidx] /\
                                                        (n + self.alpha * self.beta[colidx])

        self.resample_beta(m)
        self.resample_A(data)


class sticky_hdphmm_transitions(hdphmm_transitions):
    '''
    doesn't resample kappa! should add that like ebfox's version...
    '''
    def __init__(self,kappa,*args,**kwargs):
        self.kappa = kappa
        super(sticky_hdphmm_transitions,self).__init__(*args,**kwargs)

    def resample_A(self,data):
        aug_data = data + np.diag(self.kappa * np.ones(data.shape[0]))
        super(sticky_hdphmm_transitions,self).resample_A(aug_data)


# NOTE none of the below should be able to learn the number of states very well,
# since they don't have the hierarchical construction to regularize the total
# number of states

class hmm_transitions(object):
    # self_trans is like a simple sticky bias
    def __init__(self,state_dim,gamma,A=None,**kwargs):
        self.state_dim = state_dim
        self.gamma = gamma
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
        # data += self.self_trans * np.eye(len(data))
        if len(statess) > 0:
            for states in statess:
                if len(states) >= 2:
                    for idx in xrange(len(states)-1):
                        data[states[idx],states[idx+1]] += 1

        self.resample_A(data)

class ltr_hmm_transitions(hmm_transitions):
    '''upper triangle only'''
    def resample_A(self,data):
        self.A = stats.gamma.rvs(self.gamma/self.state_dim + data)
        self.A = np.triu(self.A)
        self.A /= np.sum(self.A,axis=1)[:,na]
        assert not np.isnan(self.A).any()

# TODO make this work as an hdp-hmm
class sticky_ltr_hmm_transitions(ltr_hmm_transitions):
    def __init__(self,kappa,*args,**kwargs):
        self.kappa = kappa
        super(sticky_ltr_hmm_transitions,self).__init__(*args,**kwargs)

    def resample_A(self,data):
        aug_data = data + np.diag(self.kappa * np.ones(data.shape[0]))
        super(sticky_ltr_hmm_transitions,self).resample_A(aug_data)
