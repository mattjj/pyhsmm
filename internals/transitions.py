from __future__ import division
import numpy as np
import scipy.stats as stats
from numpy import newaxis as na
import operator

from ..basic.distributions import DirGamma
from ..util.general import rle

# TODO could reuse parts of basic.distributions.Multinomial

######################
#  HDP-HMM classes  #
######################

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

    def resample(self,states_list=[]):
        if len(states_list) > 0:
            trans_counts = self._count_transitions(states_list)
            m = self._get_m(trans_counts)
        else:
            trans_counts = np.zeros((self.state_dim,self.state_dim),dtype=np.int32)
            m = np.zeros((self.state_dim,self.state_dim),dtype=np.int32)

        self._resample_beta(m)
        self._resample_A(trans_counts)

    def _resample_beta(self,m):
        self.beta = np.random.dirichlet(self.gamma / self.state_dim + m.sum(0))

    def _resample_A(self,trans_counts):
        self.A = stats.gamma.rvs(self.alpha * self.beta + trans_counts)
        self.A /= self.A.sum(1)[:,na]

    def _count_transitions(self,states_list):
        trans_counts = np.zeros((self.state_dim,self.state_dim),dtype=np.int32)
        for states in states_list:
            if len(states) >= 2:
                for idx in xrange(len(states)-1):
                    trans_counts[states[idx],states[idx+1]] += 1
        return trans_counts

    def _get_m(self,trans_counts):
        m = np.zeros((self.state_dim,self.state_dim),dtype=np.int32)
        if not (0 == trans_counts).all():
            for (rowidx, colidx), val in np.ndenumerate(trans_counts):
                m[rowidx,colidx] = (np.random.rand(val) < self.alpha * self.beta[colidx]\
                        /(np.arange(val) + self.alpha*self.beta[colidx])).sum()
        return m


class HDPHMMTransitionsConcResampling(HDPHMMTransitions):
    pass # TODO


class LTRHDPHMMTransitions(HDPHMMTransitions):
    def _count_transitions(self,states_list):
        trans_counts = super(LTRHDPHMMTransitions,self)._count_transitions(states_list)
        assert (0==np.tril(trans_counts,-1)).all()
        totalfrom = trans_counts.sum(1)
        totalweight = np.triu(self.fullA).sum(1)
        for i in range(trans_counts.shape[0]):
            tot = np.random.geometric(totalweight[i],size=totalfrom[i]).sum()
            trans_counts[i,:i] = np.random.multinomial(tot,self.fullA[i,:i]/self.fullA[i,:i].sum())
        return trans_counts

    def _resample_A(self,trans_counts):
        super(LTRHDPHMMTransitions,self)._resample_A(trans_counts)
        self.fullA = self.A
        self.A = np.triu(self.A) / np.triu(self.A).sum(1)[:,na]

######################
#  HDP-HSMM classes  #
######################

class HDPHSMMTransitions(HDPHMMTransitions):
    '''
    HDPHSMM transition distribution class.
    Uses a weak-limit HDP prior. Zeroed diagonal to forbid self-transitions.

    Hyperparameters follow the notation in Fox et al.
        gamma: concentration paramter for beta
        alpha: total mass concentration parameter for each row of trans matrix

    Parameters are the shared transition vector beta, the full transition matrix,
    and the matrix with the diagonal zeroed.
        beta, A, fullA
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
        if len(stateseqs) > 0:
            states_noreps = map(operator.itemgetter(0),map(rle, stateseqs))
            trans_counts = self._count_transitions(states_noreps)
            augmented_data = self.augmented_data = self._augment_data(trans_counts)
            m = self.m = self._get_m(augmented_data)
        else:
            augmented_data = np.zeros((self.state_dim,self.state_dim),dtype=np.int32)
            m = np.zeros((self.state_dim,self.state_dim),dtype=np.int32)

        self._resample_beta(m)
        self._resample_A(augmented_data)

    def _augment_data(self,trans_counts):
        froms = trans_counts.sum(1)
        self_transitions = [np.random.geometric(1-pi_ii,size=n).sum() if n > 0 else 0
                for pi_ii,n in zip(self.fullA.diagonal(),froms)]
        return trans_counts + np.diag(self_transitions)

    def _resample_A(self,augmented_data):
        super(HDPHSMMTransitions,self)._resample_A(augmented_data)
        self.fullA = self.A.copy()
        self.A.flat[::self.A.shape[0]+1] = 0
        self.A /= self.A.sum(1)[:,na]


class HDPHSMMTransitionsConcResampling(HDPHSMMTransitions):
    def __init__(self,state_dim,
            alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0,
            **kwargs):

        self.gamma_obj = DirGamma(state_dim,gamma_a_0,gamma_b_0,state_dim)
        self.alpha_obj = DirGamma(state_dim,alpha_a_0,alpha_b_0,state_dim)

        super(HDPHSMMTransitionsConcResampling,self).__init__(state_dim,
                alpha=self.alpha_obj.concentration*state_dim,
                gamma=self.gamma_obj.concentration*state_dim,
                **kwargs)

    def resample(self,*args,**kwargs):
        self.augmented_data = []
        self.m = []
        super(HDPHSMMTransitionsConcResampling,self).resample(*args,**kwargs)
        self.alpha_obj.resample(self.augmented_data,weighted_cols=self.beta,niter=10)
        self.alpha = self.alpha_obj.concentration*self.state_dim
        self.gamma_obj.resample(self.m,niter=10)
        self.gamma = self.gamma_obj.concentration*self.state_dim


############################
#  Sticky HDP-HMM classes  #
############################

class StickyHDPHMMTransitions(HDPHMMTransitions):
    def __init__(self,kappa,*args,**kwargs):
        self.kappa = kappa
        super(StickyHDPHMMTransitions,self).__init__(*args,**kwargs)

    def resample_beta(self,m):
        newm = m.copy()
        newm.flat[::m.shape[0]+1] = np.random.binomial(
                m.flat[::m.shape[0]+1],
                self.beta/(self.beta + self.kappa))
        return super(StickyHDPHMMTransitions,self).resample_beta(newm)

    def resample_A(self,data):
        aug_data = data + np.diag(self.kappa * np.ones(data.shape[0]))
        super(StickyHDPHMMTransitions,self).resample_A(aug_data)


class StickyHDPHMMTransitionsConcResampling(StickyHDPHMMTransitions):
    # resample kappa too!
    pass # TODO
