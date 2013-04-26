from __future__ import division
import numpy as np
import scipy.stats as stats
from numpy import newaxis as na
np.seterr(invalid='raise')
import operator

from ..basic.distributions import DirGamma
from ..util.general import rle

##########
#  misc  #
##########

# TODO scaling by self.state_dim in concresampling is the confusing result of
# having a DirGamma object and not a WLDPGamma object! make one
# TODO reuse Multinomial/Categorical code

class ConcentrationResampling(object):
    def __init__(self,state_dim,alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0):
        self.gamma_obj = DirGamma(state_dim,gamma_a_0,gamma_b_0)
        self.alpha_obj = DirGamma(state_dim,alpha_a_0,alpha_b_0)

    def resample(self):
        # multiply by state_dim because the trans objects divide by it (since
        # their parameters correspond to the DP parameters, and so they convert
        # into weak limit scaling)
        self.alpha_obj.resample(self.trans_counts,weighted_cols=self.beta)
        self.alpha = self.alpha_obj.concentration
        self.gamma_obj.resample(self.m)
        self.gamma = self.gamma_obj.concentration

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

    ### Gibbs sampling

    def resample(self,states_list=[]):
        trans_counts = self._count_transitions(states_list)
        m = self._get_m(trans_counts)

        self._resample_beta(m)
        self._resample_A(trans_counts)

    def _resample_beta(self,m):
        self.beta = np.random.dirichlet(self.gamma / self.state_dim + m.sum(0) + 1e-2)

    def _resample_A(self,trans_counts):
        self.A = stats.gamma.rvs(self.alpha * self.beta + trans_counts + 1e-2)
        self.A /= self.A.sum(1)[:,na]

    def _count_transitions(self,states_list):
        trans_counts = np.zeros((self.state_dim,self.state_dim),dtype=np.int32)
        for states in states_list:
            if len(states) >= 2:
                for idx in xrange(len(states)-1):
                    trans_counts[states[idx],states[idx+1]] += 1
        self.trans_counts = trans_counts
        return trans_counts

    # TODO push this into eigen
    def _get_m(self,trans_counts):
        m = np.zeros((self.state_dim,self.state_dim),dtype=np.int32)
        if not (0 == trans_counts).all():
            for (rowidx, colidx), val in np.ndenumerate(trans_counts):
                if val > 0:
                    m[rowidx,colidx] = (np.random.rand(val) < self.alpha * self.beta[colidx] \
                            /(np.arange(val) + self.alpha*self.beta[colidx])).sum()
        self.m = m
        return m

    ### max likelihood
    # TODO these methods shouldn't really be in this class... maybe put them in
    # a base class

    def max_likelihood(self,stateseqs,expectations_list=None):
        if expectations_list is not None:
            trans_counts = self._count_weighted_transitions(expectations_list,self.A)
        else:
            trans_counts = self._count_transitions(stateseqs)

        errs = np.seterr(invalid='ignore',divide='ignore')
        self.A = trans_counts / trans_counts.sum(1)[:,na]
        np.seterr(**errs)

        self.A[np.isnan(self.A)] = 0.

    # NOTE: only needs aBl because the message computation saves betal and not
    # betastarl TODO compute betastarl like a civilized gentleman
    @staticmethod
    def _count_weighted_transitions(expectations_list,A):
        trans_softcounts = np.zeros_like(A)
        Al = np.log(A)

        for alphal, betal, aBl in expectations_list:
            log_joints = alphal[:-1,:,na] + (betal[1:,na,:] + aBl[1:,na,:]) + Al[na,...]
            log_joints -= np.logaddexp.reduce(alphal[0] + betal[0]) # p(y)
            joints = np.exp(log_joints,out=log_joints)

            trans_softcounts += joints.sum(0)

        return trans_softcounts


class HDPHMMTransitionsConcResampling(HDPHMMTransitions,ConcentrationResampling):
    def __init__(self,state_dim,
            alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0,
            **kwargs):

        ConcentrationResampling.__init__(self,state_dim,alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0)
        super(HDPHMMTransitionsConcResampling,self).__init__(state_dim,
                alpha=self.alpha_obj.concentration*state_dim,
                gamma=self.gamma_obj.concentration*state_dim,
                **kwargs)

    def resample(self,*args,**kwargs):
        super(HDPHMMTransitionsConcResampling,self).resample(*args,**kwargs)
        ConcentrationResampling.resample(self)


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


class UniformTransitionsFixedSelfTrans(HDPHMMTransitions):
    '''
    All states have the same self-transition probability, i.e. np.diag(A) is a
    constant LMBDA.  The transition-out probabilities are all the same, set according
    to a fixed state weighting PI. Note that transitions out are sampled
    proportional to PI but without the possibility of self-transition.
    '''
    def __init__(self,lmbda,pi):
        assert 0 < lmbda < 1

        self.lmbda = lmbda
        self.pi = pi
        self.state_dim = pi.K
        self._set_A()

    ### Gibbs sampling

    def resample(self,states_list=[],niter=100,trans_counts=None):
        # trans_counts arg is for convenient testing
        if trans_counts is None:
            trans_counts = self._count_transitions(states_list)

        for itr in range(niter):
            aug_trans_counts = self._augment_transitions(trans_counts)
            self.pi.resample(aug_trans_counts.sum(0))

        self._set_A()

    def _augment_transitions(self,trans):
        trans = trans.copy()
        if trans.sum() > 0:
            trans.flat[::trans.shape[0]+1] = 0
            for i, tot in enumerate(trans.sum(1)):
                trans[i,i] = np.random.geometric(1.-self.pi.weights[i],size=tot).sum() - tot
        return trans

    def _set_A(self):
        self.A = np.tile(self.pi.weights,(self.state_dim,1))
        self.A.flat[::self.state_dim+1] = 0
        self.A /= self.A.sum(1)[:,na]
        self.A *= (1.-self.lmbda)
        self.A.flat[::self.state_dim+1] = self.lmbda

    @classmethod
    def test_sampling(cls,N=50,K=10,alpha_0=4.,lmbda=0.95):
        from ..basic.distributions import Categorical
        from matplotlib import pyplot as plt

        true_pi = np.random.dirichlet(np.repeat(alpha_0/K,K))
        counts = np.array([np.random.multinomial(N,true_pi) for i in range(K)]) # diagional ignored

        pi = Categorical(alpha_0=alpha_0,K=K)
        trans = cls(lmbda,pi)

        plt.figure()
        plt.plot(true_pi,'r')
        cmap = plt.cm.get_cmap('Blues')
        for i in range(5):
            trans.resample(trans_counts=counts)
            plt.plot(trans.pi.weights,'--',label=str(i),color=cmap((i+2)/(5+2)))
        plt.legend()
        plt.show()

    ### max likelihood

    def max_likelihood(self,stateseqs,expectations_list=None):
        if expectations_list is not None:
            trans_counts = self._count_weighted_transitions(expectations_list,self.A)
        else:
            trans_counts = self._count_transitions(stateseqs)

        trans_counts = self._E_augment_transitions(trans_counts)
        self.pi.max_likelihood(trans_counts.sum(0))
        self._set_A()

    def _E_augment_transitions(self,trans_softcounts):
        trans_softcounts.flat[::self.state_dim+1] = trans_softcounts.sum(1) * 1./(1.-self.lmbda)
        return trans_softcounts


class UniformTransitions(UniformTransitionsFixedSelfTrans):
    '''
    Like UniformTransitionsFixedSelfTrans except also samples over the
    self-transition probability LMBDA.
    '''
    def __init__(self,lmbda_a_0,lmbda_b_0,pi,lmbda=None):
        self.a_0 = lmbda_a_0
        self.b_0 = lmbda_b_0
        self.pi = pi
        self.state_dim = pi.K

        if lmbda is not None:
            self.lmbda = lmbda
            self._set_A()
        else:
            self.resample()

    def resample(self,states_list=[],niter=100,trans_counts=None):
        if trans_counts is None:
            trans_counts = self._count_transitions(states_list)

        for itr in range(niter):
            # resample lmbda
            self_trans = trans_counts.diagonal().sum()
            total_out = trans_counts.sum() - self_trans
            self.lmbda = np.random.beta(self.a_0 + self_trans, self.b_0 + total_out)

            # resample everything else as usual
            super(UniformTransitions,self).resample(states_list,trans_counts=trans_counts,niter=1)

    def max_likelihood(self,stateseqs,expectations_list=None):
        raise NotImplementedError, "max_likelihood doesn't make sense on this class"

    @classmethod
    def test_sampling(cls,N=50,K=10,alpha_0=4.,lmbda_a_0=20.,lmbda_b_0=1.,true_lmbda=0.95):
        from ..basic.distributions import Categorical
        from matplotlib import pyplot as plt

        true_pi = np.random.dirichlet(np.repeat(alpha_0/K,K))
        counts = np.array([np.random.multinomial(N,true_pi) for i in range(K)])
        counts.flat[::K+1] = 0
        for i,tot in enumerate(counts.sum(1)):
            counts[i,i] = np.random.geometric(1-true_lmbda,size=tot).sum()

        pi = Categorical(alpha_0=alpha_0,K=K)
        trans = cls(lmbda_a_0=lmbda_a_0,lmbda_b_0=lmbda_b_0,pi=pi)

        plt.figure()
        plt.plot(true_pi,'r')
        cmap = plt.cm.get_cmap('Blues')
        print 'True lmbda: %0.5f' % true_lmbda
        for i in range(5):
            trans.resample(trans_counts=counts)
            print 'Sampled lmbda: %0.5f' % trans.lmbda
            plt.plot(trans.pi.weights,'--',label=str(i),color=cmap((i+2)/(5+2)))
        plt.legend()
        plt.show()

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
        states_noreps = map(operator.itemgetter(0),map(rle, stateseqs))

        augmented_data = self._augment_data(self._count_transitions(states_noreps))
        m = self._get_m(augmented_data)

        self._resample_beta(m)
        self._resample_A(augmented_data)

    def _augment_data(self,trans_counts):
        trans_counts = trans_counts.copy()
        if trans_counts.sum() > 0:
            froms = trans_counts.sum(1)
            self_transitions = [np.random.geometric(1-pi_ii,size=n).sum() if n > 0 else 0
                    for pi_ii,n in zip(self.fullA.diagonal(),froms)]
            trans_counts += np.diag(self_transitions)
        self.trans_counts = trans_counts
        return trans_counts

    def _resample_A(self,augmented_data):
        super(HDPHSMMTransitions,self)._resample_A(augmented_data)
        self.fullA = self.A.copy()
        self.A.flat[::self.A.shape[0]+1] = 0
        self.A /= self.A.sum(1)[:,na]


class HDPHSMMTransitionsConcResampling(HDPHSMMTransitions,ConcentrationResampling):
    def __init__(self,state_dim,
            alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0,
            **kwargs):

        ConcentrationResampling.__init__(self,state_dim,alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0)
        super(HDPHSMMTransitionsConcResampling,self).__init__(state_dim,
                alpha=self.alpha_obj.concentration*state_dim,
                gamma=self.gamma_obj.concentration*state_dim,
                **kwargs)

    def resample(self,*args,**kwargs):
        super(HDPHSMMTransitionsConcResampling,self).resample(*args,**kwargs)
        ConcentrationResampling.resample(self)


############################
#  Sticky HDP-HMM classes  #
############################

class StickyHDPHMMTransitions(HDPHMMTransitions):
    def __init__(self,kappa,*args,**kwargs):
        self.kappa = kappa
        super(StickyHDPHMMTransitions,self).__init__(*args,**kwargs)

    def _resample_beta(self,m):
        # the counts in newm are the ones we keep for the transition matrix;
        # (m - newm) are the counts for kappa, i.e. the 'override' counts
        newm = m.copy()
        if m.sum() > 0:
            # np.random.binomial fails when n=0, so pull out nonzero indices
            indices = np.nonzero(newm.flat[::m.shape[0]+1])
            newm.flat[::m.shape[0]+1][indices] = np.array(np.random.binomial(
                    m.flat[::m.shape[0]+1][indices],
                    self.beta[indices]*self.alpha/(self.beta[indices]*self.alpha + self.kappa)),
                    dtype=np.int32)
        self.newm = newm
        return super(StickyHDPHMMTransitions,self)._resample_beta(newm)

    def _resample_A(self,data):
        # equivalent to adding kappa to the appropriate part of beta for each row
        aug_data = data + np.diag(self.kappa * np.ones(data.shape[0]))
        super(StickyHDPHMMTransitions,self)._resample_A(aug_data)


class StickyHDPHMMTransitionsConcResampling(StickyHDPHMMTransitions,ConcentrationResampling):
    # NOTE: this parameterizes (alpha plus kappa) and rho, as in EB Fox's thesis
    # the class member named 'alpha_obj' should really be 'alpha_plus_kappa_obj'
    # the member 'alpha' is still just alpha
    def __init__(self,state_dim,
            rho_a_0,rho_b_0,
            alphakappa_a_0,alphakappa_b_0,gamma_a_0,gamma_b_0,
            **kwargs):

        self.rho_a_0, self.rho_b_0 = rho_a_0, rho_b_0
        self.rho = np.random.beta(rho_a_0,rho_b_0)
        ConcentrationResampling.__init__(self,state_dim,alphakappa_a_0,alphakappa_b_0,gamma_a_0,gamma_b_0)
        super(StickyHDPHMMTransitionsConcResampling,self).__init__(state_dim=state_dim,
                kappa=self.rho * self.alpha_obj.concentration*state_dim,
                alpha=(1-self.rho) * self.alpha_obj.concentration*state_dim,
                gamma=self.gamma_obj.concentration*state_dim,
                **kwargs)

    def resample(self,*args,**kwargs):
        super(StickyHDPHMMTransitionsConcResampling,self).resample(*args,**kwargs)
        ConcentrationResampling.resample(self)
        self.rho = np.random.beta(self.rho_a_0 + (self.m-self.newm).sum(), self.rho_b_0 + self.newm.sum())
        self.kappa = self.rho * self.alpha_obj.concentration*self.state_dim


class StickyLTRHDPHMMTransitions(LTRHDPHMMTransitions,StickyHDPHMMTransitions):
    def _resample_A(self,data):
        aug_data = data + np.diag(self.kappa * np.ones(data.shape[0]))
        super(StickyLTRHDPHMMTransitions,self)._resample_A(aug_data)

