from __future__ import division
import numpy as np
import scipy.stats as stats
import scipy.weave
from numpy import newaxis as na
np.seterr(invalid='raise')
import operator
import copy

from ..basic.distributions import GammaCompoundDirichlet, Multinomial, \
        MultinomialAndConcentration
from ..util.general import rle, count_transitions

########################
#  HMM / HSMM classes  #
########################

class HMMTransitions(object):
    # this class just wraps a set of Multinomial objects, one for each row
    # if only alpha is specified, the priors are symmetric and use weak limit
    # scaling (but no hierarchical prior!)
    def __init__(self,num_states=None,alpha=None,alphav=None,trans_matrix=None):
        self.N = num_states
        self.alpha = alpha
        self.alphav = alphav
        self.trans_matrix = trans_matrix

        if trans_matrix is None and (None not in (self.alpha,self.N) or self.alphav is not None):
            self._row_distns = [Multinomial(alpha_0=self.alpha,K=self.N,alphav_0=self.alphav)
                    for n in xrange(self.N)] # sample from prior

    def _get_trans_matrix(self):
        return np.array([d.weights for d in self._row_distns])

    def _set_trans_matrix(self,trans_matrix):
        if trans_matrix is not None:
            N = self.N = trans_matrix.shape[0]
            self._row_distns = \
                    [Multinomial(alpha_0=self.alpha,K=N,alphav_0=self.alphav,weights=row)
                            for row in trans_matrix]

    trans_matrix = property(_get_trans_matrix,_set_trans_matrix)

    def _count_transitions(self,stateseqs):
        if isinstance(stateseqs,np.ndarray) or isinstance(stateseqs[0],int) \
                or isinstance(stateseqs[0],float):
            return count_transitions(stateseqs,minlength=self.N)
        else:
            return sum(count_transitions(stateseq,minlength=self.N)
                    for stateseq in stateseqs)

    ### Gibbs sampling

    def resample(self,stateseqs=[],trans_counts=None):
        trans_counts = self._count_transitions(stateseqs) if trans_counts is None \
                else trans_counts
        for d, counts in zip(self._row_distns,trans_counts):
            d.resample(counts)
        return self

    ### max likelihood

    def max_likelihood(self,stateseqs=None,expected_transcounts=None):
        trans_counts = sum(expected_transcounts) or self._count_transitions(stateseqs)

        # could just call max_likelihood on each trans row, but this way it
        # handles a few lazy-initialization cases (e.g. if _row_distns aren't
        # initialized)
        errs = np.seterr(invalid='ignore',divide='ignore')
        self.trans_matrix = np.nan_to_num(trans_counts / trans_counts.sum(1)[:,na])
        np.seterr(**errs)

        return self

    ### mean field

    @property
    def exp_expected_log_trans_matrix(self):
        # this is Atilde in Matthew Beal's PhD Ch. 3
        return np.exp(np.array([d.expected_log_likelihood()
            for d in self._row_distns])) # NOTE: see Multinomial.expected_log_likelihood

    def meanfieldupdate(self,expected_transcounts):
        trans_softcounts = sum(expected_transcounts)
        for d, counts in zip(self._row_distns, trans_softcounts):
            d.meanfieldupdate(None,counts) # None is placeholder, see Multinomial class def
        return self

    def get_vlb(self):
        return sum(d.get_vlb() for d in self._row_distns)


class HSMMTransitions(HMMTransitions):
    def _get_trans_matrix(self):
        out = self.full_trans_matrix
        out.flat[::out.shape[0]+1] = 0
        out /= out.sum(1)[:,na]
        return out

    trans_matrix = property(_get_trans_matrix,HMMTransitions._set_trans_matrix)

    def max_likelihood(self,*args,**kwargs):
        raise NotImplementedError

    @property
    def exp_expected_log_trans_matrix(self):
        raise NotImplementedError

    def meanfieldupdate(self,*args,**kwargs):
        raise NotImplementedError

    def get_vlb(self):
        raise NotImplementedError

    # NOTE: only need to augment the data to make concentration resampling work
    # easily (here or in HSMMTransitionsConc); otherwise, only need data
    # augmentation for hierarchical prior case. I chose to put the augmentation
    # in this class because it's also nice for _row_distns to reflect valid
    # sampled posterior beliefs (not just the normalized off-diagonal parts)

    @property
    def full_trans_matrix(self):
        return super(HSMMTransitions,self).trans_matrix

    def _count_transitions(self,stateseqs):
        stateseq_noreps = [rle(stateseq)[0] for stateseq in stateseqs]
        trans_counts = super(HSMMTransitions,self)._count_transitions(stateseq_noreps)

        if trans_counts.sum() > 0:
            froms = trans_counts.sum(1)
            self_trans = [np.random.geometric(1-A_ii,size=n).sum() if n > 0 else 0
                    for A_ii, n in zip(self.full_trans_matrix.diagonal(),froms)]
            trans_counts += np.diag(self_trans)

        return trans_counts


class _ConcentrationResamplingMixin(object):
    # NOTE: because all rows share the same concentration parameter, we can't
    # use e.g. CategoricalAndConcentration
    def __init__(self,num_states,alpha_a_0,alpha_b_0,**kwargs):
        self.alpha_obj = GammaCompoundDirichlet(num_states,alpha_a_0,alpha_b_0)
        super(_ConcentrationResamplingMixin,self).__init__(
                num_states=num_states,alpha=self.alpha,**kwargs)

    def _get_alpha(self):
        return self.alpha_obj.concentration

    def _set_alpha(self,alpha):
        if alpha is not None:
            self.alpha_obj.concentration = alpha # a no-op when called internally
            for d in self._row_distns:
                d.alpha_0 = alpha

    alpha = property(_get_alpha,_set_alpha)

    def resample(self,stateseqs=[],trans_counts=None):
        trans_counts = trans_counts or self._count_transitions(stateseqs)
        self.alpha_obj.resample(trans_counts)
        self.alpha = self.alpha_obj.concentration
        return super(_ConcentrationResamplingMixin,self).resample(
                stateseqs=stateseqs,trans_counts=trans_counts)

    def meanfieldupdate(self,*args,**kwargs):
        raise NotImplementedError # TODO

class HMMTransitionsConc(_ConcentrationResamplingMixin,HMMTransitions):
    pass

class HSMMTransitionsConc(_ConcentrationResamplingMixin,HSMMTransitions):
    pass

############################
#  Weak-Limit HDP classes  #
############################

class WeakLimitHDPHMMTransitions(HMMTransitions):
    # like HMMTransitions but with a hierarchical prior
    # max_likelihood ignores prior and just passes through to parent for
    # convenience
    def __init__(self,gamma,alpha,num_states=None,beta=None,trans_matrix=None):
        if num_states is None:
            assert beta is not None or trans_matrix is not None
            self.N = len(beta) if beta is not None else trans_matrix.shape[0]
        else:
            self.N = num_states

        self.beta_obj = Multinomial(alpha_0=gamma,K=self.N,weights=beta)

        super(WeakLimitHDPHMMTransitions,self).__init__(
                num_states=self.N,alpha=alpha,
                alphav=alpha*self.beta,trans_matrix=trans_matrix)

    @property
    def beta(self):
        return self.beta_obj.weights

    @property
    def gamma(self):
        return self.beta_obj.alpha_0

    ### Gibbs sampling

    def resample(self,stateseqs=[],trans_counts=None,ms=None):
        trans_counts = self._count_transitions(stateseqs) if trans_counts is None \
                else trans_counts
        ms = self._get_m(trans_counts) if ms is None else ms
        self.beta_obj.resample(ms)
        for d in self._row_distns:
            d.alphav_0 = self.alpha * self.beta
        return super(WeakLimitHDPHMMTransitions,self).resample(
                stateseqs=stateseqs,trans_counts=trans_counts)

    def _get_m(self,trans_counts):
        # splits tables by running a CRP on each
        # TODO factor this out into a util function, get rid of weave
        m = np.zeros_like(trans_counts)
        N = m.shape[0]
        if not (0 == trans_counts).all():
            alpha, beta = self.alpha, self.beta
            scipy.weave.inline(
            '''
            for (int i=0; i<N; i++) {
                for (int j=0; j<N; j++) {
                    int tot = 0;
                    for (int k=0; k<trans_counts[N*i+j]; k++) {
                        tot += ((double)rand())/RAND_MAX < (alpha * beta[j])/(k+alpha*beta[j]);
                    }
                    m[N*i+j] = tot;
                }
            }
            ''',
            ['trans_counts','N','m','alpha','beta'],
            extra_compile_args=['-O3'])
        self.m = m
        return m

    ### mean field

    # NOTE: mean field weak-limit style (not the standard superior truncations)
    # wouldn't be too hard if we're okay with sampling m's stochastically;
    # otherwise, to guarantee improvements in the vlb, they'll need their own
    # factors in this class (or another representation entirely)

    @property
    def exp_expected_log_trans_matrix(self):
        raise NotImplementedError

    def meanfieldupdate(self,*args,**kwargs):
        raise NotImplementedError

    def get_vlb(self):
        raise NotImplementedError


# TODO sticky
# need init for kappa, need my own _get_m that subtracts
# that'd work well as an extension, but can we also do the conc version as a
# mixin? init takes off kappa, thats easy as a mixin. yeah kappa glued on the
# end is perfect for mixin
# what do i want at the end of the day? sticky with fixed all, sticky with conc
# all. just extend directly, override _get_m

class WeakLimitStickyHDPHMMTransitions(WeakLimitHDPHMMTransitions):
    def __init__(self,kappa,**kwargs):
        self.kappa = kappa
        super(WeakLimitStickyHDPHMMTransitions,self).__init__(**kwargs)

    def _get_m(self,trans_counts):
        ms = super(WeakLimitStickyHDPHMMTransitions,self)._get_m(trans_counts)
        # need to thin the m's, then use those thinned counts to resample kappa
        newms = ms.copy()
        if ms.sum() > 0:
            # np.random.binomial fails when n=0, so pull out nonzero indices
            indices = np.nonzero(newms.flat[::ms.shape[0]+1])
            newms.flat[::ms.shape[0]+1][indices] = np.array(np.random.binomial(
                    ms.flat[::ms.shape[0]+1][indices],
                    self.beta[indices]*self.alpha/(self.beta[indices]*self.alpha + self.kappa)),
                    dtype=np.int32)
        return newms


class WeakLimitHDPHMMTransitionsConc(HMMTransitions):
    # NOTE: this class handles both gamma and alpha resampling because alpha
    # resampling is affected by beta. inheriting from WeakLimitHDPHMMTransitions
    # didn't work because of the order in which this class should resample
    # things, so parts of that class are copied here

    def __init__(self,num_states,gamma_a_0,gamma_b_0,alpha_a_0,alpha_b_0,
            beta=None,trans_matrix=None,**kwargs):
        if num_states is None:
            assert beta is not None or trans_matrix is not None
            self.N = len(beta) if beta is not None else trans_matrix.shape[0]
        else:
            self.N = num_states

        self.beta_obj = MultinomialAndConcentration(a_0=gamma_a_0,b_0=gamma_b_0,
                K=self.N,weights=beta)
        self.alpha_obj = GammaCompoundDirichlet(self.N,alpha_a_0,alpha_b_0)

        super(WeakLimitHDPHMMTransitionsConc,self).__init__(
                num_states=self.N,alphav=self.alpha*self.beta,
                trans_matrix=trans_matrix,**kwargs)

    # next three methods copied from WeakLimitHDPHMMTransitions

    @property
    def beta(self):
        return self.beta_obj.weights

    @property
    def gamma(self):
        return self.beta_obj.alpha_0

    def _get_m(self,trans_counts):
        # splits tables by running a CRP on each
        # TODO factor this out into a util function, get rid of weave
        m = np.zeros_like(trans_counts)
        N = m.shape[0]
        if not (0 == trans_counts).all():
            alpha, beta = self.alpha, self.beta
            scipy.weave.inline(
            '''
            for (int i=0; i<N; i++) {
                for (int j=0; j<N; j++) {
                    int tot = 0;
                    for (int k=0; k<trans_counts[N*i+j]; k++) {
                        tot += ((double)rand())/RAND_MAX < (alpha * beta[j])/(k+alpha*beta[j]);
                    }
                    m[N*i+j] = tot;
                }
            }
            ''',
            ['trans_counts','N','m','alpha','beta'],
            extra_compile_args=['-O3'])
        self.m = m
        return m

    def _get_alpha(self):
            return self.alpha_obj.concentration

    def _set_alpha(self,alpha):
        pass # NOTE: this is necessary because of HMMTransitions.__init__

    alpha = property(_get_alpha,_set_alpha)

    def resample(self,stateseqs=[],trans_counts=None,ms=None):
        trans_counts = self._count_transitions(stateseqs) if trans_counts is None \
                else trans_counts
        ms = self._get_m(trans_counts) if ms is None else ms
        self.beta_obj.resample(ms)
        self.alpha_obj.resample(trans_counts,weighted_cols=self.beta)
        for d in self._row_distns:
            d.alphav_0 = self.alpha * self.beta
        return super(WeakLimitHDPHMMTransitionsConc,self).resample(
                stateseqs=stateseqs,trans_counts=trans_counts)

    def meanfieldupdate(self,*args,**kwargs):
        raise NotImplementedError # TODO


class WeakLimitHDPHSMMTransitions(WeakLimitHDPHMMTransitions,HSMMTransitions):
    # NOTE: required data augmentation handled in HSMMTransitions._count_transitions
    pass


class WeakLimitHDPHSMMTransitionsConc(WeakLimitHDPHMMTransitionsConc,HSMMTransitions):
    # NOTE: required data augmentation handled in HSMMTransitions._count_transitions
    pass

# class _ConcentrationResamplingMixin(object):
#     def __init__(self,state_dim,alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0):
#         self.gamma_obj = GammaCompoundDirichlet(state_dim,gamma_a_0,gamma_b_0)
#         self.alpha_obj = GammaCompoundDirichlet(state_dim,alpha_a_0,alpha_b_0)

#     def resample(self,*args,**kwargs):
#         # TODO can this work? needs to re-generation trans counts, m. looks at
#         # beta. at what point should this happen in the Gibbs steps? each should
#         # be resampled before the respective weights are resampled
#         self.alpha_obj.resample(self.trans_counts,weighted_cols=self.beta)
#         self.gamma_obj.resample(self.m)
#         super(_ConcentrationResamplingMixin,self).resample(*args,**kwargs)

#     def _get_alpha(self):
#         return self.alpha_obj.concentration

#     def _set_alpha(self,alpha):
#         self.alpha_obj.concentration = alpha

#     alpha = property(_get_alpha,_set_alpha)

#     def _get_gamma(self):
#         return self.gamma_obj.concentration

#     def _set_gamma(self,gamma):
#         self.gamma_obj.concentration = gamma

#     gamma = property(_get_alpha,_set_alpha)

################################
#  Weak-Limit HDP-HMM classes  #
################################

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

    def copy_sample(self):
        new = copy.deepcopy(self)
        if hasattr(new,'trans_counts'):
            del new.trans_counts
        if hasattr(new,'m'):
            del new.m
        return new

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

    def _get_m_slow(self,trans_counts):
        m = np.zeros((self.state_dim,self.state_dim),dtype=np.int32)
        if not (0 == trans_counts).all():
            for (rowidx, colidx), val in np.ndenumerate(trans_counts):
                if val > 0:
                    m[rowidx,colidx] = (np.random.rand(val) < self.alpha * self.beta[colidx] \
                            /(np.arange(val) + self.alpha*self.beta[colidx])).sum()
        self.m = m
        return m

    def _get_m(self,trans_counts):
        N = trans_counts.shape[0]
        m = np.zeros((N,N),dtype=np.int32)
        if not (0 == trans_counts).all():
            alpha, beta = self.alpha, self.beta
            scipy.weave.inline(
                    '''
                    for (int i=0; i<N; i++) {
                        for (int j=0; j<N; j++) {
                            int tot = 0;
                            for (int k=0; k<trans_counts[N*i+j]; k++) {
                                tot += ((double)rand())/RAND_MAX < (alpha * beta[j])/(k+alpha*beta[j]);
                            }
                            m[N*i+j] = tot;
                        }
                    }
                    ''',
                    ['trans_counts','N','m','alpha','beta'],
                    extra_compile_args=['-O3'])
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

        self.A[np.isnan(self.A)] = 0. # 1./self.state_dim # NOTE: just a reasonable hack, o/w undefined

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


class HDPHMMTransitionsConcResampling(_ConcentrationResamplingMixin,HDPHMMTransitions):
    def __init__(self,state_dim,
            alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0,
            **kwargs):

        _ConcentrationResamplingMixin.__init__(
                self,state_dim,alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0)
        HDPHMMTransitions.__init__(state_dim,alpha=self.alpha,gamma=self.gamma,**kwargs)


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


class HDPHSMMTransitionsConcResampling(HDPHSMMTransitions,_ConcentrationResamplingMixin):
    def __init__(self,state_dim,
            alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0,
            **kwargs):

        _ConcentrationResamplingMixin.__init__(
                self,state_dim,alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0)
        super(HDPHSMMTransitionsConcResampling,self).__init__(state_dim,
                alpha=self.alpha_obj.concentration*state_dim,
                gamma=self.gamma_obj.concentration*state_dim,
                **kwargs)

    def resample(self,*args,**kwargs):
        super(HDPHSMMTransitionsConcResampling,self).resample(*args,**kwargs)
        _ConcentrationResamplingMixin.resample(self)


############################
#  Sticky HDP-HMM classes  #
############################

class StickyHDPHMMTransitions(HDPHMMTransitions):
    def __init__(self,kappa,*args,**kwargs):
        self.kappa = kappa
        super(StickyHDPHMMTransitions,self).__init__(*args,**kwargs)

    def copy_sample(self):
        new = super(StickyHDPHMMTransitions,self).copy_sample()
        if hasattr(new,'newm'):
            del new.newm
        return new

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


class StickyHDPHMMTransitionsConcResampling(StickyHDPHMMTransitions,_ConcentrationResamplingMixin):
    # NOTE: this parameterizes (alpha plus kappa) and rho, as in EB Fox's thesis
    # the class member named 'alpha_obj' should really be 'alpha_plus_kappa_obj'
    # the member 'alpha' is still just alpha
    def __init__(self,state_dim,
            rho_a_0,rho_b_0,
            alphakappa_a_0,alphakappa_b_0,gamma_a_0,gamma_b_0,
            **kwargs):

        self.rho_a_0, self.rho_b_0 = rho_a_0, rho_b_0
        self.rho = np.random.beta(rho_a_0,rho_b_0)
        _ConcentrationResamplingMixin.__init__(
                self,state_dim,alphakappa_a_0,alphakappa_b_0,gamma_a_0,gamma_b_0)
        super(StickyHDPHMMTransitionsConcResampling,self).__init__(state_dim=state_dim,
                kappa=self.rho * self.alpha_obj.concentration*state_dim,
                alpha=(1-self.rho) * self.alpha_obj.concentration*state_dim,
                gamma=self.gamma_obj.concentration*state_dim,
                **kwargs)

    def resample(self,*args,**kwargs):
        super(StickyHDPHMMTransitionsConcResampling,self).resample(*args,**kwargs)
        _ConcentrationResamplingMixin.resample(self)
        self.rho = np.random.beta(self.rho_a_0 + (self.m-self.newm).sum(), self.rho_b_0 + self.newm.sum())
        self.kappa = self.rho * self.alpha_obj.concentration*self.state_dim


class StickyLTRHDPHMMTransitions(LTRHDPHMMTransitions,StickyHDPHMMTransitions):
    def _resample_A(self,data):
        aug_data = data + np.diag(self.kappa * np.ones(data.shape[0]))
        super(StickyLTRHDPHMMTransitions,self)._resample_A(aug_data)

