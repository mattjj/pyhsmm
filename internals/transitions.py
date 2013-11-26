from __future__ import division
import numpy as np
import scipy.stats as stats
import scipy.weave
from numpy import newaxis as na
np.seterr(invalid='raise')
import operator
import copy

from ..basic.abstractions import GibbsSampling
from ..basic.distributions import GammaCompoundDirichlet, Multinomial, \
        MultinomialAndConcentration
from ..util.general import rle, count_transitions

# TODO this file names the conc parameter for beta to be 'gamma', but that's
# switched from my old code. don't do that! switch it back!
# TODO should probably have a _0 suffix as well
# TODO WeakLimitSticky concentration resampling classes (also resample kappa)
# TODO update HDP left-to-right classes, old versions in scrap.py

########################
#  HMM / HSMM classes  #
########################

# NOTE: no hierarchical priors here (i.e. no number-of-states inference)

class _HMMTransitionsBase(object):
    def __init__(self,num_states=None,alpha=None,alphav=None,trans_matrix=None):
        self.N = num_states
        self.trans_matrix = trans_matrix

        if trans_matrix is None and (None not in (alpha,self.N) or alphav is not None):
            self._row_distns = [Multinomial(alpha_0=alpha,K=self.N,alphav_0=alphav)
                    for n in xrange(self.N)] # sample from prior

    @property
    def trans_matrix(self):
        return np.array([d.weights for d in self._row_distns])

    @trans_matrix.setter
    def trans_matrix(self,trans_matrix):
        if trans_matrix is not None:
            N = self.N = trans_matrix.shape[0]
            self._row_distns = \
                    [Multinomial(alpha_0=self.alpha,K=N,alphav_0=self.alphav,weights=row)
                            for row in trans_matrix]

    @property
    def alpha(self):
        return self._row_distns[0].alpha_0

    @alpha.setter
    def alpha(self,val):
        for distn in self._row_distns:
            distn.alpha_0 = val

    @property
    def alphav(self):
        return self._row_distns[0].alphav_0

    @alphav.setter
    def alphav(self,weights):
        for distn in self._row_distns:
            distn.alphav_0 = weights

    def _count_transitions(self,stateseqs):
        if len(stateseqs) == 0 or isinstance(stateseqs,np.ndarray) \
                or isinstance(stateseqs[0],int) or isinstance(stateseqs[0],float):
            return count_transitions(stateseqs,minlength=self.N)
        else:
            return sum(count_transitions(stateseq,minlength=self.N)
                    for stateseq in stateseqs)

class _HMMTransitionsGibbs(_HMMTransitionsBase):
    def resample(self,stateseqs=[],trans_counts=None):
        trans_counts = self._count_transitions(stateseqs) if trans_counts is None \
                else trans_counts
        for distn, counts in zip(self._row_distns,trans_counts):
            distn.resample(counts)
        return self

class _HMMTransitionsMaxLikelihood(_HMMTransitionsBase):
    def max_likelihood(self,stateseqs=None,expected_transcounts=None):
        trans_counts = sum(expected_transcounts) if stateseqs is None \
                else self._count_transitions(stateseqs)
        # NOTE: could just call max_likelihood on each trans row, but this way
        # it handles a few lazy-initialization cases (e.g. if _row_distns aren't
        # initialized)
        errs = np.seterr(invalid='ignore',divide='ignore')
        self.trans_matrix = np.nan_to_num(trans_counts / trans_counts.sum(1)[:,na])
        np.seterr(**errs)

        return self

class _HMMTransitionsMeanField(_HMMTransitionsBase):
    @property
    def exp_expected_log_trans_matrix(self):
        # NOTE: this is Atilde in Matthew Beal's PhD
        # NOTE: see Multinomial.expected_log_likelihood for no-args explanation
        return np.exp(np.array([distn.expected_log_likelihood()
            for distn in self._row_distns]))

    def meanfieldupdate(self,expected_transcounts):
        trans_softcounts = sum(expected_transcounts)
        # NOTE: None is placeholder, see Multinomial class def
        for distn, counts in zip(self._row_distns, trans_softcounts):
            distn.meanfieldupdate(None,counts)
        return self

    def get_vlb(self):
        return sum(distn.get_vlb() for distn in self._row_distns)

class HMMTransitions(
        _HMMTransitionsGibbs,
        _HMMTransitionsMeanField,
        _HMMTransitionsMaxLikelihood):
    pass


class _HSMMTransitionsBase(_HMMTransitionsBase):
    @property
    def trans_matrix(self):
        out = self.full_trans_matrix
        out.flat[::out.shape[0]+1] = 0
        out /= out.sum(1)[:,na]
        return out

    @trans_matrix.setter
    def trans_matrix(self,trans_matrix):
        HMMTransitions.trans_matrix.fset(self,trans_matrix)

class _HSMMTransitionsGibbs(_HMMTransitionsGibbs):
    # NOTE: in this non-hierarchical case, we wouldn't need the below data
    # augmentation if we were only to update the distribution on off-diagonal
    # components. but it's easier to code if we just keep everything complete
    # dirichlet/multinomial and do the data augmentation here, especially since
    # we'll need it for the hierarchical case anyway.

    @property
    def full_trans_matrix(self):
        return super(_HSMMTransitionsGibbs,self).trans_matrix

    def _count_transitions(self,stateseqs):
        stateseq_noreps = [rle(stateseq)[0] for stateseq in stateseqs]
        trans_counts = super(_HSMMTransitionsGibbs,self)._count_transitions(stateseq_noreps)

        if trans_counts.sum() > 0:
            froms = trans_counts.sum(1)
            self_trans = [np.random.geometric(1-A_ii,size=n).sum() if n > 0 else 0
                    for A_ii, n in zip(self.full_trans_matrix.diagonal(),froms)]
            trans_counts += np.diag(self_trans)

        return trans_counts

class HSMMTransitions(_HSMMTransitionsGibbs):
    pass


class _ConcentrationResamplingMixin(object):
    # NOTE: because all rows share the same concentration parameter, we can't
    # use CategoricalAndConcentration; gotta use GammaCompoundDirichlet directly
    def __init__(self,num_states,alpha_a_0,alpha_b_0,**kwargs):
        self.alpha_obj = GammaCompoundDirichlet(num_states,alpha_a_0,alpha_b_0)
        super(_ConcentrationResamplingMixin,self).__init__(
                num_states=num_states,alpha=self.alpha,**kwargs)

    @property
    def alpha(self):
        return self.alpha_obj.concentration

    @alpha.setter
    def alpha(self,alpha):
        if alpha is not None:
            self.alpha_obj.concentration = alpha # a no-op when called internally
            for d in self._row_distns:
                d.alpha_0 = alpha

    def resample(self,stateseqs=[],trans_counts=None):
        trans_counts = self._count_transitions(stateseqs) if trans_counts is None \
                else trans_counts

        self._resample_alpha(trans_counts)

        return super(_ConcentrationResamplingMixin,self).resample(
                stateseqs=stateseqs,trans_counts=trans_counts)

    def _resample_alpha(self,trans_counts):
        self.alpha_obj.resample(trans_counts)
        self.alpha = self.alpha_obj.concentration

    def meanfieldupdate(self,*args,**kwargs):
        raise NotImplementedError # TODO

class HMMTransitionsConc(_ConcentrationResamplingMixin,_HMMTransitionsGibbs):
    pass

class HSMMTransitionsConc(_ConcentrationResamplingMixin,_HSMMTransitionsGibbs):
    pass

############################
#  Weak-Limit HDP classes  #
############################

class _WeakLimitHDPHMMTransitionsBase(_HMMTransitionsBase):
    def __init__(self,gamma,alpha,num_states=None,beta=None,trans_matrix=None):
        if num_states is None:
            assert beta is not None or trans_matrix is not None
            self.N = len(beta) if beta is not None else trans_matrix.shape[0]
        else:
            self.N = num_states

        self.alpha = alpha
        self.beta_obj = Multinomial(alpha_0=gamma,K=self.N,weights=beta)

        super(_WeakLimitHDPHMMTransitionsBase,self).__init__(
                num_states=self.N,alpha=alpha,
                alphav=alpha*self.beta,trans_matrix=trans_matrix)

    @property
    def beta(self):
        return self.beta_obj.weights

    @beta.setter
    def beta(self,weights):
        self.beta_obj.weights = weights
        self.alphav = self.alpha * self.beta

    @property
    def gamma(self):
        return self.beta_obj.alpha_0

    @gamma.setter
    def gamma(self,val):
        self.beta_obj.alpha_0 = val

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self,val):
        self._alpha = val

class _WeakLimitHDPHMMTransitionsGibbs(
        _WeakLimitHDPHMMTransitionsBase,
        _HMMTransitionsGibbs):
    def resample(self,stateseqs=[],trans_counts=None,ms=None):
        trans_counts = self._count_transitions(stateseqs) if trans_counts is None \
                else trans_counts
        ms = self._get_m(trans_counts) if ms is None else ms

        self._resample_beta(ms)

        return super(_WeakLimitHDPHMMTransitionsGibbs,self).resample(
                stateseqs=stateseqs,trans_counts=trans_counts)

    def _resample_beta(self,ms):
        self.beta_obj.resample(ms)
        self.alphav = self.alpha * self.beta

    def _get_m(self,trans_counts):
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

class WeakLimitHDPHMMTransitions(_WeakLimitHDPHMMTransitionsGibbs,_HMMTransitionsMaxLikelihood):
    # NOTE: include MaxLikelihood for convenience
    pass


class _WeakLimitHDPHMMTransitionsConcBase(_WeakLimitHDPHMMTransitionsBase):
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

        # NOTE: we don't want to call WeakLimitHDPHMMTransitions.__init__
        # because it sets beta_obj in a different way
        _HMMTransitionsBase.__init__(
                self, num_states=self.N, alphav=self.alpha*self.beta,
                trans_matrix=trans_matrix, **kwargs)

    @property
    def alpha(self):
        return self.alpha_obj.concentration

    @alpha.setter
    def alpha(self,val):
        self.alpha_obj.concentration = val

class _WeakLimitHDPHMMTransitionsConcGibbs(
        _WeakLimitHDPHMMTransitionsConcBase,_WeakLimitHDPHMMTransitionsGibbs):
    def resample(self,stateseqs=[],trans_counts=None,ms=None):
        trans_counts = self._count_transitions(stateseqs) if trans_counts is None \
                else trans_counts
        ms = self._get_m(trans_counts) if ms is None else ms

        self._resample_beta(ms)
        self._resample_alpha(trans_counts)

        return super(_WeakLimitHDPHMMTransitionsConcGibbs,self).resample(
                stateseqs=stateseqs,trans_counts=trans_counts)

    def _resample_beta(self,ms):
        # NOTE: unlike parent, alphav is updated in _resample_alpha
        self.beta_obj.resample(ms)

    def _resample_alpha(self,trans_counts):
        self.alpha_obj.resample(trans_counts,weighted_cols=self.beta)
        self.alphav = self.alpha * self.beta

class WeakLimitHDPHMMTransitionsConc(_WeakLimitHDPHMMTransitionsConcGibbs):
    pass


class _WeakLimitStickyHDPHMMTransitionsBase(_WeakLimitHDPHMMTransitionsBase):
    def __init__(self,kappa,**kwargs):
        self.kappa = kappa
        super(_WeakLimitStickyHDPHMMTransitionsBase,self).__init__(**kwargs)


class _WeakLimitStickyHDPHMMTransitionsGibbs(
        _WeakLimitStickyHDPHMMTransitionsBase,_WeakLimitHDPHMMTransitionsGibbs):
    def _set_alphav(self,weights):
        for distn, delta_ij in zip(self._row_distns,np.eye(self.N)):
            distn.alphav_0 = weights + self.kappa * delta_ij

    alphav = property(_WeakLimitHDPHMMTransitionsGibbs.alphav.fget,_set_alphav)

    def _get_m(self,trans_counts):
        # NOTE: this thins the m's
        ms = super(_WeakLimitStickyHDPHMMTransitionsGibbs,self)._get_m(trans_counts)
        newms = ms.copy()
        if ms.sum() > 0:
            # np.random.binomial fails when n=0, so pull out nonzero indices
            indices = np.nonzero(newms.flat[::ms.shape[0]+1])
            newms.flat[::ms.shape[0]+1][indices] = np.array(np.random.binomial(
                    ms.flat[::ms.shape[0]+1][indices],
                    self.beta[indices]*self.alpha/(self.beta[indices]*self.alpha + self.kappa)),
                    dtype=np.int32)
        return newms

class WeakLimitStickyHDPHMMTransitions(_WeakLimitStickyHDPHMMTransitionsGibbs):
    pass


class WeakLimitHDPHSMMTransitions(
        _HSMMTransitionsGibbs,_WeakLimitHDPHMMTransitionsGibbs):
    # NOTE: required data augmentation handled in HSMMTransitions._count_transitions
    pass


class WeakLimitHDPHSMMTransitionsConc(
        _HSMMTransitionsGibbs,_WeakLimitHDPHMMTransitionsConcGibbs):
    # NOTE: required data augmentation handled in HSMMTransitions._count_transitions
    pass

