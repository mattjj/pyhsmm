from __future__ import division
from builtins import range
import numpy as np
from numpy import newaxis as na
np.seterr(invalid='raise')
import copy

from scipy.special import digamma

from pyhsmm.basic.distributions import GammaCompoundDirichlet, Multinomial, \
    MultinomialAndConcentration
from pyhsmm.util.general import rle, cumsum, rcumsum
try:
    from pyhsmm.util.cstats import sample_crp_tablecounts, count_transitions
except ImportError:
    from warnings import warn
    warn('using slow transition counting')
    from pyhsmm.util.stats import sample_crp_tablecounts, count_transitions

# TODO separate out bayesian and nonbayesian versions?

########################
#  HMM / HSMM classes  #
########################

# NOTE: no hierarchical priors here (i.e. no number-of-states inference)

### HMM

class _HMMTransitionsBase(object):
    def __init__(self,num_states=None,alpha=None,alphav=None,trans_matrix=None):
        self.N = num_states

        if trans_matrix is not None:
            self._row_distns = [Multinomial(alpha_0=alpha,K=self.N,alphav_0=alphav,
                weights=row) for row in trans_matrix]
        elif None not in (alpha,self.N) or alphav is not None:
            self._row_distns = [Multinomial(alpha_0=alpha,K=self.N,alphav_0=alphav)
                    for n in range(self.N)] # sample from prior

    @property
    def trans_matrix(self):
        return np.array([d.weights for d in self._row_distns])

    @trans_matrix.setter
    def trans_matrix(self,trans_matrix):
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
        assert isinstance(stateseqs,list) and all(isinstance(s,np.ndarray) for s in stateseqs)
        return sum((count_transitions(s,num_states=self.N) for s in stateseqs),
                np.zeros((self.N,self.N),dtype=np.int32))

    def copy_sample(self):
        new = copy.copy(self)
        new._row_distns = [distn.copy_sample() for distn in self._row_distns]
        return new

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
        trans_matrix = np.nan_to_num(trans_counts / trans_counts.sum(1)[:,na])
        np.seterr(**errs)

        # all-zero rows get set to uniform
        trans_matrix[trans_matrix.sum(1) == 0] = 1./trans_matrix.shape[0]
        assert np.allclose(trans_matrix.sum(1),1.)

        self.trans_matrix = trans_matrix

        return self

class _HMMTransitionsMeanField(_HMMTransitionsBase):
    @property
    def exp_expected_log_trans_matrix(self):
        return np.exp(np.array([distn.expected_log_likelihood()
            for distn in self._row_distns]))

    def meanfieldupdate(self,expected_transcounts):
        assert isinstance(expected_transcounts,list) and len(expected_transcounts) > 0
        trans_softcounts = sum(expected_transcounts)
        for distn, counts in zip(self._row_distns,trans_softcounts):
            distn.meanfieldupdate(None,counts)
        return self

    def get_vlb(self):
        return sum(distn.get_vlb() for distn in self._row_distns)

    def _resample_from_mf(self):
        for d in self._row_distns:
            d._resample_from_mf()

class _HMMTransitionsSVI(_HMMTransitionsMeanField):
    def meanfield_sgdstep(self,expected_transcounts,prob,stepsize):
        assert isinstance(expected_transcounts,list)
        if len(expected_transcounts) > 0:
            trans_softcounts = sum(expected_transcounts)
            for distn, counts in zip(self._row_distns,trans_softcounts):
                distn.meanfield_sgdstep(None,counts,prob,stepsize)
        return self

class HMMTransitions(
        _HMMTransitionsGibbs,
        _HMMTransitionsSVI,
        _HMMTransitionsMeanField,
        _HMMTransitionsMaxLikelihood):
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

### HSMM

class _HSMMTransitionsBase(_HMMTransitionsBase):
    def _get_trans_matrix(self):
        out = self.full_trans_matrix
        out.flat[::out.shape[0]+1] = 0
        errs = np.seterr(invalid='ignore')
        out /= out.sum(1)[:,na]
        out = np.nan_to_num(out)
        np.seterr(**errs)
        return out

    trans_matrix = property(_get_trans_matrix,_HMMTransitionsBase.trans_matrix.fset)

    @property
    def full_trans_matrix(self):
        return super(_HSMMTransitionsBase,self).trans_matrix

    def _count_transitions(self,stateseqs):
        stateseq_noreps = [rle(stateseq)[0] for stateseq in stateseqs]
        return super(_HSMMTransitionsBase,self)._count_transitions(stateseq_noreps)

class _HSMMTransitionsGibbs(_HSMMTransitionsBase,_HMMTransitionsGibbs):
    # NOTE: in this non-hierarchical case, we wouldn't need the below data
    # augmentation if we were only to update the distribution on off-diagonal
    # components. but it's easier to code if we just keep everything complete
    # dirichlet/multinomial and do the data augmentation here, especially since
    # we'll need it for the hierarchical case anyway.

    def _count_transitions(self,stateseqs):
        trans_counts = super(_HSMMTransitionsGibbs,self)._count_transitions(stateseqs)

        if trans_counts.sum() > 0:
            froms = trans_counts.sum(1)
            self_trans = [np.random.geometric(1-A_ii,size=n).sum() if n > 0 else 0
                    for A_ii, n in zip(self.full_trans_matrix.diagonal(),froms)]
            trans_counts += np.diag(self_trans)

        return trans_counts

class _HSMMTransitionsMaxLikelihood(_HSMMTransitionsBase,_HMMTransitionsMaxLikelihood):
    def max_likelihood(self,stateseqs=None,expected_transcounts=None):
        trans_counts = sum(expected_transcounts) if stateseqs is None \
                else self._count_transitions(stateseqs)
        # NOTE: we could just call max_likelihood on each trans row, but this
        # way it's a bit nicer
        errs = np.seterr(invalid='ignore',divide='ignore')
        trans_matrix = np.nan_to_num(trans_counts / trans_counts.sum(1)[:,na])
        np.seterr(**errs)

        # all-zero rows get set to uniform
        trans_matrix[trans_matrix.sum(1) == 0] = 1./(trans_matrix.shape[0]-1)
        trans_matrix.flat[::trans_matrix.shape[0]+1] = 0.

        self.trans_matrix = trans_matrix
        assert np.allclose(0.,np.diag(self.trans_matrix))
        assert np.allclose(1.,self.trans_matrix.sum(1))

        return self

class _HSMMTransitionsMeanField(_HSMMTransitionsBase,_HMMTransitionsMeanField):
    pass

class _HSMMTransitionsSVI(_HSMMTransitionsMeanField,_HMMTransitionsSVI):
    pass

class HSMMTransitions(_HSMMTransitionsGibbs,
        _HSMMTransitionsMaxLikelihood,
        _HSMMTransitionsSVI,
        _HSMMTransitionsMeanField):
    # NOTE: include MaxLikelihood for convenience, uses
    # _HMMTransitionsBase._count_transitions
    pass

class HSMMTransitionsConc(_ConcentrationResamplingMixin,_HSMMTransitionsGibbs):
    pass

############################
#  Weak-Limit HDP classes  #
############################

### HDP-HMM

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

    def copy_sample(self):
        new = super(_WeakLimitHDPHMMTransitionsBase,self).copy_sample()
        new.beta_obj = self.beta_obj.copy_sample()
        return new

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
        if not (0 == trans_counts).all():
            m = sample_crp_tablecounts(float(self.alpha),trans_counts,self.beta)
        else:
            m = np.zeros_like(trans_counts)
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

    def copy_sample(self):
        new = super(_WeakLimitHDPHMMTransitionsConcGibbs,self).copy_sample()
        new.alpha_obj = self.alpha_obj.copy_sample()
        return new

class WeakLimitHDPHMMTransitionsConc(_WeakLimitHDPHMMTransitionsConcGibbs):
    pass

# Sticky HDP-HMM

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

class _WeakLimitStickyHDPHMMTransitionsConcGibbs(
        _WeakLimitStickyHDPHMMTransitionsGibbs,_WeakLimitHDPHMMTransitionsConcGibbs):
    pass

class WeakLimitStickyHDPHMMTransitions(
        _WeakLimitStickyHDPHMMTransitionsGibbs,_HMMTransitionsMaxLikelihood):
    # NOTE: includes MaxLikelihood for convenience
    pass

class WeakLimitStickyHDPHMMTransitionsConc(
        _WeakLimitStickyHDPHMMTransitionsConcGibbs):
    pass

# DA Truncation

class _DATruncHDPHMMTransitionsBase(_HMMTransitionsBase):
    # NOTE: self.beta stores \beta_{1:K}, so \beta_{\text{rest}} is implicit

    def __init__(self,gamma,alpha,num_states,beta=None,trans_matrix=None):
        self.N = num_states
        self.gamma = gamma
        self._alpha = alpha
        if beta is None:
            beta = np.ones(num_states) / (num_states + 1)
            # beta = self._sample_GEM(gamma,num_states)
        assert not np.isnan(beta).any()

        betafull = np.concatenate(((beta,(1.-beta.sum(),))))

        super(_DATruncHDPHMMTransitionsBase,self).__init__(
                num_states=self.N,alphav=alpha*betafull,trans_matrix=trans_matrix)

        self.beta = beta

    @staticmethod
    def _sample_GEM(gamma,K):
        v = np.random.beta(1.,gamma,size=K)
        return v * np.concatenate(((1.,),np.cumprod(1.-v[:-1])))

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self,beta):
        self._beta = beta
        self.alphav = self._alpha * np.concatenate((beta,(1.-beta.sum(),)))

    @property
    def exp_expected_log_trans_matrix(self):
        return super(_DATruncHDPHMMTransitionsBase,self).exp_expected_log_trans_matrix[:,:-1].copy()

    @property
    def trans_matrix(self):
        return super(_DATruncHDPHMMTransitionsBase,self).trans_matrix[:,:-1].copy()

class _DATruncHDPHMMTransitionsSVI(_DATruncHDPHMMTransitionsBase,_HMMTransitionsSVI):
    def meanfieldupdate(self,expected_transcounts):
        super(_DATruncHDPHMMTransitionsSVI,self).meanfieldupdate(
                self._pad_zeros(expected_transcounts))

    def meanfield_sgdstep(self,expected_transcounts,prob,stepsize):
        # NOTE: since we take a step on q(beta) and on q(pi) at the same time
        # (as usual with SVI), we compute the beta gradient and perform the pi
        # step before applying the beta gradient

        beta_gradient = self._beta_gradient()
        super(_DATruncHDPHMMTransitionsSVI,self).meanfield_sgdstep(
                self._pad_zeros(expected_transcounts),prob,stepsize)
        self.beta = self._feasible_step(self.beta,beta_gradient,stepsize)
        assert (self.beta >= 0.).all() and self.beta.sum() < 1
        return self

    def _pad_zeros(self,counts):
        if isinstance(counts,np.ndarray):
            return np.pad(counts,((0,1),(0,1)),mode='constant',constant_values=0)
        return [self._pad_zeros(c) for c in counts]

    @staticmethod
    def _feasible_step(pt,grad,stepsize):
        def newpt(pt,grad,stepsize):
            return pt + stepsize*grad
        def feas(pt):
            return (pt>0.).all() and pt.sum() < 1.
        grad = grad / np.abs(grad).max()
        while True:
            new = newpt(pt,grad,stepsize)
            if feas(new):
                return new
            else:
                grad /= 1.5

    def _beta_gradient(self):
        return self._grad_log_p_beta(self.beta,self.gamma) + \
            sum(self._grad_E_log_p_pi_given_beta(self.beta,self._alpha,
                distn._alpha_mf) for distn in self._row_distns)

    @staticmethod
    def _grad_log_p_beta(beta,alpha):
        # NOTE: switched argument name gamma <-> alpha
        return  -(alpha-1)*rcumsum(1./(1-cumsum(beta))) \
                + 2*rcumsum(1./(1-cumsum(beta,strict=True)),strict=True)

    def _grad_E_log_p_pi_given_beta(self,beta,gamma,alphatildes):
        # NOTE: switched argument name gamma <-> alpha
        retval = gamma*(digamma(alphatildes[:-1]) - digamma(alphatildes[-1])) \
                - gamma * (digamma(gamma*beta) - digamma(gamma))
        return retval

    def get_vlb(self):
        return super(_DATruncHDPHMMTransitionsSVI,self).get_vlb() \
                + self._beta_vlb()

    def _beta_vlb(self):
        return np.log(self.beta).sum() + self.gamma*np.log(1-cumsum(self.beta)).sum() \
               - 3*np.log(1-cumsum(self.beta,strict=True)).sum()

class DATruncHDPHMMTransitions(_DATruncHDPHMMTransitionsSVI):
    pass

### HDP-HSMM

# Weak limit

class WeakLimitHDPHSMMTransitions(
        _HSMMTransitionsGibbs,
        _WeakLimitHDPHMMTransitionsGibbs,
        _HSMMTransitionsMaxLikelihood):
    # NOTE: required data augmentation handled in HSMMTransitions._count_transitions
    # NOTE: include MaxLikelihood for convenience
    pass

class WeakLimitHDPHSMMTransitionsConc(
        _WeakLimitHDPHMMTransitionsConcGibbs,
        _HSMMTransitionsGibbs,
        _HSMMTransitionsMaxLikelihood):
    # NOTE: required data augmentation handled in HSMMTransitions._count_transitions
    pass

# DA Truncation

class _DATruncHDPHSMMTransitionsSVI(_DATruncHDPHMMTransitionsSVI,_HSMMTransitionsSVI):
    # TODO the diagonal terms are still included in the vlb, so it's off by some
    # constant offset

    def _beta_gradient(self):
        return self._grad_log_p_beta(self.beta,self.gamma) + \
            sum(self._zero_ith_component(
                    self._grad_E_log_p_pi_given_beta(self.beta,self._alpha,distn._alpha_mf),i)
                    for i, distn in enumerate(self._row_distns))

    @staticmethod
    def _zero_ith_component(v,i):
        v = v.copy()
        v[i] = 0
        return v

class DATruncHDPHSMMTransitions(_DATruncHDPHSMMTransitionsSVI):
    pass

