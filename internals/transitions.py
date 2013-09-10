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
        if len(stateseqs) == 0 or isinstance(stateseqs,np.ndarray) \
                or isinstance(stateseqs[0],int) or isinstance(stateseqs[0],float):
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
        trans_counts = sum(expected_transcounts) if stateseqs is None \
                else self._count_transitions(stateseqs)

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
    # for convenience, max_likelihood ignores prior and just passes through to
    # parent, though max likelihood doesn't make sense on a Bayesian class
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

    # NOTE: mean field weak-limit style (not the standard truncations)
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
        raise NotImplementedError


class WeakLimitStickyHDPHMMTransitions(WeakLimitHDPHMMTransitions):
    # NOTE: kappa is NOT resampled in this class; that's rolled in with the
    # concentration resampling class below
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

class WeakLimitStickyHDPPHMMTransitionsConc(
        WeakLimitStickyHDPHMMTransitions,WeakLimitHDPHMMTransitionsConc):
    pass


class WeakLimitHDPHSMMTransitions(HSMMTransitions,WeakLimitHDPHMMTransitions):
    # NOTE: required data augmentation handled in HSMMTransitions._count_transitions
    pass


class WeakLimitHDPHSMMTransitionsConc(HSMMTransitions,WeakLimitHDPHMMTransitionsConc):
    # NOTE: required data augmentation handled in HSMMTransitions._count_transitions
    pass


# TODO update HDP left-to-right classes, old versions in scrap.py

