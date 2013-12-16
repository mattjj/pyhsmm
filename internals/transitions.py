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

