from __future__ import division
import numpy as np

from ..util.general import top_eigenvector
from ..basic.abstractions import GibbsSampling, MaxLikelihood
from ..basic.distributions import Categorical

class InitialState(Categorical):
    def __init__(self,num_states=None,rho=None,pi_0=None):
        super(InitialState,self).__init__(alpha_0=rho,K=num_states,weights=pi_0)

    @property
    def pi_0(self):
        return self.weights

class StartInZero(GibbsSampling,MaxLikelihood):
    def __init__(self,num_states,**kwargs):
        self.pi_0 = np.zeros(num_states)
        self.pi_0[0] = 1.

    @property
    def params(self):
        return dict(pi_0=self.pi_0)

    @property
    def hypparams(self):
        return dict()

    def resample(self,init_states=np.array([])):
        pass

    def rvs(self,size=[]):
        return np.zeros(size)

    def max_likelihood(*args,**kwargs):
        pass

class SteadyState(object):
    def __init__(self,model):
        self.model = model
        self.clear_caches()

    def clear_caches(self):
        self._pi = None

    @property
    def pi_0(self):
        if self._pi is None:
            self._pi = top_eigenvector(self.model.trans_distn.trans_matrix)
        return self._pi

    def resample(self,*args,**kwargs):
        pass

class HSMMSteadyState(SteadyState):
    @property
    def pi_0(self):
        if self._pi is None:
            markov_part = super(HSMMSteadyState,self).pi_0
            duration_expectations = np.array([d.mean for d in self.model.dur_distns])
            self._pi = markov_part * duration_expectations
            self._pi /= self._pi.sum()
        return self._pi

