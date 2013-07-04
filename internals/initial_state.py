from __future__ import division
import numpy as np

from ..util.general import top_eigenvector
from ..basic.abstractions import GibbsSampling, MaxLikelihood
from ..basic.distributions import Categorical

class InitialState(Categorical):
    def __init__(self,state_dim,rho,pi_0=None):
        super(InitialState,self).__init__(alpha_0=rho,K=state_dim,weights=pi_0)

    @property
    def pi_0(self):
        return self.weights

class StartInZero(GibbsSampling,MaxLikelihood):
    def __init__(self,state_dim,**kwargs):
        self.pi_0 = np.zeros(state_dim)
        self.pi_0[0] = 1.

    def resample(self,init_states=np.array([])):
        pass

    def rvs(self,size=[]):
        return np.zeros(size)

    def max_likelihood(*args,**kwargs):
        pass

class Uniform(GibbsSampling,MaxLikelihood):
    def __init__(self,state_dim,**kwargs):
        self.pi_0 = np.ones(state_dim)

    def resample(self,init_states=np.array([])):
        pass

    def rvs(self,size=[]):
        return np.random.random_integers(self.pi_0.shape[0],size=size)

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
            self._pi = top_eigenvector(self.model.trans_distn.A)
        return self._pi

class HSMMSteadyState(SteadyState):
    @property
    def pi_0(self):
        if self._pi is None:
            markov_part = super(HSMMSteadyState,self).pi_0.fget(self)
            duration_expectations = np.array([d.mean for d in self.model.dur_distns])
            self._pi = markov_part * duration_expectations
            self._pi /= self._pi.sum()
        return self._pi

