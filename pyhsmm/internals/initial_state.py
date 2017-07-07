from __future__ import division
import numpy as np
import copy

import pyhsmm
from pyhsmm.util.general import top_eigenvector
from pyhsmm.basic.abstractions import GibbsSampling, MaxLikelihood
from pyhsmm.basic.distributions import Categorical

class UniformInitialState(object):
    def __init__(self,model):
        self.model = model

    @property
    def pi_0(self):
        N = self.model.num_states
        return np.ones(N) / N

    @property
    def steady_state_distribution(self):
        return self.pi_0

    @property
    def exp_expected_log_init_state_distn(self):
        return self.pi_0

    def resample(*args,**kwargs):
        pass

    def get_vlb(self):
        return 0.

    def meanfieldupdate(*args,**kwargs):
        pass

    def meanfield_sgdstep(*args,**kwargs):
        pass

    def max_likelihood(*args,**kwargs):
        pass

    def clear_caches(self):
        pass

    def copy_sample(self, new_model):
        new = copy.copy(self)
        new.model = new_model
        return new

class HMMInitialState(Categorical):
    def __init__(self,model,init_state_concentration=None,pi_0=None):
        self.model = model
        if init_state_concentration is not None or pi_0 is not None:
            self._is_steady_state = False
            super(HMMInitialState,self).__init__(
                    alpha_0=init_state_concentration,K=model.num_states,weights=pi_0)
        else:
            self._is_steady_state = True

    @property
    def pi_0(self):
        if self._is_steady_state:
            return self.steady_state_distribution
        else:
            return self.weights

    @pi_0.setter
    def pi_0(self,pi_0):
        self.weights = pi_0

    @property
    def exp_expected_log_init_state_distn(self):
        return np.exp(self.expected_log_likelihood())

    @property
    def steady_state_distribution(self):
        return top_eigenvector(self.model.trans_distn.trans_matrix)

    def clear_caches(self):
        pass

    def meanfieldupdate(self,expected_initial_states_list):
        super(HMMInitialState,self).meanfieldupdate(None,expected_initial_states_list)

    def meanfield_sgdstep(self,expected_initial_states_list,prob,stepsize):
        super(HMMInitialState,self).meanfield_sgdstep(
                None,expected_initial_states_list,prob,stepsize)

    def max_likelihood(self,samples=None,expected_states_list=None):
        super(HMMInitialState,self).max_likelihood(
                data=samples,weights=expected_states_list)

    def copy_sample(self, new_model):
        new = copy.deepcopy(self)
        new.model = new_model
        return new

class StartInZero(GibbsSampling,MaxLikelihood):
    def __init__(self,num_states,**kwargs):
        self.pi_0 = np.zeros(num_states)
        self.pi_0[0] = 1.

    def resample(self,init_states=np.array([])):
        pass

    def rvs(self,size=[]):
        return np.zeros(size)

    def max_likelihood(*args,**kwargs):
        pass

    def copy_sample(self, new_model):
        new = copy.copy(self)
        new.model = new_model
        return new

class HSMMInitialState(HMMInitialState):
    @property
    def steady_state_distribution(self):
        if self._steady_state_distribution is None:
            markov_part = super(HSMMSteadyState,self).pi_0
            duration_expectations = np.array([d.mean for d in self.model.dur_distns])
            self._steady_state_distribution = markov_part * duration_expectations
            self._steady_state_distribution /= self._steady_state_distribution.sum()
        return self._steady_state_distribution

    def clear_caches(self):
        self._steady_state_distribution = None
