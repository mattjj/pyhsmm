import numpy as np
import scipy.stats as stats
from ..util.stats import sample_discrete

# TODO this just repeats code from multinomial distribution in observations.py.
# I should figure out a better way to organize these things.

class initial_state(object):
    '''
    Initial state distribution class. Not usually of much consequence.
    '''
    def __init__(self,rho=2.,state_dim=10,pi_0=None,**kwargs):
        self.rho = rho
        self.state_dim = state_dim
        self.pi_0 = pi_0
        if self.pi_0 is None:
            self.resample()
        else:
            self.state_dim = len(pi_0)

    def resample(self,init_states=[]):
        # TODO these checks can probably be removed
        assert type(init_states) == type([])

        data = np.zeros(self.state_dim)
        for init_state in init_states:
            data[init_state] += 1
        self.pi_0 = stats.gamma.rvs(self.rho / self.state_dim + data)
        self.pi_0 /= np.sum(self.pi_0)
        assert not np.isnan(self.pi_0).any()

    def rvs(self,size=[]):
        return sample_discrete(self.pi_0,size=size)


class start_zero(object):
    '''
    always start in state 0
    '''
    deterministic = True
    def __init__(self,state_dim,**kwargs):
        self.pi_0 = np.zeros(state_dim)
        self.pi_0[0] = 1.

    def resample(self,init_states=np.array([])):
        pass

    def rvs(self,size=[]):
        return np.zeros(size)
