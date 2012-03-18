from __future__ import division
import numpy as np
from numpy import newaxis as na

import states, initial_state, transitions

class hsmm(object):
    '''
    The HSMM class is a convenient wrapper that provides useful constructors and
    packages all the components.
    '''

    # TODO remove T

    def __init__(self,T,obs_distns,dur_distns,trunc=None,**kwargs):
        state_dim = len(obs_distns)
        self.state_dim = state_dim
        self.T = T
        self.trunc = trunc

        self.obs_distns = obs_distns
        self.dur_distns = dur_distns

        self.trans_distn = transitions.transitions(state_dim=state_dim,**kwargs) if 'transitions' not in kwargs else kwargs['transitions']
        self.init_state_distn = initial_state.initial_state(state_dim=state_dim,**kwargs) if 'initial_state_distn' not in kwargs else kwargs['initial_state_distn']

        self.states = states.states(T,state_dim,obs_distns,dur_distns,self.trans_distn,self.init_state_distn,trunc=trunc,**kwargs)


    def resample(self,obs):
        # resample obsparams
        for state, distn in enumerate(self.obs_distns):
            distn.resample(obs[self.states.stateseq == state])

        # resample durparams
        for state, distn in enumerate(self.dur_distns):
            distn.resample(self.states.durations[self.states.stateseq_norep == state])

        # resample transitions
        self.trans_distn.resample(self.states.stateseq_norep)

        # resample pi_0
        self.init_state_distn.resample(self.states.stateseq[0])

        # resample states
        self.states.resample(obs)

    def generate(self):
        return self.states.generate(), self.states.stateseq

