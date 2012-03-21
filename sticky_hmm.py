from __future__ import division
from warnings import warn

from .hmm import hmm
from .hsmm_internals import states, transitions

# TODO this is UNTESTED

class sticky_hmm(hmm):
    '''
    The HMM class is a convenient wrapper that provides useful constructors and
    packages all the components.
    '''

    def __init__(self,obs_distns,**kwargs):
        warn('the %s class is completely untested!' % type(self))
        if 'transitions' not in kwargs:
            hmm.__init__(self,obs_distns,transitions=transitions.sticky_hdphmm_transitions(state_dim=self.state_dim,**kwargs),**kwargs)
        else:
            hmm.__init__(self,obs_distns,**kwargs)

def use_eigen():
    states.use_eigen()
