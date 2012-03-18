from __future__ import division
from matplotlib import pyplot as plt
from matplotlib import cm

from hsmm_internals import states, initial_state, transitions

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

    def plot(self,obs=None):
        assert len(self.obs_distns) != 0
        assert len(set([type(o) for o in self.obs_distns])) == 1, 'plot can only be used when all observation distributions are the same'

        # set up figure and state-color mapping dict
        plt.figure()
        state_colors = {}
        cmap = cm.get_cmap()
        used_states = set(self.states.stateseq_norep)
        num_states = len(used_states)
        for idx,state in enumerate(set(self.states.stateseq)):
            state_colors[state] = idx/(num_states-1)

        # plot the current observation distributions (and obs, if given)
        plt.subplot(3,1,1)
        self.obs_distns[0]._plot_setup(self.obs_distns)
        for state,o in enumerate(self.obs_distns):
            if state in used_states:
                o.plot(color=cmap(state_colors[state]),
                        data=obs[self.states.stateseq == state] if obs is not None else None)
        plt.title('Observation Distributions')

        # plot the state sequence
        plt.subplot(3,1,2)
        self.states.plot(colors_dict=state_colors)
        plt.title('State Sequence')

        # plot the current duration distributions
        plt.subplot(3,1,3)
        for state,d in enumerate(self.dur_distns):
            if state in used_states:
                d.plot(color=cmap(state_colors[state]),
                        data=self.states.durations[self.states.stateseq_norep == state])
        plt.title('Durations')

        # TODO add a figure legend

