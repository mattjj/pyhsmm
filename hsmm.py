from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from hsmm_internals import states, initial_state, transitions

class hsmm(object):
    '''
    The HSMM class is a wrapper to package all the pieces of an HSMM:
        * HSMM internals, including distribution objects for
            - states
            - transitions
            - initial state
        * the main distributions that define the HSMM:
            - observations
            - durations
    When an HSMM is instantiated, it is a ``prior'' model object. Observation
    sequences can be added via the add_data(data_seq) method, making it a
    ``posterior'' model object and then the latent components (including all
    state sequences and parameters) can be resampled by calling the resample()
    method.
    '''

    def __init__(self,obs_distns,dur_distns,trunc=None,**kwargs):
        state_dim = len(obs_distns)
        self.state_dim = state_dim
        self.trunc = trunc
        self.states_list = []

        self.obs_distns = obs_distns
        self.dur_distns = dur_distns

        self.trans_distn = transitions.hsmm_transitions(state_dim=state_dim,**kwargs) if 'transitions' not in kwargs else kwargs['transitions']
        self.init_state_distn = initial_state.initial_state(state_dim=state_dim,**kwargs) if 'initial_state_distn' not in kwargs else kwargs['initial_state_distn']

    def add_data(self,data):
        self.states_list.append(states.hsmm_states(len(data),self.state_dim,self.obs_distns,self.dur_distns,
            self.trans_distn,self.init_state_distn,trunc=self.trunc,data=data))

    def resample(self):
        # resample obsparams
        for state, distn in enumerate(self.obs_distns):
            distn.resample(np.concatenate([s.data[s.stateseq == state] for s in self.states_list]))

        # resample durparams
        for state, distn in enumerate(self.dur_distns):
            distn.resample(np.concatenate([s.durations[s.stateseq_norep == state] for s in self.states_list]))

        # resample transitions
        self.trans_distn.resample([s.stateseq_norep for s in self.states_list])

        # resample pi_0
        self.init_state_distn.resample([s.stateseq[0] for s in self.states_list])

        # resample states
        for s in self.states_list:
            s.resample()

    def generate(self,T,keep=True):
        '''
        Generates a forward sample using the current values of all parameters.
        Returns an observation sequence and a state sequence of length T.

        If keep is True, the states object created is appended to the
        states_list. This is mostly useful for generating synthetic data and
        keeping it around in an HSMM object as the latent truth.

        To construct a posterior sample, one must call both the add_data and
        resample methods first. Then, calling generate() will produce a sample
        from the posterior (as long as the Gibbs sampling has converged). In
        these cases, the keep argument should be False.
        '''
        tempstates = states.hsmm_states(T,self.state_dim,self.obs_distns,self.dur_distns,
                self.trans_distn,self.init_state_distn,trunc=self.trunc)
        obs, labels = tempstates.generate(), tempstates.stateseq

        if keep:
            tempstates.added_with_generate = True # I love Python
            tempstates.data = obs
            self.states_list.append(tempstates)

        return obs, labels

    def plot(self):
        assert len(self.obs_distns) != 0
        assert len(set([type(o) for o in self.obs_distns])) == 1, 'plot can only be used when all observation distributions are the same'

        fig = plt.figure()
        for subfig_idx,s in enumerate(self.states_list):
            # set up figure and state-color mapping dict
            fig.set_size_inches((6,6))
            state_colors = {}
            cmap = cm.get_cmap()
            used_states = set(s.stateseq_norep)
            num_states = len(used_states)
            for idx,state in enumerate(set(s.stateseq)):
                state_colors[state] = idx/(num_states-1)

            # plot the current observation distributions (and obs, if given)
            plt.subplot(3,len(self.states_list),1+subfig_idx)
            self.obs_distns[0]._plot_setup(self.obs_distns)
            for state,o in enumerate(self.obs_distns):
                if state in used_states:
                    o.plot(color=cmap(state_colors[state]),
                            data=s.data[s.stateseq == state] if s.data is not None else None)
            plt.title('Observation Distributions')

            # plot the state sequence
            plt.subplot(3,len(self.states_list),1+len(self.states_list)+subfig_idx)
            s.plot(colors_dict=state_colors)
            plt.title('State Sequence')

            # plot the current duration distributions
            plt.subplot(3,len(self.states_list),1+2*len(self.states_list)+subfig_idx)
            for state,d in enumerate(self.dur_distns):
                if state in used_states:
                    d.plot(color=cmap(state_colors[state]),
                            data=s.durations[s.stateseq_norep == state])
            plt.xlim((0,s.durations.max()*1.1))
            plt.title('Durations')

            # TODO add a figure legend

def use_eigen():
    states.use_eigen()
