from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from hsmm_internals import states, initial_state, transitions

# TODO there's a lot of overlapping code with hsmm.py; I should merge the common
# elements into a base class

class hmm(object):
    '''
    The HMM class is a convenient wrapper that provides useful constructors and
    packages all the components.
    '''

    def __init__(self,obs_distns,**kwargs):
        self.state_dim = len(obs_distns)

        self.obs_distns = obs_distns

        self.trans_distn = transitions.hmm_transitions(state_dim=self.state_dim,**kwargs)\
                if 'transitions' not in kwargs else kwargs['transitions']

        self.init_state_distn = initial_state.initial_state(state_dim=self.state_dim,**kwargs)\
                if 'initial_state_distn' not in kwargs else kwargs['initial_state_distn']

        self.trunc = None if 'trunc' not in kwargs else kwargs['trunc']

        self.states_list = []

    def add_data(self,data):
        self.states_list.append(states.hmm_states(len(data),self.state_dim,self.obs_distns,self.trans_distn,
                self.init_state_distn,data=data,trunc=self.trunc))

    def resample(self,niter=1):
        for itr in range(niter):
            # resample obsparams
            for state, distn in enumerate(self.obs_distns):
                distn.resample(np.concatenate([s.data[s.stateseq == state] for s in self.states_list]))

            # resample transitions
            self.trans_distn.resample([s.stateseq for s in self.states_list])

            # resample pi_0
            self.init_state_distn.resample([s.stateseq[0] for s in self.states_list])

            # resample states
            for s in self.states_list:
                s.resample()

    def generate(self,T,keep=True):
        tempstates = states.hmm_states(T,self.state_dim,self.obs_distns,self.trans_distn,self.init_state_distn,trunc=self.trunc)
        obs,labels = tempstates.generate(), tempstates.stateseq

        if keep:
            tempstates.added_with_generate = True
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
            used_states = set(s.stateseq)
            num_states = len(used_states)
            for idx,state in enumerate(set(s.stateseq)):
                state_colors[state] = idx/(num_states-1)

            # plot the current observation distributions (and obs, if given)
            plt.subplot(2,len(self.states_list),1+subfig_idx)
            self.obs_distns[0]._plot_setup(self.obs_distns)
            for state,o in enumerate(self.obs_distns):
                if state in used_states:
                    o.plot(color=cmap(state_colors[state]),
                            data=s.data[s.stateseq == state] if s.data is not None else None)
            plt.title('Observation Distributions')

            # plot the state sequence
            plt.subplot(2,len(self.states_list),1+len(self.states_list)+subfig_idx)
            s.plot(colors_dict=state_colors)
            plt.title('State Sequence')

def use_eigen():
    states.use_eigen()
