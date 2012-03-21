from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from warnings import warn

from pyhsmm.internals import states, initial_state, transitions

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

        self.trans_distn = transitions.hsmm_transitions(state_dim=state_dim,**kwargs) \
                if 'transitions' not in kwargs else kwargs['transitions']
        self.init_state_distn = initial_state.initial_state(state_dim=state_dim,**kwargs) \
                if 'initial_state_distn' not in kwargs else kwargs['initial_state_distn']

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
        assert len(set([type(o) for o in self.obs_distns])) == 1, \
                'plot can only be used when all observation distributions are the same'

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

    def loglike(self,data,trunc=None):
        warn('untested')
        T = len(data)
        if trunc is None:
            trunc = T
        # make a temporary states object to make sure no data gets clobbered
        s = states.hsmm_states(T,self.state_dim,self.obs_distns,self.dur_distns,
                self.trans_distn,self.init_state_distn,trunc=trunc)
        s.obs = data
        possible_durations = np.arange(1,trunc + 1,dtype=np.float64)
        aDl = np.zeros((T,self.state_dim))
        aDsl = np.zeros((T,self.state_dim))
        for idx, dur_distn in enumerate(self.dur_distns):
            aDl[:,idx] = dur_distn.log_pmf(possible_durations)
            aDsl[:,idx] = dur_distn.log_sf(possible_durations)

        s.aBl = s.get_aBl(data)
        betal, betastarl = s.messages_backwards(np.log(self.transition_distn.A),aDl,aDsl,trunc)
        return np.logaddexp.reduce(np.log(self.initial_distn.pi_0) + betastarl[0])


# TODO lots of overlap with hsmm here; should put the common elements in a
# parent class
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
        tempstates = states.hmm_states(T,self.state_dim,self.obs_distns,self.trans_distn,
                self.init_state_distn,trunc=self.trunc)
        obs,labels = tempstates.generate(), tempstates.stateseq

        if keep:
            tempstates.added_with_generate = True
            tempstates.data = obs
            self.states_list.append(tempstates)

        return obs, labels

    def plot(self):
        assert len(self.obs_distns) != 0
        assert len(set([type(o) for o in self.obs_distns])) == 1, \
                'plot can only be used when all observation distributions are the same class'

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

    def loglike(self,data):
        warn('untested')
        if len(self.states_list) > 0:
            s = self.states_list[0]
        else:
            # we have to create a temporary one just for its methods, though the
            # details of the actual state sequence are never used
            s = states.hmm_states(len(data),self.state_dim,self.obs_distns,self.trans_distn,
                    self.init_state_distn,trunc=self.trunc,stateseq=np.zeros(len(data),dtype=np.uint8))

        aBl = s.get_aBl(data)
        betal = s.messages_backwards(aBl)
        return np.logaddexp.reduce(np.log(self.initial_distn.pi_0) + betal[0] + aBl[0])


class hmm_sticky(hmm):
    '''
    The HMM class is a convenient wrapper that provides useful constructors and
    packages all the components.
    '''

    def __init__(self,obs_distns,**kwargs):
        warn('the %s class is completely untested!' % type(self))
        if 'transitions' not in kwargs:
            hmm.__init__(self,obs_distns,
                    transitions=transitions.sticky_hdphmm_transitions(state_dim=self.state_dim,**kwargs),
                    **kwargs)
        else:
            hmm.__init__(self,obs_distns,**kwargs)
