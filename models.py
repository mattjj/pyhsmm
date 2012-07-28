from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from warnings import warn

from pyhsmm.internals import states, initial_state, transitions

class hmm(object):
    '''
    The HMM class is a convenient wrapper that provides useful constructors and
    packages all the components.
    '''

    def __init__(self,obs_distns,alpha=None,gamma=None,**kwargs):
        self.state_dim = len(obs_distns)

        self.obs_distns = obs_distns

        if alpha is None or gamma is None:
            assert 'transitions' in kwargs, 'must specify transition distribution to initialize %s without concentration parameters' % type(self)

        self.trans_distn = transitions.hdphmm_transitions(state_dim=self.state_dim,alpha=alpha,gamma=gamma,**kwargs)\
                if 'transitions' not in kwargs else kwargs['transitions']

        self.init_state_distn = initial_state.initial_state(state_dim=self.state_dim,**kwargs)\
                if 'initial_state_distn' not in kwargs else kwargs['initial_state_distn']

        self.states_list = []

    def add_data(self,data,stateseq=None):
        self.states_list.append(states.hmm_states(len(data),self.state_dim,self.obs_distns,self.trans_distn,
                self.init_state_distn,data=data,stateseq=stateseq))

    def resample(self):
        # resample obsparams
        for state, distn in enumerate(self.obs_distns):
            # TODO make obs distns take lols
            all_obs = [s.data[s.stateseq == state] for s in self.states_list]
            if len(all_obs) > 0:
                distn.resample(np.concatenate(all_obs))
            else:
                distn.resample()

        # resample transitions
        self.trans_distn.resample([s.stateseq for s in self.states_list])

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
        tempstates = states.hmm_states(T,self.state_dim,self.obs_distns,self.trans_distn,
                self.init_state_distn)

        return self._generate(tempstates,keep)

    def _generate(self,tempstates,keep):
        # TODO probably already generated when tempstates was added, so this
        # call to generate only needs to be a call to generate_obs
        obs,labels = tempstates.generate(), tempstates.stateseq

        if keep:
            tempstates.added_with_generate = True # I love Python
            tempstates.data = obs
            self.states_list.append(tempstates)

        return obs, labels

    def plot(self,color=None):
        assert len(self.obs_distns) != 0
        assert len(set([type(o) for o in self.obs_distns])) == 1, \
                'plot can only be used when all observation distributions are the same class'

        # set up figure and state-color mapping dict
        fig = plt.gcf()
        fig.set_size_inches((6,6))
        state_colors = {}
        cmap = cm.get_cmap()
        used_states = reduce(set.union,[set(s.stateseq) for s in self.states_list])
        num_states = len(used_states)
        num_subfig_cols = len(self.states_list)
        for idx,state in enumerate(used_states):
            state_colors[state] = idx/(num_states-1) if color is None else color

        for subfig_idx,s in enumerate(self.states_list):

            # plot the current observation distributions (and obs, if given)
            plt.subplot(2,num_subfig_cols,1+subfig_idx)
            self.obs_distns[0]._plot_setup(self.obs_distns)
            for state,o in enumerate(self.obs_distns):
                if state in s.stateseq:
                    o.plot(color=cmap(state_colors[state]),
                            data=s.data[s.stateseq == state] if s.data is not None else None)
            plt.title('Observation Distributions')

            # plot the state sequence
            plt.subplot(2,num_subfig_cols,1+num_subfig_cols+subfig_idx)
            s.plot(colors_dict=state_colors)
            plt.title('State Sequence')

    def loglike(self,data):
        warn('untested')
        if len(self.states_list) > 0:
            s = self.states_list[0] # any state sequence object will work
        else:
            # we have to create a temporary one just for its methods, though the
            # details of the actual state sequence are never used
            s = states.hmm_states(len(data),self.state_dim,self.obs_distns,self.trans_distn,
                    self.init_state_distn,stateseq=np.zeros(len(data),dtype=np.uint8))

        aBl = s.get_aBl(data)
        betal = s.messages_backwards(aBl)
        return np.logaddexp.reduce(np.log(self.initial_distn.pi_0) + betal[0] + aBl[0])


class hsmm(hmm):
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

    def __init__(self,alpha,gamma,obs_distns,dur_distns,trunc=None,**kwargs):
        self.trunc = trunc # duplicated with hmm behavior at the moment
        self.dur_distns = dur_distns
        if 'transitions' in kwargs:
            trans = kwargs['transitions']
            del kwargs['transitions']
            assert type(trans) == transitions.hsmm_transitions
        else:
            trans = transitions.hsmm_transitions(alpha=alpha,gamma=gamma,state_dim=len(obs_distns))
        super(hsmm,self).__init__(alpha=alpha,gamma=gamma,obs_distns=obs_distns,transitions=trans,**kwargs)

    def add_data(self,data):
        self.states_list.append(states.hsmm_states(len(data),self.state_dim,self.obs_distns,self.dur_distns,
            self.trans_distn,self.init_state_distn,trunc=self.trunc,data=data))

    def resample(self):
        # resample durparams
        for state, distn in enumerate(self.dur_distns):
            all_durs = [s.durations[s.stateseq_norep == state] for s in self.states_list]
            if len(all_durs) > 0:
                distn.resample(np.concatenate(all_durs))
            else:
                distn.resample()

        # resample everything else an hmm does
        super(hsmm,self).resample()

    def generate(self,T,keep=True):
        tempstates = states.hsmm_states(T,self.state_dim,self.obs_distns,self.dur_distns,
                self.trans_distn,self.init_state_distn,trunc=self.trunc)
        return self._generate(tempstates,keep)

    def plot(self,color=None):
        assert len(self.obs_distns) != 0
        assert len(set([type(o) for o in self.obs_distns])) == 1, \
                'plot can only be used when all observation distributions are the same'

        # set up figure and state-color mapping dict
        fig = plt.gcf()
        fig.set_size_inches((6,6))
        state_colors = {}
        cmap = cm.get_cmap()
        used_states = reduce(set.union,[set(s.stateseq_norep) for s in self.states_list])
        num_states = len(used_states)
        for idx,state in enumerate(used_states):
            state_colors[state] = idx/(num_states-1) if color is None else color

        num_subfig_cols = len(self.states_list)
        for subfig_idx,s in enumerate(self.states_list):

            # plot the current observation distributions (and obs, if given)
            plt.subplot(3,num_subfig_cols,1+subfig_idx)
            self.obs_distns[0]._plot_setup(self.obs_distns)
            for state,o in enumerate(self.obs_distns):
                if state in s.stateseq_norep:
                    o.plot(color=cmap(state_colors[state]),
                            data=s.data[s.stateseq == state] if s.data is not None else None)
            plt.title('Observation Distributions')

            # plot the state sequence
            plt.subplot(3,num_subfig_cols,1+num_subfig_cols+subfig_idx)
            s.plot(colors_dict=state_colors)
            plt.title('State Sequence')

            # plot the current duration distributions
            plt.subplot(3,num_subfig_cols,1+2*num_subfig_cols+subfig_idx)
            for state,d in enumerate(self.dur_distns):
                if state in s.stateseq_norep:
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
