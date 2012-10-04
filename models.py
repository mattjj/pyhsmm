from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from warnings import warn

from basic.abstractions import ModelGibbsSampling
from internals import states, initial_state, transitions

# TODO get rid of superfluous kwargs

class HMM(ModelGibbsSampling):
    '''
    The HMM class is a convenient wrapper that provides useful constructors and
    packages all the components.
    '''

    def __init__(self,obs_distns,
            alpha=None,gamma=None,
            alpha_a_0=None,alpha_b_0=None,gamma_a_0=None,gamma_b_0=None,
            **kwargs):

        self.state_dim = len(obs_distns)

        self.obs_distns = obs_distns

        assert ('transitions' in kwargs
                    and isinstance(kwargs['transitions'],transitions.HDPHMMTransitions)) ^ \
                (alpha is not None and gamma is not None) ^ \
                (alpha_a_0 is not None and alpha_b_0 is not None
                        and gamma_a_0 is not None and gamma_b_0 is not None)

        if 'transitions' in kwargs:
            self.trans_distn = kwargs['transitions']
        elif alpha is not None:
            self.trans_distn = transitions.HDPHMMTransitions(state_dim=self.state_dim,
                    alpha=alpha,gamma=gamma,**kwargs)
        else:
            self.trans_distn = transitions.HDPHMMTransitionsConcResampling(
                    state_dim=self.state_dim,
                    alpha_a_0=alpha_a_0,alpha_b_0=alpha_b_0,
                    gamma_a_0=gamma_a_0,gamma_b_0=gamma_b_0)

        self.init_state_distn = initial_state.InitialState(state_dim=self.state_dim,rho=5,**kwargs)\
                if 'initial_state_distn' not in kwargs else kwargs['initial_state_distn']

        self.states_list = []

    def add_data(self,data,stateseq=None):
        self.states_list.append(states.HMMStates(len(data),self.state_dim,self.obs_distns,self.trans_distn,
                self.init_state_distn,data=data,stateseq=stateseq))

    def resample_model(self):
        # resample obsparams
        for state, distn in enumerate(self.obs_distns):
            distn.resample([s.data[s.stateseq == state] for s in self.states_list])

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
        tempstates = states.HMMStates(T,self.state_dim,self.obs_distns,self.trans_distn,
                self.init_state_distn)

        return self._generate(tempstates,keep)

    def _generate(self,tempstates,keep):
        obs,labels = tempstates.generate_obs(), tempstates.stateseq

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
            s = states.HMMStates(len(data),self.state_dim,self.obs_distns,self.trans_distn,
                    self.init_state_distn,stateseq=np.zeros(len(data),dtype=np.uint8))

        aBl = s.get_aBl(data)
        betal = s.messages_backwards(aBl)
        return np.logaddexp.reduce(np.log(self.initial_distn.pi_0) + betal[0] + aBl[0])


class StickyHMM(HMM, ModelGibbsSampling):
    '''
    The HMM class is a convenient wrapper that provides useful constructors and
    packages all the components.
    '''
    def __init__(self,obs_distns,**kwargs):
        warn('the %s class is completely untested!' % type(self))
        if 'transitions' not in kwargs:
            super(StickyHMM,self).__init__(obs_distns,
                    transitions=transitions.StickyHDPHMMTransitions(state_dim=self.state_dim,**kwargs),
                    **kwargs)
        else:
            super(StickyHMM,self).__init__(obs_distns,**kwargs)


class HSMM(HMM, ModelGibbsSampling):
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

    def __init__(self,obs_distns,dur_distns,
            trunc=None,
            alpha=None,gamma=None,
            alpha_a_0=None,alpha_b_0=None,gamma_a_0=None,gamma_b_0=None,
            **kwargs):

        self.trunc = trunc
        self.dur_distns = dur_distns

        assert ('transitions' in kwargs
                    and isinstance(kwargs['transitions'],transitions.HDPHSMMTransitions)) ^ \
                (alpha is not None and gamma is not None) ^ \
                (alpha_a_0 is not None and alpha_b_0 is not None
                        and gamma_a_0 is not None and gamma_b_0 is not None)

        if 'transitions' in kwargs:
            self.trans_distn = kwargs['transitions']
        elif alpha is not None:
            self.trans_distn = transitions.HDPHSMMTransitions(state_dim=self.state_dim,
                    alpha=alpha,gamma=gamma,**kwargs)
        else:
            self.trans_distn = transitions.HDPHSMMTransitionsConcResampling(
                    state_dim=self.state_dim,
                    alpha_a_0=alpha_a_0,alpha_b_0=alpha_b_0,
                    gamma_a_0=gamma_a_0,gamma_b_0=gamma_b_0)

        super(HSMM,self).__init__(obs_distns=obs_distns,transitions=self.trans_distn,**kwargs)

    def add_data(self,data,stateseq=None,censoring=True):
        self.states_list.append(states.HSMMStates(len(data),self.state_dim,self.obs_distns,self.dur_distns,
            self.trans_distn,self.init_state_distn,trunc=self.trunc,data=data,stateseq=stateseq,
            censoring=censoring))

    def resample_model(self):
        # resample durparams
        for state, distn in enumerate(self.dur_distns):
            distn.resample([s.durations[s.stateseq_norep == state] for s in self.states_list])

        # resample everything else an hmm does
        super(HSMM,self).resample_model()

    def generate(self,T,keep=True):
        tempstates = states.HSMMStates(T,self.state_dim,self.obs_distns,self.dur_distns,
                self.trans_distn,self.init_state_distn,trunc=self.trunc)
        return self._generate(tempstates,keep)

    def plot(self,color=None):
        import itertools
        assert len(self.obs_distns) != 0
        assert len(set([type(o) for o in self.obs_distns])) == 1, \
                'plot can only be used when all observation distributions are the same'

        # set up figure and state-color mapping dict
        fig = plt.gcf()
        fig.set_size_inches((6,6))
        num_states = len(reduce(set.union,[set(s.stateseq_norep) for s in self.states_list]))

        # color is ordered by order seen
        idx = 0
        state_colors = {}
        for state in itertools.chain(*[s.stateseq_norep for s in self.states_list]):
            if state not in state_colors:
                state_colors[state] = idx/(num_states-1) if color is None else color
                idx += 1

        cmap = cm.get_cmap()
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
                if state in s.stateseq:
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
        s = states.HSMMStates(T,self.state_dim,self.obs_distns,self.dur_distns,
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


class HSMMPossibleChangepoints(HSMM, ModelGibbsSampling):
    def add_data(self,data,changepoints,stateseq=None):
        self.states_list.append(states.HSMMStatesPossibleChangepoints(changepoints,len(data),self.state_dim,self.obs_distns,self.dur_distns,self.trans_distn,self.init_state_distn,trunc=self.trunc,data=data))

    def generate(self,T,changepoints,keep=True):
        raise NotImplementedError

    def loglike(self,data,trunc=None):
        raise NotImplementedError

