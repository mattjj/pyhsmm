from __future__ import division
import numpy as np
import itertools, collections, operator
from matplotlib import pyplot as plt
from matplotlib import cm
from warnings import warn

from basic.abstractions import ModelGibbsSampling
from internals import states, initial_state, transitions

class HMM(ModelGibbsSampling):
    '''
    The HMM class is a convenient wrapper that provides useful constructors and
    packages all the components.
    '''

    def __init__(self,
            obs_distns,
            trans_distn=None,
            alpha=None,gamma=None,
            alpha_a_0=None,alpha_b_0=None,gamma_a_0=None,gamma_b_0=None,
            init_state_distn=None,
            init_state_concentration=None):

        self.state_dim = len(obs_distns)
        self.obs_distns = obs_distns
        self.states_list = []

        assert (trans_distn is not None) ^ \
                (alpha is not None and gamma is not None) ^ \
                (alpha_a_0 is not None and alpha_b_0 is not None
                        and gamma_a_0 is not None and gamma_b_0 is not None)
        if trans_distn is not None:
            self.trans_distn = trans_distn
        elif alpha is not None:
            self.trans_distn = transitions.HDPHMMTransitions(
                    state_dim=self.state_dim,
                    alpha=alpha,gamma=gamma)
        else:
            self.trans_distn = transitions.HDPHMMTransitionsConcResampling(
                    state_dim=self.state_dim,
                    alpha_a_0=alpha_a_0,alpha_b_0=alpha_b_0,
                    gamma_a_0=gamma_a_0,gamma_b_0=gamma_b_0)

        assert (init_state_distn is not None) ^ \
                (init_state_concentration is not None)

        if init_state_distn is not None:
            self.init_state_distn = init_state_distn
        else:
            self.init_state_distn = initial_state.InitialState(
                    state_dim=self.state_dim,
                    rho=init_state_concentration)

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

    def _get_used_states(self,states_objs=None):
        if states_objs is None:
            states_objs = self.states_list
        canonical_ids = collections.defaultdict(itertools.count().next)
        for s in self.states_list:
            for state in s.stateseq:
                canonical_ids[state]
        return map(operator.itemgetter(0),sorted(canonical_ids.items(),key=operator.itemgetter(1)))

    def _get_colors(self):
        states = self._get_used_states()
        numstates = len(states)
        return dict(zip(states,np.linspace(0,1,numstates,endpoint=True)))

    def plot_observations(self,colors=None,states_objs=None):
        self.obs_distns[0]._plot_setup(self.obs_distns)
        if colors is None:
            colors = self._get_colors()
        if states_objs is None:
            states_objs = self.states_list

        cmap = cm.get_cmap()
        used_states = self._get_used_states(states_objs)
        for state,o in enumerate(self.obs_distns):
            if state in used_states:
                o.plot(color=cmap(colors[state]),
                        data=[s.data[s.stateseq == state] if s.data is not None else None
                            for s in states_objs])
        plt.title('Observation Distributions')

    def plot(self,color=None):
        plt.gcf() #.set_size_inches((10,10))
        colors = self._get_colors()

        num_subfig_cols = len(self.states_list)
        for subfig_idx,s in enumerate(self.states_list):
            plt.subplot(2,num_subfig_cols,1+subfig_idx)
            self.plot_observations(colors=colors,states_objs=[s])

            plt.subplot(2,num_subfig_cols,1+num_subfig_cols+subfig_idx)
            s.plot(colors_dict=colors)

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
    def __init__(self,
            obs_distns,
            trans_distn=None,
            kappa=None,alpha=None,gamma=None,
            kappa_a_0=None,kappa_b_0=None,alpha_a_0=None,alpha_b_0=None,gamma_a_0=None,gamma_b_0=None,
            **kwargs):

        assert (trans_distn is not None) ^ \
                (kappa is not None and alpha is not None and gamma is not None) ^ \
                (kappa_a_0 is not None and kappa_b_0 is not None
                        and alpha_a_0 is not None and alpha_b_0 is not None
                        and gamma_a_0 is not None and gamma_b_0 is not None)
        if trans_distn is not None:
            self.trans_distn = trans_distn
        elif kappa is not None:
            self.trans_distn = transitions.StickyHDPHMMTransitions(
                    state_dim=self.state_dim,
                    alpha=alpha,gamma=gamma)
        else:
            self.trans_distn = transitions.StickyHDPHMMTransitionsConcResampling(
                    state_dim=self.state_dim,
                    alpha_a_0=alpha_a_0,alpha_b_0=alpha_b_0,
                    gamma_a_0=gamma_a_0,gamma_b_0=gamma_b_0)

        super(StickyHMM,self).__init__(obs_distns,trans_distn=self.trans_distn,**kwargs)

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

    def __init__(self,
            obs_distns,dur_distns,
            trunc=None,
            trans_distn=None,
            alpha=None,gamma=None,
            alpha_a_0=None,alpha_b_0=None,gamma_a_0=None,gamma_b_0=None,
            **kwargs):

        self.state_dim = len(obs_distns)
        self.trunc = trunc
        self.dur_distns = dur_distns

        assert (trans_distn is not None) ^ \
                (alpha is not None and gamma is not None) ^ \
                (alpha_a_0 is not None and alpha_b_0 is not None
                        and gamma_a_0 is not None and gamma_b_0 is not None)
        if trans_distn is not None:
            self.trans_distn = trans_distn
        elif alpha is not None:
            self.trans_distn = transitions.HDPHSMMTransitions(
                    state_dim=self.state_dim,
                    alpha=alpha,gamma=gamma)
        else:
            self.trans_distn = transitions.HDPHSMMTransitionsConcResampling(
                    state_dim=self.state_dim,
                    alpha_a_0=alpha_a_0,alpha_b_0=alpha_b_0,
                    gamma_a_0=gamma_a_0,gamma_b_0=gamma_b_0)

        super(HSMM,self).__init__(obs_distns=obs_distns,trans_distn=self.trans_distn,**kwargs)

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

    ### plotting stuff below here

    def plot_durations(self,colors=None,states_objs=None):
        if colors is None:
            colors = self._get_colors()
        if states_objs is None:
            states_objs = self.states_list

        cmap = cm.get_cmap()
        used_states = self._get_used_states(states_objs)
        for state,d in enumerate(self.dur_distns):
            if state in used_states:
                d.plot(color=cmap(colors[state]),
                        data=[s.durations[s.stateseq_norep == state]
                            for s in states_objs])
        plt.title('Durations')

    def plot(self,color=None):
        plt.gcf() #.set_size_inches((10,10))
        colors = self._get_colors()

        num_subfig_cols = len(self.states_list)
        for subfig_idx,s in enumerate(self.states_list):
            plt.subplot(3,num_subfig_cols,1+subfig_idx)
            self.plot_observations(colors=colors,states_objs=[s])

            plt.subplot(3,num_subfig_cols,1+num_subfig_cols+subfig_idx)
            s.plot(colors_dict=colors)

            plt.subplot(3,num_subfig_cols,1+2*num_subfig_cols+subfig_idx)
            self.plot_durations(colors=colors,states_objs=[s])

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

