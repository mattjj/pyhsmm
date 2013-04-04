from __future__ import division
import numpy as np
import itertools, collections, operator, random
from matplotlib import pyplot as plt
from matplotlib import cm
from warnings import warn

from basic.abstractions import ModelGibbsSampling, ModelEM
from internals import states, initial_state, transitions

class HMM(ModelGibbsSampling, ModelEM):
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

    def add_data(self,data,stateseq=None,**kwargs):
        self.states_list.append(states.HMMStates(len(data),self.state_dim,self.obs_distns,self.trans_distn,
                self.init_state_distn,data=data,stateseq=stateseq,**kwargs))

    def add_data_parallel(self,data_id):
        from pyhsmm import parallel
        self.add_data(parallel.alldata[data_id])
        self.states_list[-1].data_id = data_id

    def log_likelihood(self,data):
        # TODO avoid this temp states stuff by making messages methods static
        if len(self.states_list) > 0:
            s = self.states_list[0] # any state sequence object will work
        else:
            # we have to create a temporary one just for its methods, though the
            # details of the actual state sequence are never used
            s = states.HMMStates(len(data),self.state_dim,self.obs_distns,self.trans_distn,
                    self.init_state_distn,stateseq=np.zeros(len(data),dtype=np.uint8))

        aBl = s.get_aBl(data)
        betal = s._messages_backwards(aBl)
        return np.logaddexp.reduce(np.log(self.init_state_distn.pi_0) + betal[0] + aBl[0])

    ### generation

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
            tempstates.added_with_generate = True
            tempstates.data = obs
            self.states_list.append(tempstates)

        return obs, labels

    ### Gibbs sampling

    def resample_model(self):
        # resample obsparams
        for state, distn in enumerate(self.obs_distns):
            distn.resample([s.data[s.stateseq == state] for s in self.states_list])

        # resample transitions
        self.trans_distn.resample([s.stateseq for s in self.states_list])

        # resample pi_0
        self.init_state_distn.resample([s.stateseq[:1] for s in self.states_list])

        # resample states
        for s in self.states_list:
            s.resample()

    def resample_model_parallel(self,numtoresample='all'):
        from pyhsmm import parallel
        if numtoresample == 'all':
            numtoresample = len(self.states_list)
        elif numtoresample == 'engines':
            numtoresample = len(parallel.dv)

        ### resample parameters locally
        for state, distn in enumerate(self.obs_distns):
            distn.resample([s.data[s.stateseq == state] for s in self.states_list])
        self.trans_distn.resample([s.stateseq for s in self.states_list])
        self.init_state_distn.resample([s.stateseq[:1] for s in self.states_list])

        ### choose which sequences to resample
        states_to_resample = random.sample(self.states_list,numtoresample)

        ### resample states in parallel
        self._push_self_parallel(states_to_resample)
        self._build_states_parallel(states_to_resample)

        ### purge to prevent memory buildup
        parallel.c.purge_results('all')

    def _push_self_parallel(self,states_to_resample):
        from pyhsmm import parallel
        states_to_restore = [s for s in self.states_list if s not in states_to_resample]
        self.states_list = []
        parallel.dv.push({'global_model':self},block=True)
        self.states_list = states_to_restore

    def _build_states_parallel(self,states_to_resample):
        from pyhsmm import parallel
        raw_stateseq_tuples = parallel.build_states.map([s.data_id for s in states_to_resample])
        for data_id, stateseq in raw_stateseq_tuples:
            self.add_data(
                    data=parallel.alldata[data_id],
                    stateseq=stateseq)
            self.states_list[-1].data_id = data_id
    ### EM

    def EM_step(self):
        assert len(self.states_list) > 0, 'Must have data to run EM'

        ## E step
        for s in self.states_list:
            s.E_step()

        ## M step
        # observation distribution parameters
        for state, distn in enumerate(self.obs_distns):
            distn.max_likelihood([s.data for s in self.states_list],
                    [s.expectations[:,state] for s in self.states_list])

        # initial distribution parameters
        self.init_state_distn.max_likelihood(None,[s.expectations[0] for s in self.states_list])

        # transition parameters (requiring more than just the marginal expectations)
        self.trans_distn.max_likelihood([(s.alphal,s.betal,s.aBl) for s in self.states_list])

        ## for plotting!
        for s in self.states_list:
            s.stateseq = s.expectations.argmax(1)

    def num_parameters(self):
        return sum(o.num_parameters() for o in self.obs_distns) + self.state_dim**2

    def BIC(self):
        # NOTE: in principle this method computes the BIC only after finding the
        # maximum likelihood parameters (or, of course, an EM fixed-point as an
        # approximation!)
        assert len(self.states_list) > 0, 'Must have data to get BIC'
        return -2*sum(self.log_likelihood(s.data).sum() for s in self.states_list) + \
                    self.num_parameters() * np.log(sum(s.data.shape[0] for s in self.states_list))

    ### plotting

    def _get_used_states(self,states_objs=None):
        if states_objs is None:
            states_objs = self.states_list
        canonical_ids = collections.defaultdict(itertools.count().next)
        for s in states_objs:
            for state in s.stateseq:
                canonical_ids[state]
        return map(operator.itemgetter(0),sorted(canonical_ids.items(),key=operator.itemgetter(1)))

    def _get_colors(self):
        states = self._get_used_states()
        numstates = len(states)
        return dict(zip(states,np.linspace(0,1,numstates,endpoint=True)))

    def plot_observations(self,colors=None,states_objs=None):
        if colors is None:
            colors = self._get_colors()
        if states_objs is None:
            states_objs = self.states_list

        cmap = cm.get_cmap()
        used_states = self._get_used_states(states_objs)
        for state,o in enumerate(self.obs_distns):
            if state in used_states:
                o.plot(
                        color=cmap(colors[state]),
                        data=[s.data[s.stateseq == state] if s.data is not None else None
                            for s in states_objs],
                        label='%d' % state)
        plt.title('Observation Distributions')

    def plot(self,color=None,legend=True):
        plt.gcf() #.set_size_inches((10,10))
        colors = self._get_colors()

        num_subfig_cols = len(self.states_list)
        for subfig_idx,s in enumerate(self.states_list):
            plt.subplot(2,num_subfig_cols,1+subfig_idx)
            self.plot_observations(colors=colors,states_objs=[s])

            plt.subplot(2,num_subfig_cols,1+num_subfig_cols+subfig_idx)
            s.plot(colors_dict=colors)


class StickyHMM(HMM, ModelGibbsSampling):
    '''
    The HMM class is a convenient wrapper that provides useful constructors and
    packages all the components.
    '''
    def __init__(self,
            obs_distns,
            trans_distn=None,
            kappa=None,alpha=None,gamma=None,
            rho_a_0=None,rho_b_0=None,alphakappa_a_0=None,alphakappa_b_0=None,gamma_a_0=None,gamma_b_0=None,
            **kwargs):

        assert (trans_distn is not None) ^ \
                (kappa is not None and alpha is not None and gamma is not None) ^ \
                (rho_a_0 is not None and rho_b_0 is not None
                        and alphakappa_a_0 is not None and alphakappa_b_0 is not None
                        and gamma_a_0 is not None and gamma_b_0 is not None)
        if trans_distn is not None:
            self.trans_distn = trans_distn
        elif kappa is not None:
            self.trans_distn = transitions.StickyHDPHMMTransitions(
                    state_dim=len(obs_distns),
                    alpha=alpha,gamma=gamma,kappa=kappa)
        else:
            self.trans_distn = transitions.StickyHDPHMMTransitionsConcResampling(
                    state_dim=len(obs_distns),
                    rho_a_0=rho_a_0,rho_b_0=rho_b_0,
                    alphakappa_a_0=alphakappa_a_0,alphakappa_b_0=alphakappa_b_0,
                    gamma_a_0=gamma_a_0,gamma_b_0=gamma_b_0)

        super(StickyHMM,self).__init__(obs_distns,trans_distn=self.trans_distn,**kwargs)

    def EM_step(self):
        raise NotImplementedError, "Can't run EM on a StickyHMM"


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

    def add_data(self,data,stateseq=None,censoring=True,**kwargs):
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

    ### parallel sampling
    def add_data_parallel(self,data_id):
        from pyhsmm import parallel
        self.add_data(parallel.alldata[data_id])
        self.states_list[-1].data_id = data_id

    def resample_model_parallel(self,numtoresample='all'):
        from pyhsmm import parallel
        if numtoresample == 'all':
            numtoresample = len(self.states_list)
        elif numtoresample == 'engines':
            numtoresample = len(parallel.dv)

        ### resample parameters locally
        self.trans_distn.resample([s.stateseq for s in self.states_list])
        self.init_state_distn.resample([s.stateseq[:1] for s in self.states_list])
        for state, (o,d) in enumerate(zip(self.obs_distns,self.dur_distns)):
            d.resample([s.durations[s.stateseq_norep == state] for s in self.states_list])
            o.resample([s.data[s.stateseq == state] for s in self.states_list])

        ### choose which sequences to resample
        states_to_resample = random.sample(self.states_list,numtoresample)

        ### resample states in parallel
        self._push_self_parallel(states_to_resample)
        self._build_states_parallel(states_to_resample)

        ### purge to prevent memory buildup
        parallel.c.purge_results('all')

    ### plotting

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

    def plot_summary(self,color=None):
        # if there are too many state sequences in states_list, make an
        # alternative plot
        raise NotImplementedError

    def log_likelihood(self,data,trunc=None):
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
        betal, betastarl = s._messages_backwards(np.log(self.transition_distn.A),aDl,aDsl,trunc)
        return np.logaddexp.reduce(np.log(self.initial_distn.pi_0) + betastarl[0])

    def EM_step(self):
        raise NotImplementedError # TODO


class HSMMPossibleChangepoints(HSMM, ModelGibbsSampling):
    def add_data(self,data,changepoints,**kwargs):
        self.states_list.append(
                states.HSMMStatesPossibleChangepoints(
                    changepoints, len(data), self.state_dim, self.obs_distns,
                    self.dur_distns, self.trans_distn, self.init_state_distn,
                    trunc=self.trunc, data=data,
                    **kwargs))

    def add_data_parallel(self,data_id,**kwargs):
        from pyhsmm import parallel
        self.add_data(parallel.alldata[data_id],parallel.allchangepoints[data_id],**kwargs)
        self.states_list[-1].data_id = data_id

    def _build_states_parallel(self,states_to_resample):
        from pyhsmm import parallel
        raw_stateseq_tuples = parallel.build_states_changepoints.map([s.data_id for s in states_to_resample])
        for data_id, stateseq in raw_stateseq_tuples:
            self.add_data(
                    data=parallel.alldata[data_id],
                    changepoints=parallel.allchangepoints[data_id],
                    stateseq=stateseq)
            self.states_list[-1].data_id = data_id

    def generate(self,T,changepoints,keep=True):
        raise NotImplementedError

    def log_likelihood(self,data,trunc=None):
        raise NotImplementedError

