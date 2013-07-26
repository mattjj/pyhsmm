from __future__ import division
import numpy as np
import itertools, collections, operator, random, abc, copy
from matplotlib import pyplot as plt
from matplotlib import cm

from basic.abstractions import ModelGibbsSampling, ModelEM, ModelMAPEM
import basic.distributions
from internals import states, initial_state, transitions
import util.general

# TODO TODO treat right censoring like left censoring (and pass as explicit
# truncation to tudration resample method)
# TODO think about factoring out base classes for HMMs and HSMMs
# TODO maybe states classes should handle log_likelihood and predictive
# likelihood methods
# TODO generate_obs should be here, not in states.py

class HMM(ModelGibbsSampling, ModelEM, ModelMAPEM):
    _states_class = states.HMMStatesPython
    _trans_class = transitions.HDPHMMTransitions
    _trans_class_conc_class = transitions.HDPHMMTransitionsConcResampling
    _init_steady_state_class = initial_state.SteadyState

    def __init__(self,
            obs_distns,
            trans_distn=None,
            alpha=None,gamma=None,
            alpha_a_0=None,alpha_b_0=None,gamma_a_0=None,gamma_b_0=None,
            init_state_distn=None,init_state_concentration=None):

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
            self.trans_distn = self._trans_class(
                    state_dim=self.state_dim,
                    alpha=alpha,gamma=gamma)
        else:
            self.trans_distn = self._trans_class_conc_class(
                    state_dim=self.state_dim,
                    alpha_a_0=alpha_a_0,alpha_b_0=alpha_b_0,
                    gamma_a_0=gamma_a_0,gamma_b_0=gamma_b_0)

        if init_state_distn is not None:
            self.init_state_distn = init_state_distn
        elif init_state_concentration is not None:
            self.init_state_distn = initial_state.InitialState(
                    state_dim=self.state_dim,
                    rho=init_state_concentration)
        else:
            # in this case, the initial state distribution is just the
            # steady-state of the transition matrix
            self.init_state_distn = self._init_steady_state_class(self)

    @property
    def stateseqs(self):
        'a convenient reference to the state sequence arrays'
        return [s.stateseq for s in self.states_list]

    def add_data(self,data,stateseq=None,**kwargs):
        self.states_list.append(self._states_class(model=self,data=np.asarray(data),
            stateseq=stateseq,**kwargs))

    def log_likelihood(self,data=None):
        if data is not None:
            s = self._states_class(model=self,data=np.asarray(data),
                    stateseq=np.zeros(len(data))) # placeholder
            betal = s.messages_backwards()
            return np.logaddexp.reduce(np.log(self.init_state_distn.pi_0) + betal[0] + s.aBl[0])
        else:
            betastarls = np.vstack([s.messages_backwards()[0] + s.aBl[0]
                for s in self.states_list])
            return np.logaddexp.reduce(np.log(self.init_state_distn.pi_0)
                                            + betastarls,axis=1).sum()

    def predictive_likelihoods(self,test_data,forecast_horizons):
        s = self._states_class(model=self,data=np.asarray(test_data),
                stateseq=np.zeros(test_data.shape[0])) # placeholder
        alphal = s.messages_forwards()

        cmaxes = alphal.max(axis=1)
        scaled_alphal = np.exp(alphal - cmaxes[:,None])
        prev_k = 0

        outs = []
        for k in forecast_horizons:
            step = k - prev_k

            cmaxes = cmaxes[:-step]
            scaled_alphal = scaled_alphal[:-step].dot(np.linalg.matrix_power(s.trans_matrix,step))

            future_likelihoods = np.logaddexp.reduce(
                    np.log(scaled_alphal) + cmaxes[:,None] + s.aBl[k:],axis=1)
            past_likelihoods = np.logaddexp.reduce(alphal[:-k],axis=1)
            outs.append(future_likelihoods - past_likelihoods)

            prev_k = k

        return outs

    def block_predictive_likelihoods(self,test_data,blocklens):
        s = self._states_class(model=self,data=np.asarray(test_data),
                stateseq=np.zeros(test_data.shape[0])) # placeholder
        alphal = s.messages_forwards()

        outs = []
        for k in blocklens:
            outs.append(np.logaddexp.reduce(alphal[k:],axis=1)
                    - np.logaddexp.reduce(alphal[:-k],axis=1))

        return outs

    ### generation

    def generate(self,T,keep=True):
        tempstates = self._states_class(self,T=T,initialize_from_prior=True)
        return self._generate(tempstates,keep)

    def _generate(self,tempstates,keep):
        obs,labels = tempstates.generate_obs(), tempstates.stateseq

        if keep:
            tempstates.added_with_generate = True
            tempstates.data = obs
            self.states_list.append(tempstates)

        return obs, labels

    ### caching

    def _clear_caches(self):
        for s in self.states_list:
            s.clear_caches()
        if hasattr(self.init_state_distn,'clear_caches'):
            self.init_state_distn.clear_caches()

    def __getstate__(self):
        self._clear_caches()
        return self.__dict__.copy()

    ### Gibbs sampling

    def resample_model(self,temp=None):
        self.resample_obs_distns()
        self.resample_trans_distn()
        self.resample_init_state_distn()
        self.resample_states(temp=temp)

    def resample_obs_distns(self):
        for state, distn in enumerate(self.obs_distns):
            distn.resample([s.data[s.stateseq == state] for s in self.states_list])
        self._clear_caches()

    def resample_trans_distn(self):
        self.trans_distn.resample([s.stateseq for s in self.states_list])
        self._clear_caches()

    def resample_init_state_distn(self):
        self.init_state_distn.resample([s.stateseq[:1] for s in self.states_list])
        self._clear_caches()

    def resample_states(self,temp=None):
        for s in self.states_list:
            s.resample(temp=temp)

    def copy_sample(self):
        new = copy.copy(self)
        new.obs_distns = [o.copy_sample() for o in self.obs_distns]
        new.trans_distn = self.trans_distn.copy_sample()
        new.init_state_distn = self.init_state_distn.copy_sample()
        new.states_list = [s.copy_sample(new) for s in self.states_list]
        return new

    ### parallel

    def add_data_parallel(self,data_id,**kwargs):
        from pyhsmm import parallel
        self.add_data(data=parallel.alldata[data_id],**kwargs)
        self.states_list[-1].data_id = data_id

    def resample_model_parallel(self,numtoresample='all',temp=None):
        from pyhsmm import parallel
        if numtoresample == 'all':
            numtoresample = len(self.states_list)
        elif numtoresample == 'engines':
            numtoresample = len(parallel.dv)

        ### resample parameters locally
        self.resample_obs_distns()
        self.resample_trans_distn()
        self.resample_init_state_distn()

        ### choose which sequences to resample
        states_to_resample = random.sample(self.states_list,numtoresample)

        ### resample states in parallel
        self._push_self_parallel(states_to_resample)
        self._build_states_parallel(states_to_resample,temp=temp)

        ### purge to prevent memory buildup
        parallel.c.purge_results('all')

    def _push_self_parallel(self,states_to_resample):
        from pyhsmm import parallel
        states_to_restore = [s for s in self.states_list if s not in states_to_resample]
        self.states_list = []
        res = parallel.dv.push({'global_model':self},block=False)
        self.states_list = states_to_restore
        return res

    def _build_states_parallel(self,states_to_resample,temp=None):
        from pyhsmm import parallel
        parallel.dv.push(dict(temp=temp),block=False)
        raw_stateseq_tuples = parallel.dv.map(self._state_builder,
                [s.data_id for s in states_to_resample],block=True)
        for data_id, stateseq in raw_stateseq_tuples:
            self.add_data(data=parallel.alldata[data_id],stateseq=stateseq)
            self.states_list[-1].data_id = data_id

    @staticmethod
    @util.general.interactive
    def _state_builder(data_id):
        # expects globals: global_model, alldata, temp
        global_model.add_data(alldata[data_id],initialize_from_prior=False,temp=temp)
        stateseq = global_model.states_list.pop().stateseq
        return (data_id, stateseq)

    ### EM

    def EM_step(self):
        assert len(self.states_list) > 0, 'Must have data to run EM'
        self._clear_caches()

        ## E step
        for s in self.states_list:
            s.E_step()

        ## M step
        # observation distribution parameters
        for state, distn in enumerate(self.obs_distns):
            distn.max_likelihood([s.data for s in self.states_list],
                    [s.expectations[:,state] for s in self.states_list])

        # initial distribution parameters
        self.init_state_distn.max_likelihood(
                None, # placeholder, "should" be np.arange(self.state_dim)
                [s.expectations[0] for s in self.states_list])

        # transition parameters (requiring more than just the marginal expectations)
        self.trans_distn.max_likelihood(None,[(s.alphal,s.betal,s.aBl) for s in self.states_list])

    def Viterbi_EM_fit(self):
        return self.MAP_EM_fit()

    def MAP_EM_step(self):
        return self.Viterbi_EM_step()

    def Viterbi_EM_step(self):
        assert len(self.states_list) > 0, 'Must have data to run Viterbi EM'
        self._clear_caches()

        ## Viterbi step
        for s in self.states_list:
            s.Viterbi()

        ## M step
        # observation distribution parameters
        for state, distn in enumerate(self.obs_distns):
            distn.max_likelihood([s.data[s.stateseq == state] for s in self.states_list])

        # initial distribution parameters
        self.init_state_distn.max_likelihood(
                np.array([s.stateseq[0] for s in self.states_list]))

        # transition parameters (requiring more than just the marginal expectations)
        self.trans_distn.max_likelihood([s.stateseq for s in self.states_list])

    def num_parameters(self):
        return sum(o.num_parameters() for o in self.obs_distns) + self.state_dim**2

    def BIC(self,data=None):
        '''
        BIC on the passed data. If passed data is None (default), calculates BIC
        on the model's assigned data
        '''
        # NOTE: in principle this method computes the BIC only after finding the
        # maximum likelihood parameters (or, of course, an EM fixed-point as an
        # approximation!)
        assert data is None and len(self.states_list) > 0, 'Must have data to get BIC'
        if data is None:
            return -2*sum(self.log_likelihood(s.data).sum() for s in self.states_list) + \
                        self.num_parameters() * np.log(sum(s.data.shape[0] for s in self.states_list))
        else:
            return -2*self.log_likelihood(data) + self.num_parameters() * np.log(data.shape[0])

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
                        indices=[np.where(s.stateseq == state)[0] for s in states_objs],
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

class HMMEigen(HMM):
    _states_class = states.HMMStatesEigen

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

    def MAP_EM_step(self):
        raise NotImplementedError, "Can't run EM on a StickyHMM"


class StickyHMMEigen(StickyHMM):
    _states_class = states.HMMStatesEigen

class HSMM(HMM, ModelGibbsSampling, ModelEM, ModelMAPEM):
    _states_class = states.HSMMStatesPython
    _trans_class = transitions.HDPHSMMTransitions
    _trans_class_conc_class = transitions.HDPHSMMTransitionsConcResampling
    _init_steady_state_class = initial_state.HSMMSteadyState

    def __init__(self,dur_distns,**kwargs):

        self.dur_distns = dur_distns

        super(HSMM,self).__init__(**kwargs)

        if isinstance(self.init_state_distn,self._init_steady_state_class):
            self.left_censoring_init_state_distn = self.init_state_distn
        else:
            self.left_censoring_init_state_distn = self._init_steady_state_class(self)

    @property
    def stateseqs_norep(self):
        return [s.stateseq_norep for s in self.states_list]

    @property
    def durations(self):
        return [s.durations for s in self.states_list]

    def add_data(self,data,stateseq=None,trunc=None,right_censoring=True,left_censoring=False,
            **kwargs):
        self.states_list.append(self._states_class(
            model=self,
            data=np.asarray(data),
            stateseq=stateseq,
            right_censoring=right_censoring,
            left_censoring=left_censoring,
            trunc=trunc,
            **kwargs))

    def log_likelihood(self,data,trunc=None,**kwargs):
        s = self._states_class(model=self,data=np.asarray(data),trunc=trunc,
                stateseq=np.zeros(len(data)),**kwargs)
        betal, _ = s.messages_backwards()
        return np.logaddexp.reduce(np.log(self.init_state_distn.pi_0) + betal[0] + s.aBl[0])

    ### generation

    def generate(self,T,keep=True,**kwargs):
        tempstates = self._states_class(self,T=T,initialize_from_prior=True,**kwargs)
        return self._generate(tempstates,keep)

    ### Gibbs sampling

    def resample_model(self,**kwargs):
        self.resample_dur_distns()
        super(HSMM,self).resample_model(**kwargs)

    def resample_dur_distns(self):
        for state, distn in enumerate(self.dur_distns):
            distn.resample_with_truncations(
                    data=
                    [s.durations_censored[s.untrunc_slice][s.stateseq_norep[s.untrunc_slice] == state]
                        for s in self.states_list],
                    truncated_data=
                    [s.durations_censored[s.trunc_slice][s.stateseq_norep[s.trunc_slice] == state]
                        for s in self.states_list])
        self._clear_caches()

    def copy_sample(self):
        new = super(HSMM,self).copy_sample()
        new.dur_distns = [d.copy_sample() for d in self.dur_distns]
        return new

    ### parallel

    def add_data_parallel(self,data_id,**kwargs):
        from pyhsmm import parallel
        self.add_data(data=parallel.alldata[data_id],**kwargs)
        self.states_list[-1].data_id = data_id

    def resample_model_parallel(self,numtoresample='all',**kwargs):
        self.resample_dur_distns()
        super(HSMM,self).resample_model_parallel(numtoresample,**kwargs)

    ### EM

    def EM_step(self):
        super(HSMM,self).EM_step()

        # M step for duration distributions
        for state, distn in enumerate(self.dur_distns):
            distn.max_likelihood(
                    None, # placeholder, "should" be [np.arange(s.T) for s in self.states_list]
                    [s.expectations[:,state] for s in self.states_list])

    def Viterbi_EM_step(self):
        super(HSMM,self).Viterbi_EM_step()

        # M step for duration distributions
        for state, distn in enumerate(self.dur_distns):
            distn.max_likelihood(
                    [s.durations[s.stateseq_norep == state] for s in self.states_list])

    def num_parameters(self):
        return sum(o.num_parameters() for o in self.obs_distns) \
                + sum(d.num_parameters() for d in self.dur_distns) \
                + self.state_dim**2 - self.state_dim

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
        # alternative plot that isn't so big
        raise NotImplementedError # TODO

class HSMMEigen(HSMM):
    _states_class = states.HSMMStatesEigen

class HSMMPossibleChangepoints(HSMM, ModelGibbsSampling):
    _states_class = states.HSMMStatesPossibleChangepoints

    def add_data(self,data,changepoints,**kwargs):
        self.states_list.append(
                self._states_class(model=self,changepoints=changepoints,data=np.asarray(data),**kwargs))

    def add_data_parallel(self,data_id,**kwargs):
        raise NotImplementedError # I broke this!
        from pyhsmm import parallel
        self.add_data(data=parallel.alldata[data_id],changepoints=parallel.allchangepoints[data_id],**kwargs)
        self.states_list[-1].data_id = data_id

    def _build_states_parallel(self,states_to_resample):
        from pyhsmm import parallel
        raw_stateseq_tuples = parallel.hsmm_build_states_changepoints.map([s.data_id for s in states_to_resample])
        for data_id, stateseq, stateseq_norep, durations in raw_stateseq_tuples:
            self.add_data(
                    data=parallel.alldata[data_id],
                    changepoints=parallel.allchangepoints[data_id],
                    stateseq=stateseq,
                    stateseq_norep=stateseq_norep,
                    durations=durations)
            self.states_list[-1].data_id = data_id

    def generate(self,T,changepoints,keep=True):
        raise NotImplementedError

    def log_likelihood(self,data,trunc=None):
        raise NotImplementedError

class HSMMGeoApproximation(HSMM):
    _states_class = states.HSMMStatesGeoApproximation

class _HSMMIntNegBinBase(HSMM, HMMEigen):
    __metaclass__ = abc.ABCMeta

    def EM_step(self):
        # needs to use HMM messages that the states objects give us (only betal)
        # on top of that, need to hand things duration distributions... UGH
        # probably need betastarl too plus some indicator variable magic
        raise NotImplementedError # TODO

    def log_likelihood(self,data=None,**kwargs):
        if data is not None:
            s = self._states_class(model=self,data=np.asarray(data),
                    stateseq=np.zeros(len(data)),**kwargs) # stateseq b/c forward gen is slow
            return np.logaddexp.reduce(np.log(s.pi_0) + s.messages_backwards()[0][0] + s.aBl[0])
        else:
            all_initials = np.vstack([s.messages_backwards()[0][0] + np.log(s.pi_0) + s.aBl[0]
                for s in self.states_list])
            return np.logaddexp.reduce(all_initials,axis=1).sum()

    def predictive_likelihoods(self,test_data,forecast_horizons):
        return HMMEigen.predictive_likelihoods(self,test_data,forecast_horizons) # TODO improve speed

    def block_predictive_likelihoods(self,test_data,blocklens):
        return HMMEigen.block_predictive_likelihoods(self,test_data,blocklens) # TODO improve speed

class HSMMIntNegBinVariant(_HSMMIntNegBinBase):
    _states_class = states.HSMMStatesIntegerNegativeBinomialVariant

    def __init__(self,obs_distns,dur_distns,*args,**kwargs):
        assert all(d.__class__ == basic.distributions.NegativeBinomialIntegerRVariantDuration or
                   d.__class__ == basic.distributions.NegativeBinomialFixedRVariantDuration
                   for d in dur_distns)
        super(HSMMIntNegBinVariant,self).__init__(obs_distns=obs_distns,dur_distns=dur_distns,*args,**kwargs)

class HSMMIntNegBin(_HSMMIntNegBinBase):
    _states_class = states.HSMMStatesIntegerNegativeBinomial

    def __init__(self,obs_distns,dur_distns,*args,**kwargs):
        assert all(d.__class__ == basic.distributions.NegativeBinomialIntegerRDuration or
                   d.__class__ == basic.distributions.NegativeBinomialFixedRDuration
                   for d in dur_distns)
        super(HSMMIntNegBin,self).__init__(obs_distns,dur_distns,*args,**kwargs)

