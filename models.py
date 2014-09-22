from __future__ import division
import numpy as np
from numpy import newaxis as na
import itertools, collections, operator, random, abc, copy
from matplotlib import pyplot as plt
from matplotlib import cm
from warnings import warn

from basic.abstractions import Model, ModelGibbsSampling, \
        ModelEM, ModelMAPEM, ModelMeanField, ModelMeanFieldSVI, ModelParallelTempering
import basic.distributions
from internals import hmm_states, hsmm_states, hsmm_inb_states, \
        initial_state, transitions
import util.general
from util.profiling import line_profiled

################
#  HMM Mixins  #
################

class _HMMBase(Model):
    _states_class = hmm_states.HMMStatesPython
    _trans_class = transitions.HMMTransitions
    _trans_conc_class = transitions.HMMTransitionsConc
    _init_state_class = initial_state.HMMInitialState

    def __init__(self,
            obs_distns,
            trans_distn=None,
            alpha=None,alpha_a_0=None,alpha_b_0=None,trans_matrix=None,
            init_state_distn=None,init_state_concentration=None,pi_0=None):
        self.obs_distns = obs_distns
        self.states_list = []

        if trans_distn is not None:
            self.trans_distn = trans_distn
        elif not None in (alpha_a_0,alpha_b_0):
            self.trans_distn = self._trans_conc_class(
                    num_states=len(obs_distns),
                    alpha_a_0=alpha_a_0,alpha_b_0=alpha_b_0,
                    trans_matrix=trans_matrix)
        else:
            self.trans_distn = self._trans_class(
                    num_states=len(obs_distns),alpha=alpha,trans_matrix=trans_matrix)

        if init_state_distn is not None:
            if init_state_distn == 'uniform':
                self.init_state_distn = initial_state.UniformInitialState(model=self)
            else:
                self.init_state_distn = init_state_distn
        else:
            self.init_state_distn = self._init_state_class(
                    model=self,
                    init_state_concentration=init_state_concentration,
                    pi_0=pi_0)

        self._clear_caches()

    def add_data(self,data,stateseq=None,**kwargs):
        self.states_list.append(
                self._states_class(
                    model=self,data=data,
                    stateseq=stateseq,**kwargs))

    def generate(self,T,keep=True):
        s = self._states_class(model=self,T=T,initialize_from_prior=True)
        data, stateseq = s.generate_obs(), s.stateseq
        if keep:
            self.states_list.append(s)
        return data, stateseq

    def log_likelihood(self,data=None,**kwargs):
        if data is not None:
            if isinstance(data,np.ndarray):
                self.add_data(data=data,generate=False,**kwargs)
                return self.states_list.pop().log_likelihood()
            else:
                assert isinstance(data,list)
                loglike = 0.
                for d in data:
                    self.add_data(data=d,generate=False,**kwargs)
                    loglike += self.states_list.pop().log_likelihood()
                return loglike
        else:
            return sum(s.log_likelihood() for s in self.states_list)

    @property
    def stateseqs(self):
        return [s.stateseq for s in self.states_list]

    @property
    def num_states(self):
        return len(self.obs_distns)

    @property
    def num_parameters(self):
        return sum(o.num_parameters() for o in self.obs_distns) + self.num_states**2

    ### predicting

    def heldout_viterbi(self,data,**kwargs):
        self.add_data(data=data,stateseq=np.zeros(len(data)),**kwargs)
        s = self.states_list.pop()
        s.Viterbi()
        return s.stateseq

    def heldout_state_marginals(self,data,**kwargs):
        self.add_data(data=data,stateseq=np.zeros(len(data)),**kwargs)
        s = self.states_list.pop()
        s.E_step()
        return s.expected_states

    def _resample_from_mf(self):
        self.trans_distn._resample_from_mf()
        self.init_state_distn._resample_from_mf()
        for o in self.obs_distns:
            o._resample_from_mf()

    ### caching

    def _clear_caches(self):
        for s in self.states_list:
            s.clear_caches()

    def __getstate__(self):
        self._clear_caches()
        return self.__dict__.copy()

    ### plotting

    def _get_used_states(self,states_objs=None):
        if states_objs is None:
            states_objs = self.states_list
        canonical_ids = collections.defaultdict(itertools.count().next)
        for s in states_objs:
            for state in s.stateseq:
                canonical_ids[state]
        return map(operator.itemgetter(0),
                sorted(canonical_ids.items(),key=operator.itemgetter(1)))

    def _get_colors(self,states_objs=None):
        states_objs = self.states_list if states_objs is None else states_objs
        if len(states_objs) > 0:
            states = self._get_used_states(states_objs)
        else:
            states = range(len(self.obs_distns))
        numstates = len(states)
        return dict(zip(states,np.linspace(0,1,numstates,endpoint=True)))

    def plot_observations(self,colors=None,states_objs=None):
        if states_objs is None:
            states_objs = self.states_list

        if colors is None:
            colors = self._get_colors()
        cmap = cm.get_cmap()

        if len(states_objs) > 0:
            used_states = self._get_used_states(states_objs)
            for state,o in enumerate(self.obs_distns):
                if state in used_states:
                    o.plot(
                        color=cmap(colors[state]),
                        data=[s.data[(s.stateseq == state) & (~np.isnan(s.data).any(1))]
                                if s.data is not None else None
                            for s in states_objs],
                        indices=[np.where(s.stateseq == state)[0] for s in states_objs],
                        label='%d' % state)
        else:
            N = len(self.obs_distns)
            weights = np.repeat(1./N,N).dot(
                    np.linalg.matrix_power(self.trans_distn.trans_matrix,1000))
            for state, o in enumerate(self.obs_distns):
                o.plot(
                        color=cmap(colors[state]),
                        label='%d' % state,
                        alpha=min(1.,weights[state]+0.05))
        plt.title('Observation Distributions')

    def plot(self,color=None,legend=False):
        plt.gcf() #.set_size_inches((10,10))

        if len(self.states_list) > 0:
            colors = self._get_colors(self.states_list)
            num_subfig_cols = len(self.states_list)
            for subfig_idx,s in enumerate(self.states_list):
                plt.subplot(2,num_subfig_cols,1+subfig_idx)
                self.plot_observations(colors=colors,states_objs=[s])

                plt.subplot(2,num_subfig_cols,1+num_subfig_cols+subfig_idx)
                s.plot(colors_dict=colors)

            if legend:
                plt.legend()
        else:
            self.plot_observations()

class _HMMGibbsSampling(_HMMBase,ModelGibbsSampling):
    @line_profiled
    def resample_model(self,joblib_jobs=0):
        self.resample_parameters()
        self.resample_states(joblib_jobs=joblib_jobs)

    @line_profiled
    def resample_parameters(self):
        self.resample_obs_distns()
        self.resample_trans_distn()
        self.resample_init_state_distn()

    def resample_obs_distns(self):
        for state, distn in enumerate(self.obs_distns):
            distn.resample([s.data[s.stateseq == state] for s in self.states_list])
        self._clear_caches()

    def resample_trans_distn(self):
        self.trans_distn.resample([s.stateseq for s in self.states_list])
        self._clear_caches()

    def resample_init_state_distn(self):
        self.init_state_distn.resample([s.stateseq[0] for s in self.states_list])
        self._clear_caches()

    def resample_states(self,joblib_jobs=0):
        if joblib_jobs == 0:
            for s in self.states_list:
                s.resample()
        else:
            self._joblib_resample_states(self.states_list,joblib_jobs)

    def copy_sample(self):
        new = copy.copy(self)
        new.obs_distns = [o.copy_sample() for o in self.obs_distns]
        new.trans_distn = self.trans_distn.copy_sample()
        new.init_state_distn = self.init_state_distn.copy_sample()
        new.states_list = [s.copy_sample(new) for s in self.states_list]
        return new

    ### joblib parallel stuff here

    def _joblib_resample_states(self,states_list,joblib_jobs):
        from joblib import Parallel, delayed
        from parallel import _get_sampled_stateseq

        warn('joblib is segfaulting on OS X only, not sure why')

        if len(states_list) > 0:
            joblib_args = util.general.list_split(
                    [(s.data,s._kwargs) for s in states_list],
                    joblib_jobs)
            raw_stateseqs = Parallel(n_jobs=joblib_jobs,backend='multiprocessing')\
                    (delayed(_get_sampled_stateseq)(self,arg) for arg in joblib_args)

            for s, (stateseq, log_likelihood) in zip(
                    states_list,[seq for grp in raw_stateseqs for seq in grp]):
                s.stateseq, s._normalizer = stateseq, log_likelihood

class _HMMMeanField(_HMMBase,ModelMeanField):
    def meanfield_coordinate_descent_step(self,joblib_jobs=0):
        self._meanfield_update_sweep(joblib_jobs=joblib_jobs)
        return self._vlb()

    def _meanfield_update_sweep(self,joblib_jobs=0):
        # NOTE: we want to update the states factor last to make the VLB
        # computation efficient, but to update the parameters first we have to
        # ensure everything in states_list has expected statistics computed
        self._meanfield_update_states_list(
            [s for s in self.states_list if not hasattr(s,'expected_states')],
            joblib_jobs)

        self.meanfield_update_parameters()
        self.meanfield_update_states(joblib_jobs)

    def meanfield_update_parameters(self):
        self.meanfield_update_obs_distns()
        self.meanfield_update_trans_distn()
        self.meanfield_update_init_state_distn()

    def meanfield_update_obs_distns(self):
        for state, o in enumerate(self.obs_distns):
            o.meanfieldupdate([s.data for s in self.states_list],
                    [s.expected_states[:,state] for s in self.states_list])

    def meanfield_update_trans_distn(self):
        self.trans_distn.meanfieldupdate(
                [s.expected_transcounts for s in self.states_list])

    def meanfield_update_init_state_distn(self):
        self.init_state_distn.meanfieldupdate(
                [s.expected_states[0] for s in self.states_list])

    def meanfield_update_states(self,joblib_jobs=0):
        self._meanfield_update_states_list(self.states_list,joblib_jobs=joblib_jobs)

    def _meanfield_update_states_list(self,states_list,joblib_jobs=0):
        if joblib_jobs == 0:
            for s in states_list:
                s.meanfieldupdate()
        else:
            self._joblib_meanfield_update_states(states_list,joblib_jobs)

    def _vlb(self):
        vlb = 0.
        vlb += sum(s.get_vlb() for s in self.states_list)
        vlb += self.trans_distn.get_vlb()
        vlb += self.init_state_distn.get_vlb()
        vlb += sum(o.get_vlb() for o in self.obs_distns)
        return vlb

    ### joblib parallel stuff here

    def _joblib_meanfield_update_states(self,states_list,joblib_jobs):
        from joblib import Parallel, delayed
        from parallel import _get_stats

        warn('joblib is segfaulting on OS X only, not sure why')

        if len(states_list) > 0:
            joblib_args = util.general.list_split(
                    [(s.data,s._kwargs) for s in states_list],
                    joblib_jobs)
            allstats = Parallel(n_jobs=joblib_jobs,backend='multiprocessing')\
                    (delayed(_get_stats)(self,arg) for arg in joblib_args)

            for s, stats in zip(states_list,[s for grp in allstats for s in grp]):
                s.all_expected_stats = stats

class _HMMSVI(_HMMBase,ModelMeanFieldSVI):
    # NOTE: classes with this mixin should also have the _HMMMeanField mixin for
    # joblib stuff to work
    def meanfield_sgdstep(self,minibatch,minibatchfrac,stepsize,joblib_jobs=0,**kwargs):
        ## compute the local mean field step for the minibatch
        mb_states_list = self._get_mb_states_list(minibatch,**kwargs)
        if joblib_jobs == 0:
            for s in mb_states_list:
                s.meanfieldupdate()
        else:
            self._joblib_meanfield_update_states(mb_states_list,joblib_jobs)

        ## take a global step on the parameters
        self._meanfield_sgdstep_parameters(mb_states_list,minibatchfrac,stepsize)

    def _get_mb_states_list(self,minibatch,**kwargs):
        minibatch = minibatch if isinstance(minibatch,list) else [minibatch]
        mb_states_list = []
        for mb in minibatch:
            self.add_data(mb,generate=False,**kwargs)
            mb_states_list.append(self.states_list.pop())
        return mb_states_list

    def _meanfield_sgdstep_parameters(self,mb_states_list,minibatchfrac,stepsize):
        self._meanfield_sgdstep_obs_distns(mb_states_list,minibatchfrac,stepsize)
        self._meanfield_sgdstep_trans_distn(mb_states_list,minibatchfrac,stepsize)
        self._meanfield_sgdstep_init_state_distn(mb_states_list,minibatchfrac,stepsize)

    def _meanfield_sgdstep_obs_distns(self,mb_states_list,minibatchfrac,stepsize):
        for state, o in enumerate(self.obs_distns):
            o.meanfield_sgdstep(
                    [s.data for s in mb_states_list],
                    [s.expected_states[:,state] for s in mb_states_list],
                    minibatchfrac,stepsize)

    def _meanfield_sgdstep_trans_distn(self,mb_states_list,minibatchfrac,stepsize):
        self.trans_distn.meanfield_sgdstep(
                [s.expected_transcounts for s in mb_states_list],
                minibatchfrac,stepsize)

    def _meanfield_sgdstep_init_state_distn(self,mb_states_list,minibatchfrac,stepsize):
        self.init_state_distn.meanfield_sgdstep(
                [s.expected_states[0] for s in mb_states_list],
                minibatchfrac,stepsize)

class _HMMEM(_HMMBase,ModelEM):
    def EM_step(self):
        assert len(self.states_list) > 0, 'Must have data to run EM'
        self._clear_caches()
        self._E_step()
        self._M_step()

    def _E_step(self):
        for s in self.states_list:
            s.E_step()

    def _M_step(self):
        self._M_step_obs_distns()
        self._M_step_init_state_distn()
        self._M_step_trans_distn()

    def _M_step_obs_distns(self):
        for state, distn in enumerate(self.obs_distns):
            distn.max_likelihood([s.data for s in self.states_list],
                    [s.expected_states[:,state] for s in self.states_list])

    def _M_step_init_state_distn(self):
        self.init_state_distn.max_likelihood(
                expected_states_list=[s.expected_states[0] for s in self.states_list])

    def _M_step_trans_distn(self):
        self.trans_distn.max_likelihood(
                expected_transcounts=[s.expected_transcounts for s in self.states_list])

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
                        self.num_parameters() * np.log(
                                sum(s.data.shape[0] for s in self.states_list))
        else:
            return -2*self.log_likelihood(data) + self.num_parameters() * np.log(data.shape[0])

class _HMMViterbiEM(_HMMBase,ModelMAPEM):
    def Viterbi_EM_fit(self, tol=0.1, maxiter=20):
        return self.MAP_EM_fit(tol, maxiter)

    def Viterbi_EM_step(self):
        assert len(self.states_list) > 0, 'Must have data to run Viterbi EM'
        self._clear_caches()
        self._Viterbi_E_step()
        self._Viterbi_M_step()

    def _Viterbi_E_step(self):
        for s in self.states_list:
            s.Viterbi()

    def _Viterbi_M_step(self):
        self._Viterbi_M_step_obs_distns()
        self._Viterbi_M_step_init_state_distn()
        self._Viterbi_M_step_trans_distn()

    def _Viterbi_M_step_obs_distns(self):
        for state, distn in enumerate(self.obs_distns):
            distn.max_likelihood([s.data[s.stateseq == state] for s in self.states_list])

    def _Viterbi_M_step_init_state_distn(self):
        self.init_state_distn.max_likelihood(
                samples=np.array([s.stateseq[0] for s in self.states_list]))

    def _Viterbi_M_step_trans_distn(self):
        self.trans_distn.max_likelihood([s.stateseq for s in self.states_list])

    MAP_EM_step = Viterbi_EM_step # for the ModelMAPEM interface

class _WeakLimitHDPMixin(object):
    def __init__(self,
            obs_distns,
            trans_distn=None,alpha=None,alpha_a_0=None,alpha_b_0=None,
            gamma=None,gamma_a_0=None,gamma_b_0=None,trans_matrix=None,
            **kwargs):

        if trans_distn is not None:
            trans_distn = trans_distn
        elif not None in (alpha_a_0,alpha_b_0):
            trans_distn = self._trans_conc_class(
                    num_states=len(obs_distns),
                    alpha_a_0=alpha_a_0,alpha_b_0=alpha_b_0,
                    gamma_a_0=gamma_a_0,gamma_b_0=gamma_b_0,
                    trans_matrix=trans_matrix)
        else:
            trans_distn = self._trans_class(
                    num_states=len(obs_distns),alpha=alpha,gamma=gamma,
                    trans_matrix=trans_matrix)

        super(_WeakLimitHDPMixin,self).__init__(
                obs_distns=obs_distns,trans_distn=trans_distn,**kwargs)

class _HMMPossibleChangepointsMixin(object):
    _states_class = hmm_states.HMMStatesPossibleChangepoints

    def add_data(self,data,changepoints=None,**kwargs):
        super(_HMMPossibleChangepointsMixin,self).add_data(
                data=data,changepoints=changepoints,**kwargs)

    def _get_mb_states_list(self,minibatch,changepoints=None,**kwargs):
        if changepoints is not None:
            if not isinstance(minibatch,(list,tuple)):
                assert isinstance(minibatch,np.ndarray)
                assert isinstance(changepoints,list) and isinstance(changepoints[0],tuple)
                minibatch = [minibatch]
                changepoints = [changepoints]
            else:
                assert  isinstance(changepoints,(list,tuple))  \
                        and isinstance(changepoints[0],(list,tuple)) \
                        and isinstance(changepoints[0][0],tuple)
                assert len(minibatch) == len(changepoints)

        changepoints = changepoints if changepoints is not None \
                else [None]*len(minibatch)

        mb_states_list = []
        for data, changes in zip(minibatch,changepoints):
            self.add_data(data,changepoints=changes,generate=False,**kwargs)
            mb_states_list.append(self.states_list.pop())
        return mb_states_list

    def log_likelihood(self,data=None,changepoints=None,**kwargs):
        if data is not None:
            if isinstance(data,np.ndarray):
                assert isinstance(changepoints,list) or changepoints is None
                self.add_data(data=data,changepoints=changepoints,
                        generate=False,**kwargs)
                return self.states_list.pop().log_likelihood()
            else:
                assert isinstance(data,list) and (changepoints is None
                    or isinstance(changepoints,list) and len(changepoints) == len(data))
                changepoints = changepoints if changepoints is not None \
                        else [None]*len(data)

                loglike = 0.
                for d, c in zip(data,changepoints):
                    self.add_data(data=d,changepoints=c,generate=False,**kwargs)
                    loglike += self.states_list.pop().log_likelihood()
                return loglike
        else:
            return sum(s.log_likelihood() for s in self.states_list)

class _HMMParallelTempering(_HMMBase,ModelParallelTempering):
    @property
    def temperature(self):
        return self._temperature if hasattr(self,'_temperature') else 1.

    @temperature.setter
    def temperature(self,T):
        self._temperature = T
        self._clear_caches()

    def swap_sample_with(self,other):
        self.obs_distns, other.obs_distns = other.obs_distns, self.obs_distns

        self.trans_distn, other.trans_distn = other.trans_distn, self.trans_distn

        self.init_state_distn, other.init_state_distn = \
                other.init_state_distn, self.init_state_distn
        self.init_state_distn.model = self
        other.init_state_distn.model = other

        for s1, s2 in zip(self.states_list, other.states_list):
            s1.stateseq, s2.stateseq = s2.stateseq, s1.stateseq

        self._clear_caches()

    @property
    def energy(self):
        energy = 0.
        for s in self.states_list:
            for state, datum in zip(s.stateseq,s.data):
                energy += self.obs_distns[state].energy(datum)
        return energy

################
#  HMM models  #
################

class HMMPython(_HMMGibbsSampling,_HMMSVI,_HMMMeanField,_HMMEM,
        _HMMViterbiEM,_HMMParallelTempering):
    pass

class HMM(HMMPython):
    _states_class = hmm_states.HMMStatesEigen

class WeakLimitHDPHMMPython(_WeakLimitHDPMixin,HMMPython):
    # NOTE: shouldn't really inherit EM or ViterbiEM, but it's convenient!
    _trans_class = transitions.WeakLimitHDPHMMTransitions
    _trans_conc_class = transitions.WeakLimitHDPHMMTransitionsConc

class WeakLimitHDPHMM(_WeakLimitHDPMixin,HMM):
    _trans_class = transitions.WeakLimitHDPHMMTransitions
    _trans_conc_class = transitions.WeakLimitHDPHMMTransitionsConc

class DATruncHDPHMMPython(_WeakLimitHDPMixin,HMMPython):
    # NOTE: weak limit mixin is poorly named; we just want its init method
    _trans_class = transitions.DATruncHDPHMMTransitions
    _trans_conc_class = None

class DATruncHDPHMM(_WeakLimitHDPMixin,HMM):
    _trans_class = transitions.DATruncHDPHMMTransitions
    _trans_conc_class = None

class WeakLimitStickyHDPHMM(WeakLimitHDPHMM):
    # TODO concentration resampling, too!
    def __init__(self,obs_distns,
            kappa=None,alpha=None,gamma=None,trans_matrix=None,
            alpha_a_0=None,alpha_b_0=None,gamma_a_0=None,gamma_b_0=None,
            **kwargs):
        assert (None not in (alpha,gamma)) ^ \
                (None not in (alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0))
        if None not in (alpha,gamma):
            trans_distn = transitions.WeakLimitStickyHDPHMMTransitions(
                    num_states=len(obs_distns),
                    kappa=kappa,alpha=alpha,gamma=gamma,trans_matrix=trans_matrix)
        else:
            trans_distn = transitions.WeakLimitStickyHDPHMMTransitionsConc(
                    num_states=len(obs_distns),
                    kappa=kappa,
                    alpha_a_0=alpha_a_0,alpha_b_0=alpha_b_0,
                    gamma_a_0=gamma_a_0,gamma_b_0=gamma_b_0,
                    trans_matrix=trans_matrix)
        super(WeakLimitStickyHDPHMM,self).__init__(
                obs_distns=obs_distns,trans_distn=trans_distn,**kwargs)

class HMMPossibleChangepoints(_HMMPossibleChangepointsMixin,HMM):
    pass

#################
#  HSMM Mixins  #
#################

class _HSMMBase(_HMMBase):
    _states_class = hsmm_states.HSMMStatesPython
    _trans_class = transitions.HSMMTransitions
    _trans_conc_class = transitions.HSMMTransitionsConc
    # _init_steady_state_class = initial_state.HSMMSteadyState # TODO

    def __init__(self,dur_distns,**kwargs):
        self.dur_distns = dur_distns
        super(_HSMMBase,self).__init__(**kwargs)

    def add_data(self,data,stateseq=None,trunc=None,
            right_censoring=True,left_censoring=False,**kwargs):
        self.states_list.append(self._states_class(
            model=self,
            data=np.asarray(data),
            stateseq=stateseq,
            right_censoring=right_censoring,
            left_censoring=left_censoring,
            trunc=trunc,
            **kwargs))

    @property
    def stateseqs_norep(self):
        return [s.stateseq_norep for s in self.states_list]

    @property
    def durations(self):
        return [s.durations for s in self.states_list]

    @property
    def num_parameters(self):
        return sum(o.num_parameters() for o in self.obs_distns) \
                + sum(d.num_parameters() for d in self.dur_distns) \
                + self.num_states**2 - self.num_states

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
        colors = self._get_colors(self.states_list)

        num_subfig_cols = len(self.states_list)
        for subfig_idx,s in enumerate(self.states_list):
            plt.subplot(3,num_subfig_cols,1+subfig_idx)
            self.plot_observations(colors=colors,states_objs=[s])

            plt.subplot(3,num_subfig_cols,1+num_subfig_cols+subfig_idx)
            s.plot(colors_dict=colors)

            plt.subplot(3,num_subfig_cols,1+2*num_subfig_cols+subfig_idx)
            self.plot_durations(colors=colors,states_objs=[s])

class _HSMMGibbsSampling(_HSMMBase,_HMMGibbsSampling):
    @line_profiled
    def resample_parameters(self):
        self.resample_dur_distns()
        super(_HSMMGibbsSampling,self).resample_parameters()

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
        new = super(_HSMMGibbsSampling,self).copy_sample()
        new.dur_distns = [d.copy_sample() for d in self.dur_distns]
        return new

class _HSMMEM(_HSMMBase,_HMMEM):
    def _M_step(self):
        super(_HSMMEM,self)._M_step()
        self._M_step_dur_distns()

    def _M_step_dur_distns(self):
        for state, distn in enumerate(self.dur_distns):
            distn.max_likelihood(
                    [np.arange(1,s.expected_durations[state].shape[0]+1)
                        for s in self.states_list],
                    [s.expected_durations[state] for s in self.states_list])

class _HSMMMeanField(_HSMMBase,_HMMMeanField):
    def meanfield_update_parameters(self):
        super(_HSMMMeanField,self).meanfield_update_parameters()
        self.meanfield_update_dur_distns()

    def meanfield_update_dur_distns(self):
        for state, d in enumerate(self.dur_distns):
            d.meanfieldupdate(
                    [np.arange(1,s.expected_durations[state].shape[0]+1)
                        for s in self.states_list],
                    [s.expected_durations[state] for s in self.states_list])

    def _vlb(self):
        vlb = super(_HSMMMeanField,self)._vlb()
        vlb += sum(d.get_vlb() for d in self.dur_distns)
        return vlb

class _HSMMSVI(_HSMMBase,_HMMSVI):
    def _meanfield_sgdstep_parameters(self,mb_states_list,minibatchfrac,stepsize):
        super(_HSMMSVI,self)._meanfield_sgdstep_parameters(mb_states_list,minibatchfrac,stepsize)
        self._meanfield_sgdstep_dur_distns(mb_states_list,minibatchfrac,stepsize)

    def _meanfield_sgdstep_dur_distns(self,mb_states_list,minibatchfrac,stepsize):
        for state, d in enumerate(self.dur_distns):
            d.meanfield_sgdstep(
                    [np.arange(1,s.expected_durations[state].shape[0]+1)
                        for s in mb_states_list],
                    [s.expected_durations[state] for s in mb_states_list],
                    minibatchfrac,stepsize)

class _HSMMINBEMMixin(_HMMEM,ModelEM):
    def EM_step(self):
        super(_HSMMINBEMMixin,self).EM_step()
        for state, distn in enumerate(self.dur_distns):
            distn.max_likelihood(data=None,stats=(
                sum(s.expected_dur_ns[state] for s in self.states_list),
                sum(s.expected_dur_tots[state] for s in self.states_list)))

class _HSMMViterbiEM(_HSMMBase,_HMMViterbiEM):
    def Viterbi_EM_step(self):
        super(_HSMMViterbiEM,self).Viterbi_EM_step()
        self._Viterbi_M_step_dur_distns()

    def _Viterbi_M_step_dur_distns(self):
        for state, distn in enumerate(self.dur_distns):
            distn.max_likelihood(
                    [s.durations[s.stateseq_norep == state] for s in self.states_list])

    def _Viterbi_M_step_trans_distn(self):
        self.trans_distn.max_likelihood([s.stateseq_norep for s in self.states_list])

class _HSMMPossibleChangepointsMixin(_HMMPossibleChangepointsMixin):
    _states_class = hsmm_states.HSMMStatesPossibleChangepoints

class _HSMMParallelTempering(_HSMMBase,_HMMParallelTempering):
    def swap_sample_with(self,other):
        self.dur_distns, other.dur_distns = other.dur_distns, self.dur_distns
        super(_HSMMParallelTempering,self).swap_sample_with(other)

#################
#  HSMM Models  #
#################

class HSMMPython(_HSMMGibbsSampling,_HSMMSVI,_HSMMMeanField,
        _HSMMViterbiEM,_HSMMEM,_HSMMParallelTempering):
    _trans_class = transitions.HSMMTransitions
    _trans_conc_class = transitions.HSMMTransitionsConc

class HSMM(HSMMPython):
    _states_class = hsmm_states.HSMMStatesEigen

class GeoHSMM(HSMMPython):
    _states_class = hsmm_states.GeoHSMMStates

class WeakLimitHDPHSMMPython(_WeakLimitHDPMixin,HSMMPython):
    # NOTE: shouldn't technically inherit EM or ViterbiEM, but it's convenient
    _trans_class = transitions.WeakLimitHDPHSMMTransitions
    _trans_conc_class = transitions.WeakLimitHDPHSMMTransitionsConc

class WeakLimitHDPHSMM(_WeakLimitHDPMixin,HSMM):
    _trans_class = transitions.WeakLimitHDPHSMMTransitions
    _trans_conc_class = transitions.WeakLimitHDPHSMMTransitionsConc

class WeakLimitGeoHDPHSMM(WeakLimitHDPHSMM):
    _states_class = hsmm_states.GeoHSMMStates

    def _M_step_dur_distns(self):
        warn('untested!')
        for state, distn in enumerate(self.dur_distns):
            distn.max_likelihood(
                    stats=(
                        sum(s._expected_ns[state] for s in self.states_list),
                        sum(s._expected_tots[state] for s in self.states_list),
                        ))

class DATruncHDPHSMM(_WeakLimitHDPMixin,HSMM):
    # NOTE: weak limit mixin is poorly named; we just want its init method
    _trans_class = transitions.DATruncHDPHSMMTransitions
    _trans_conc_class = None

class HSMMIntNegBin(_HSMMGibbsSampling,_HSMMMeanField,_HSMMSVI,_HSMMViterbiEM,
        _HSMMParallelTempering):
    _trans_class = transitions.HSMMTransitions
    _trans_conc_class = transitions.HSMMTransitionsConc
    _states_class = hsmm_inb_states.HSMMStatesIntegerNegativeBinomial

    def _resample_from_mf(self):
        super(HSMMIntNegBin,self)._resample_from_mf()
        for d in self.dur_distns:
            d._resample_from_mf()

    def _vlb(self):
        return 0. # TODO

class WeakLimitHDPHSMMIntNegBin(_WeakLimitHDPMixin,HSMMIntNegBin):
    _trans_class = transitions.WeakLimitHDPHSMMTransitions
    _trans_conc_class = transitions.WeakLimitHDPHSMMTransitionsConc

class HSMMIntNegBinVariant(_HSMMGibbsSampling,_HSMMINBEMMixin,_HSMMViterbiEM,
        _HSMMParallelTempering):
    _trans_class = transitions.HSMMTransitions
    _trans_conc_class = transitions.HSMMTransitionsConc
    _states_class = hsmm_inb_states.HSMMStatesIntegerNegativeBinomialVariant

class WeakLimitHDPHSMMIntNegBinVariant(_WeakLimitHDPMixin,HSMMIntNegBinVariant):
    _trans_class = transitions.WeakLimitHDPHSMMTransitions
    _trans_conc_class = transitions.WeakLimitHDPHSMMTransitionsConc

class GeoHSMMPossibleChangepoints(_HSMMPossibleChangepointsMixin,GeoHSMM):
    pass

class HSMMPossibleChangepoints(_HSMMPossibleChangepointsMixin,HSMMPython):
    pass

class WeakLimitHDPHSMMPossibleChangepoints(_HSMMPossibleChangepointsMixin,WeakLimitHDPHSMM):
    pass

##########
#  meta  #
##########

class _SeparateTransMixin(object):
    def __init__(self,*args,**kwargs):
        super(_SeparateTransMixin,self).__init__(*args,**kwargs)
        self.trans_distns = collections.defaultdict(
                lambda: copy.deepcopy(self.trans_distn))
        self.init_state_distns = collections.defaultdict(
                lambda: copy.deepcopy(self.init_state_distn))

    def __getstate__(self):
        dct = self.__dict__.copy()
        dct['trans_distns'] = dict(self.trans_distns.items())
        dct['init_state_distns'] = dict(self.init_state_distns.items())
        return dct

    def __setstate__(self,dct):
        self.__dict__.update(dct)
        self.trans_distns = collections.defaultdict(
                lambda: copy.deepcopy(self.trans_distn))
        self.init_state_distns = collections.defaultdict(
                lambda: copy.deepcopy(self.init_state_distn))
        self.trans_distns.update(dct['trans_distns'])
        self.init_state_distns.update(dct['init_state_distns'])

    ### parallel tempering

    def swap_sample_with(self,other):
        self.trans_distns, other.trans_distns = self.trans_distns, other.trans_distns
        self.init_state_distns, other.init_state_distns = \
                other.init_state_distns, self.init_state_distns
        for d1, d2 in zip(self.init_state_distns.values(),other.init_state_distns.values()):
            d1.model = self
            d2.model = other
        super(_SeparateTransMixin,self).swap_sample_with(other)

    ### Gibbs sampling

    def resample_trans_distn(self):
        for group_id, trans_distn in self.trans_distns.iteritems():
            trans_distn.resample([s.stateseq for s in self.states_list
                if hash(s.group_id) == hash(group_id)])
        self._clear_caches()

    def resample_init_state_distn(self):
        for group_id, init_state_distn in self.init_state_distns.iteritems():
            init_state_distn.resample([s.stateseq[0] for s in self.states_list
                if hash(s.group_id) == hash(group_id)])
        self._clear_caches()

    ### Mean field

    def meanfield_update_trans_distn(self):
        for group_id, trans_distn in self.trans_distns.iteritems():
            states_list = [s for s in self.states_list if hash(s.group_id) == hash(group_id)]
            if len(states_list) > 0:
                trans_distn.meanfieldupdate([s.expected_transcounts for s in states_list])

    def meanfield_update_init_state_distn(self):
        for group_id, init_state_distn in self.init_state_distns.iteritems():
            states_list = [s for s in self.states_list if hash(s.group_id) == hash(group_id)]
            if len(states_list) > 0:
                init_state_distn.meanfieldupdate([s.expected_states[0] for s in states_list])

    def _vlb(self):
        vlb = 0.
        vlb += sum(s.get_vlb() for s in self.states_list)
        vlb += sum(trans_distn.get_vlb()
                for trans_distn in self.trans_distns.itervalues())
        vlb += sum(init_state_distn.get_vlb()
                for init_state_distn in self.init_state_distns.itervalues())
        vlb += sum(o.get_vlb() for o in self.obs_distns)
        return vlb


    ### SVI

    def _meanfield_sgdstep_trans_distn(self,mb_states_list,minibatchfrac,stepsize):
        for group_id, trans_distn in self.trans_distns.iteritems():
            trans_distn.meanfield_sgdstep(
                    [s.expected_transcounts for s in mb_states_list
                        if hash(s.group_id) == hash(group_id)],
                    minibatchfrac,stepsize)

    def _meanfield_sgdstep_init_state_distn(self,mb_states_list,minibatchfrac,stepsize):
        for group_id, init_state_distn in self.init_state_distns.iteritems():
            init_state_distn.meanfield_sgdstep(
                    [s.expected_states[0] for s in mb_states_list
                        if hash(s.group_id) == hash(group_id)],
                    minibatchfrac,stepsize)

    ### EM

    def EM_step(self):
        raise NotImplementedError

    ### Viterbi

    def Viterbi_EM_step(self):
        raise NotImplementedError

class HMMSeparateTrans(
        _SeparateTransMixin,HMM):
    _states_class = hmm_states.HMMStatesEigenSeparateTrans

class HSMMPossibleChangepointsSeparateTrans(
        _SeparateTransMixin,
        HSMMPossibleChangepoints):
    _states_class = hsmm_states.HSMMStatesPossibleChangepointsSeparateTrans

class WeakLimitHDPHSMMPossibleChangepointsSeparateTrans(
        _SeparateTransMixin,
        WeakLimitHDPHSMMPossibleChangepoints):
    _states_class = hsmm_states.HSMMStatesPossibleChangepointsSeparateTrans


##########
#  temp  #
##########

class DiagGaussHSMMPossibleChangepointsSeparateTrans(
        HSMMPossibleChangepointsSeparateTrans):
    _states_class = hsmm_states.DiagGaussStates

    def init_meanfield_from_sample(self):
        for s in self.states_list:
            s.init_meanfield_from_sample()
        self.meanfield_update_parameters()

    def resample_obs_distns(self):
        # collects the statistics using fast code
        assert all(s.data.dtype == np.float64 for s in self.states_list)
        assert all(s.stateseq.dtype == np.int32 for s in self.states_list)

        if len(self.states_list) > 0:
            from util.temp import getstats
            allstats = getstats(
                    len(self.obs_distns),
                    [s.stateseq for s in self.states_list],
                    [s.data for s in self.states_list])

            for state, (distn, stats) in enumerate(zip(self.obs_distns,allstats)):
                distn.resample(stats=stats,temperature=self.temperature)
        else:
            for distn in self.obs_distns:
                distn.resample(temperature=self.temperature)
        self._clear_caches()

class DiagGaussGMMHSMMPossibleChangepointsSeparateTrans(
        HSMMPossibleChangepointsSeparateTrans):
    _states_class = hsmm_states.DiagGaussGMMStates

    def init_meanfield_from_sample(self,niter=5):
        for s in self.states_list:
            s.init_meanfield_from_sample()
        self.meanfield_update_parameters()

        # extra iterations for GMM fitting
        for i in xrange(niter-1):
            self.meanfield_update_obs_distns()

    def resample_obs_distns(self):
        from .util.temp import resample_gmm_labels

        datas = [s.data for s in self.states_list]
        stateseqs = [s.stateseq.astype('int32') for s in self.states_list]

        for itr in xrange(self.obs_distns[0].niter):
            mus = np.array([[c.mu for c in d.components] for d in self.obs_distns])
            sigmas = np.array([[c.sigmas for c in d.components] for d in self.obs_distns])
            logweights = np.log(np.array([d.weights.weights for d in self.obs_distns]))

            if self.temperature is not None:
                sigmas *= self.temperature

            randseqs = [np.random.uniform(size=d.shape[0]) for d in datas]

            # compute likelihoods, resample labels, and collect statistics
            allstats, allcounts = \
                resample_gmm_labels(stateseqs,datas,randseqs,sigmas,mus,logweights)

            for stats, counts, o in zip(allstats,allcounts,self.obs_distns):
                # resample gaussian params using statistics
                for s, c in zip(stats,o.components):
                    c.resample(stats=s,temperature=self.temperature)

                # resample mixture weights using counts
                o.weights.resample(counts=counts)

    @property
    def energy(self):
        from .util.temp import hsmm_gmm_energy

        datas = [s.data for s in self.states_list]
        stateseqs = [s.stateseq.astype('int32') for s in self.states_list]

        mus = np.array([[c.mu for c in d.components] for d in self.obs_distns])
        sigmas = np.array([[c.sigmas for c in d.components] for d in self.obs_distns])
        logweights = np.log(np.array([d.weights.weights for d in self.obs_distns]))

        randseqs = [np.random.uniform(size=d.shape[0]) for d in datas]

        return hsmm_gmm_energy(stateseqs,datas,randseqs,sigmas,mus,logweights)

    def get_sample(self):
        SAVETYPE = 'float16'

        stateseqs = [s.stateseq for s in self.states_list]

        mus = np.array([[c.mu for c in d.components]
            for d in self.obs_distns],dtype=SAVETYPE)
        sigmas = np.array([[c.sigmas for c in d.components]
            for d in self.obs_distns],dtype=SAVETYPE)
        weights = np.array([d.weights.weights for d in self.obs_distns])

        transitions = {k:d.full_trans_matrix for k,d in self.trans_distns.iteritems()}
        init_state_distns = {k:d.pi_0 for k,d in self.init_state_distns.iteritems()}

        return {
                'stateseqs':stateseqs,
                'mus':mus,
                'sigmas':sigmas,
                'weights':weights,
                'transitions':transitions,
                'init_state_distns':init_state_distns,
                }

    def set_sample(self,s):
        LOADTYPE = 'float64'

        for states, sample_stateseq in zip(self.states_list,s['stateseqs']):
            states.stateseq = sample_stateseq

        for d, mus, sigss, w in zip(self.obs_distns,s['mus'],s['sigmas'],s['weights']):
            d.weights.weights = w
            for c, mu, sigs in zip(d.components,mus,sigss):
                c.mu = mu.astype(LOADTYPE)
                c.sigmas = sigs.astype(LOADTYPE)

        for group_id in self.trans_distns:
            self.trans_distns[group_id].trans_matrix = s['transitions'][group_id]
            self.init_state_distns[group_id].pi_0 = s['init_state_distns'][group_id]

    def save_sample(self,filename):
        import gzip, cPickle
        sample = self.get_sample()
        rngstate = np.random.get_state()
        with gzip.open(filename,'w') as outfile:
            cPickle.dump((sample,rngstate),outfile,protocol=-1)

    def load_sample(self,filename):
        import gzip, cPickle
        with gzip.open(filename,'r') as infile:
            sample, rngstate = cPickle.load(infile)
        self.set_sample(sample)
        np.random.set_state(rngstate)

