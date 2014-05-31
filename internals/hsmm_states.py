from __future__ import division
import numpy as np
from numpy import newaxis as na
from numpy.random import random
from matplotlib import pyplot as plt
import abc, copy, warnings

from ..util.stats import sample_discrete, sample_discrete_from_log, sample_markov
from ..util.general import rle, top_eigenvector, rcumsum, cumsum
from ..util.profiling import line_profiled

PROFILING = False

from hmm_states import _StatesBase, _SeparateTransMixin, \
        HMMStatesPython, HMMStatesEigen

class HSMMStatesPython(_StatesBase):
    def __init__(self,model,right_censoring=True,left_censoring=False,trunc=None,
            stateseq=None,**kwargs):
        self.right_censoring = right_censoring
        self.left_censoring = left_censoring
        self.trunc = trunc

        super(HSMMStatesPython,self).__init__(model,stateseq=stateseq,**kwargs)

    ### properties for the outside world

    @property
    def stateseq(self):
        return self._stateseq

    @stateseq.setter
    def stateseq(self,stateseq):
        self._stateseq = stateseq
        self._stateseq_norep = None
        self._durations_censored = None

    @property
    def stateseq_norep(self):
        if self._stateseq_norep is None:
            self._stateseq_norep, self._durations_censored = rle(self.stateseq)
        return self._stateseq_norep

    @property
    def durations_censored(self):
        if self._durations_censored is None:
            self._stateseq_norep, self._durations_censored = rle(self.stateseq)
        return self._durations_censored

    @property
    def durations(self):
        durs = self.durations_censored.copy()
        if self.left_censoring:
            durs[0] = self.dur_distns[self.stateseq_norep[0]].rvs_given_greater_than(durs[0]-1)
        if self.right_censoring:
            durs[-1] = self.dur_distns[self.stateseq_norep[-1]].rvs_given_greater_than(durs[-1]-1)
        return durs

    @property
    def untrunc_slice(self):
        return slice(1 if self.left_censoring else 0, -1 if self.right_censoring else None)

    @property
    def trunc_slice(self):
        if self.left_censoring and self.right_censoring:
            return [0,-1] if len(self.stateseq_norep) > 1 else [0]
        elif self.left_censoring:
            return [0]
        elif self.right_censoring:
            return [1] if len(self.stateseq_norep) > 1 else [0]
        else:
            return []

    ### model parameter properties

    @property
    def pi_0(self):
        if not self.left_censoring:
            return self.model.init_state_distn.pi_0
        else:
            return self.model.left_censoring_init_state_distn.pi_0

    @property
    def dur_distns(self):
        return self.model.dur_distns

    @property
    def log_trans_matrix(self):
        if self._log_trans_matrix is None:
            self._log_trans_matrix = np.log(self.trans_matrix)
        return self._log_trans_matrix


    @property
    def mf_pi_0(self):
        return self.model.init_state_distn.exp_expected_log_init_state_distn

    @property
    def mf_log_trans_matrix(self):
        if self._mf_log_trans_matrix is None:
            self._mf_log_trans_matrix = np.log(self.mf_trans_matrix)
        return self._mf_log_trans_matrix

    @property
    def mf_trans_matrix(self):
        return np.maximum(self.model.trans_distn.exp_expected_log_trans_matrix,1e-3)

    ### generation

    # TODO rewrite this thing
    def generate_states(self):
        if self.left_censoring:
            raise NotImplementedError
        idx = 0
        nextstate_distr = self.pi_0
        A = self.trans_matrix

        stateseq = np.empty(self.T,dtype=np.int32)
        # durations = []

        while idx < self.T:
            # sample a state
            state = sample_discrete(nextstate_distr)
            # sample a duration for that state
            duration = self.dur_distns[state].rvs()
            # save everything
            # durations.append(duration)
            stateseq[idx:idx+duration] = state # this can run off the end, that's okay
            # set up next state distribution
            nextstate_distr = A[state,]
            # update index
            idx += duration

        self.stateseq = stateseq

    ### caching

    def clear_caches(self):
        self._aBl = self._mf_aBl = None
        self._aDl = self._mf_aDl = None
        self._aDsl = self._mf_aDsl = None
        self._log_trans_matrix = self._mf_log_trans_matrix = None
        self._normalizer = None
        super(HSMMStatesPython,self).clear_caches()

    ### array properties for homog model

    @property
    def aBl(self):
        if self._aBl is None:
            data = self.data
            aBl = self._aBl = np.empty((data.shape[0],self.num_states))
            for idx, obs_distn in enumerate(self.obs_distns):
                aBl[:,idx] = np.nan_to_num(obs_distn.log_likelihood(data))
        return self._aBl

    @property
    def aDl(self):
        if self._aDl is None:
            aDl = np.empty((self.T,self.num_states))
            possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
            for idx, dur_distn in enumerate(self.dur_distns):
                aDl[:,idx] = dur_distn.log_pmf(possible_durations)
            self._aDl = aDl
        return self._aDl

    @property
    def aDsl(self):
        if self._aDsl is None:
            aDsl = np.empty((self.T,self.num_states))
            possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
            for idx, dur_distn in enumerate(self.dur_distns):
                aDsl[:,idx] = dur_distn.log_sf(possible_durations)
            self._aDsl = aDsl
        return self._aDsl

    @property
    def mf_aBl(self):
        if self._mf_aBl is None:
            self._mf_aBl = aBl = np.empty((self.data.shape[0],self.num_states))
            for idx, o in enumerate(self.obs_distns):
                aBl[:,idx] = o.expected_log_likelihood(self.data)
        return self._mf_aBl

    @property
    def mf_aDl(self):
        if self._mf_aDl is None:
            self._mf_aDl = aDl = np.empty((self.T,self.num_states))
            possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
            for idx, dur_distn in enumerate(self.dur_distns):
                aDl[:,idx] = dur_distn.expected_log_pmf(possible_durations)
        return self._mf_aDl

    @property
    def mf_aDsl(self):
        if self._mf_aDsl is None:
            self._mf_aDsl = aDsl = np.empty((self.T,self.num_states))
            possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
            for idx, dur_distn in enumerate(self.dur_distns):
                aDsl[:,idx] = dur_distn.expected_log_sf(possible_durations)
        return self._mf_aDsl

    # @property
    # def betal(self):
    #     if self._betal is None:
    #         self._betal = np.empty((self.Tblock,self.num_states))
    #     return self._betal

    # @property
    # def betastarl(self):
    #     if self._betastarl is None:
    #         self._betastarl = np.empty((self.Tblock,self.num_states))
    #     return self._betastarl

    # @property
    # def alphal(self):
    #     if self._alphal is None:
    #         self._alphal = np.empty((self.Tblock,self.num_states))
    #     return self._alphal

    # @property
    # def alphastarl(self):
    #     if self._alphastarl is None:
    #         self._alphastarl = np.empty((self.Tblock,self.num_states))
    #     return self._alphastarl

    ### NEW message passing, with external pure functions

    def messages_forwards(self):
        alphal, alphastarl, _ = hsmm_messages_forwards_log(
                self.trans_potentials,
                np.log(self.pi_0),
                self.reverse_cumulative_obs_potentials,
                self.reverse_dur_potentials,
                self.reverse_dur_survival_potentials,
                np.empty((self.T,self.num_states)),np.empty((self.T,self.num_states)))
        return alphal, alphastarl

    def messages_backwards(self):
        betal, betastarl, loglike = hsmm_messages_backwards_log(
                self.trans_potentials,
                np.log(self.pi_0),
                self.cumulative_obs_potentials,
                self.dur_potentials,
                self.dur_survival_potentials,
                np.empty((self.T,self.num_states)),np.empty((self.T,self.num_states)))
        self._normalizer = loglike
        return betal, betastarl



    def log_likelihood(self):
        if self._normalizer is None:
            self.messages_backwards() # NOTE: sets self._normalizer
        return self._normalizer

    def get_vlb(self):
        if self._normalizer is None:
            self.meanfieldupdate() # a bit excessive...
        return self._normalizer

    # forwards messages potentials

    def trans_potentials(self,t):
        return self.log_trans_matrix

    def cumulative_obs_potentials(self,t):
        stop = None if self.trunc is None else min(self.T,t+self.trunc)
        return np.cumsum(self.aBl[t:stop],axis=0)

    def dur_potentials(self,t):
        stop = self.T-t if self.trunc is None else min(self.T-t,self.trunc)
        return self.aDl[:stop]

    def dur_survival_potentials(self,t):
        return self.aDsl[self.T-t -1] if (self.trunc is None or self.T-t > self.trunc) \
                else -np.inf

    # backwards messages potentials

    def reverse_cumulative_obs_potentials(self,t):
        start = 0 if self.trunc is None else max(0,t-self.trunc+1)
        return rcumsum(self.aBl[start:t+1])

    def reverse_dur_potentials(self,t):
        stop = t+1 if self.trunc is None else min(t+1,self.trunc)
        return self.aDl[:stop][::-1]

    def reverse_dur_survival_potentials(self,t):
        # NOTE: untested, unused without left-censoring
        return self.aDsl[t] if (self.trunc is None or t+1 < self.trunc) \
                else -np.inf

    # mean field messages potentials

    def mf_trans_potentials(self,t):
        return self.mf_log_trans_matrix

    def mf_cumulative_obs_potentials(self,t):
        stop = None if self.trunc is None else min(self.T,t+self.trunc)
        return np.cumsum(self.mf_aBl[t:stop],axis=0)

    def mf_reverse_cumulative_obs_potentials(self,t):
        start = 0 if self.trunc is None else max(0,t-self.trunc+1)
        return rcumsum(self.mf_aBl[start:t+1])

    def mf_dur_potentials(self,t):
        stop = self.T-t if self.trunc is None else min(self.T-t,self.trunc)
        return self.mf_aDl[:stop]

    def mf_reverse_dur_potentials(self,t):
        stop = t+1 if self.trunc is None else min(t+1,self.trunc)
        return self.mf_aDl[:stop][::-1]

    def mf_dur_survival_potentials(self,t):
        return self.mf_aDsl[self.T-t -1] if (self.trunc is None or self.T-t > self.trunc) \
                else -np.inf

    def mf_reverse_dur_survival_potentials(self,t):
        # NOTE: untested, unused without left-censoring
        return self.mf_aDsl[t] if (self.trunc is None or t+1 < self.trunc) \
                else -np.inf

    ### Gibbs sampling

    def resample(self):
        betal, betastarl = self.messages_backwards()
        self.sample_forwards(betal,betastarl)

    def copy_sample(self,newmodel):
        new = super(HSMMStatesPython,self).copy_sample(newmodel)
        return new

    def sample_forwards(self,betal,betastarl):
        self.stateseq, _ = hsmm_sample_forwards_log(
                self.trans_potentials,
                np.log(self.pi_0),
                self.cumulative_obs_potentials,
                self.dur_potentials,
                self.dur_survival_potentials,
                betal, betastarl)
        return self.stateseq

    ### Viterbi

    def Viterbi(self):
        self.stateseq = hsmm_maximizing_assignment(
            self.num_states, self.T,
            self.trans_potentials, np.log(self.pi_0),
            self.cumulative_obs_potentials,
            self.reverse_cumulative_obs_potentials,
            self.dur_potentials, self.dur_survival_potentials)

    def mf_Viterbi(self):
        self.stateseq = hsmm_maximizing_assignment(
            self.num_states, self.T,
            self.mf_trans_potentials, np.log(self.mf_pi_0),
            self.mf_cumulative_obs_potentials,
            self.mf_reverse_cumulative_obs_potentials,
            self.mf_dur_potentials, self.mf_dur_survival_potentials)

    ### EM

    # these two methods just call _expected_statistics with the right stuff

    def E_step(self):
        self.clear_caches()
        self.all_expected_stats = self._expected_statistics(
                self.trans_potentials, np.log(self.pi_0),
                self.cumulative_obs_potentials, self.reverse_cumulative_obs_potentials,
                self.dur_potentials, self.reverse_dur_potentials,
                self.dur_survival_potentials, self.reverse_dur_survival_potentials)
        self.stateseq = self.expected_states.argmax(1) # for plotting

    def meanfieldupdate(self):
        self.clear_caches()
        self.all_expected_stats = self._expected_statistics(
                self.mf_trans_potentials, np.log(self.mf_pi_0),
                self.mf_cumulative_obs_potentials, self.mf_reverse_cumulative_obs_potentials,
                self.mf_dur_potentials, self.mf_reverse_dur_potentials,
                self.mf_dur_survival_potentials, self.mf_reverse_dur_survival_potentials)
        self.stateseq = self.expected_states.argmax(1) # for plotting

    @property
    def all_expected_stats(self):
        return self.expected_states, self.expected_transcounts, \
                self.expected_durations, self._normalizer

    @all_expected_stats.setter
    def all_expected_stats(self,vals):
        self.expected_states, self.expected_transcounts, \
                self.expected_durations, self._normalizer = vals

    # here's the real work

    @line_profiled
    def _expected_statistics(self,
            trans_potentials, initial_state_potential,
            cumulative_obs_potentials, reverse_cumulative_obs_potentials,
            dur_potentials, reverse_dur_potentials,
            dur_survival_potentials, reverse_dur_survival_potentials):

        alphal, alphastarl, _ = hsmm_messages_forwards_log(
                trans_potentials,
                initial_state_potential,
                reverse_cumulative_obs_potentials,
                reverse_dur_potentials,
                reverse_dur_survival_potentials,
                np.empty((self.T,self.num_states)),np.empty((self.T,self.num_states)))

        betal, betastarl, normalizer = hsmm_messages_backwards_log(
                trans_potentials,
                initial_state_potential,
                cumulative_obs_potentials,
                dur_potentials,
                dur_survival_potentials,
                np.empty((self.T,self.num_states)),np.empty((self.T,self.num_states)))

        expected_states = self._expected_states(
                alphal, betal, alphastarl, betastarl, normalizer)

        expected_transitions = self._expected_transitions(
                alphal, betastarl, trans_potentials, normalizer) # TODO assumes homog trans

        expected_durations = self._expected_durations(
                dur_potentials,cumulative_obs_potentials,
                alphastarl, betal, normalizer)

        return expected_states, expected_transitions, expected_durations, normalizer

    def _expected_states(self,alphal,betal,alphastarl,betastarl,normalizer):
        gammal = alphal + betal
        gammastarl = alphastarl + betastarl
        gamma = np.exp(gammal - normalizer)
        gammastar = np.exp(gammastarl - normalizer)

        assert gamma.min() > 0.-1e-3 and gamma.max() < 1.+1e-3
        assert gammastar.min() > 0.-1e-3 and gammastar.max() < 1.+1e-3

        expected_states = \
            (gammastar - np.vstack((np.zeros(gamma.shape[1]),gamma[:-1]))).cumsum(0)

        assert not np.isnan(expected_states).any()
        assert expected_states.min() > 0.-1e-3 and expected_states.max() < 1 + 1e-3
        assert np.allclose(expected_states.sum(1),1.,atol=1e-2)

        expected_states = np.maximum(0.,expected_states)
        expected_states /= expected_states.sum(1)[:,na]

        # TODO break this out into a function
        self._changepoint_probs = gammastar.sum(1)

        return expected_states

    def _expected_transitions(self,alphal,betastarl,trans_potentials,normalizer):
        # TODO assumes homog trans; otherwise, need a loop
        Al = trans_potentials(0)
        transl = alphal[:-1,:,na] + betastarl[1:,na,:] + Al[na,...]
        transl -= normalizer
        expected_transcounts = np.exp(transl).sum(0)
        return expected_transcounts

    def _expected_durations(self,
            dur_potentials,cumulative_obs_potentials,
            alphastarl,betal,normalizer):
        if self.trunc is not None:
            raise NotImplementedError, "_expected_durations can't handle trunc"
        T = self.T
        logpmfs = -np.inf*np.ones_like(alphastarl)
        errs = np.seterr(invalid='ignore')
        for t in xrange(T):
            np.logaddexp(dur_potentials(t) + alphastarl[t] + betal[t:] +
                    cumulative_obs_potentials(t) - normalizer,
                    logpmfs[:T-t], out=logpmfs[:T-t])
        np.seterr(**errs)
        expected_durations = np.exp(logpmfs.T)

        return expected_durations


    ### plotting

    def plot(self,colors_dict=None,**kwargs):
        # TODO almost identical to HMM.plot, but with reference to
        # stateseq_norep
        from matplotlib import pyplot as plt
        X,Y = np.meshgrid(np.hstack((0,self.durations_censored.cumsum())),(0,1))

        if colors_dict is not None:
            C = np.array([[colors_dict[state] for state in self.stateseq_norep]])
        else:
            C = self.stateseq_norep[na,:]

        plt.pcolor(X,Y,C,vmin=0,vmax=1,**kwargs)
        plt.ylim((0,1))
        plt.xlim((0,self.T))
        plt.yticks([])
        plt.title('State Sequence')


# TODO call this 'time homog'
class HSMMStatesEigen(HSMMStatesPython):
    # NOTE: the methods in this class only work with iid emissions (i.e. without
    # overriding methods like cumulative_likelihood_block)

    def messages_backwards(self):
        # NOTE: np.maximum calls are because the C++ code doesn't do
        # np.logaddexp(-inf,-inf) = -inf, it likes nans instead
        from hsmm_messages_interface import messages_backwards_log
        betal, betastarl = messages_backwards_log(
                np.maximum(self.trans_matrix,1e-50),self.aBl,np.maximum(self.aDl,-1000000),
                self.aDsl,np.empty_like(self.aBl),np.empty_like(self.aBl),
                self.right_censoring,self.trunc if self.trunc is not None else self.T)
        assert not np.isnan(betal).any()
        assert not np.isnan(betastarl).any()

        if not self.left_censoring:
            self._normalizer = np.logaddexp.reduce(np.log(self.pi_0) + betastarl[0])
        else:
            raise NotImplementedError

        return betal, betastarl

    def messages_backwards_python(self):
        return super(HSMMStatesEigen,self).messages_backwards()

    def sample_forwards(self,betal,betastarl):
        from hsmm_messages_interface import sample_forwards_log
        if self.left_censoring:
            raise NotImplementedError
        caBl = np.vstack((np.zeros(betal.shape[1]),np.cumsum(self.aBl[:-1],axis=0)))
        self.stateseq = sample_forwards_log(
                self.trans_matrix,caBl,self.aDl,self.pi_0,betal,betastarl,
                np.empty(betal.shape[0],dtype='int32'))
        assert not (0 == self.stateseq).all()

    def sample_forwards_python(self,betal,betastarl):
        return super(HSMMStatesEigen,self).sample_forwards(betal,betastarl)

    @staticmethod
    def _resample_multiple(states_list):
        from hsmm_messages_interface import resample_log_multiple
        if len(states_list) > 0:
            Ts = [s.T for s in states_list]
            longest = np.argmax(Ts)
            stateseqs = [np.empty(T,dtype=np.int32) for T in Ts]
            loglikes = resample_log_multiple(
                    states_list[0].trans_matrix,
                    states_list[0].pi_0,
                    states_list[longest].aDl,
                    states_list[longest].aDsl,
                    [s.aBl for s in states_list],
                    np.array([s.right_censoring for s in states_list],dtype=np.int32),
                    np.array([s.trunc for s in states_list],dtype=np.int32),
                    stateseqs,
                    )
            for s, loglike, stateseq in zip(states_list,loglikes,stateseqs):
                s._normalizer = loglike
                s.stateseq = stateseq


class GeoHSMMStates(HSMMStatesPython):
    def resample(self):
        alphan, self._normalizer = HMMStatesEigen._messages_forwards_normalized(
                self.hmm_trans_matrix,
                self.pi_0,
                self.aBl)
        self.stateseq = HMMStatesEigen._sample_backwards_normalized(
                alphan,
                self.hmm_trans_matrix.T.copy())

    @property
    def hmm_trans_matrix(self):
        A = self.trans_matrix.copy()
        ps = np.array([d.p for d in self.dur_distns])

        A *= ps[:,na]
        A.flat[::A.shape[0]+1] = 1-ps
        assert np.allclose(1.,A.sum(1))

        return A


class HSMMStatesPossibleChangepoints(HSMMStatesPython):
    def __init__(self,model,data,changepoints,**kwargs):
        self.changepoints = changepoints
        self.segmentstarts = np.array([start for start,stop in changepoints],dtype=np.int32)
        self.segmentlens = np.array([stop-start for start,stop in changepoints],dtype=np.int32)

        assert all(l > 0 for l in self.segmentlens)
        assert sum(self.segmentlens) == data.shape[0]
        assert self.changepoints[0][0] == 0 and self.changepoints[-1][-1] == data.shape[0]

        self._kwargs = dict(self._kwargs,changepoints=changepoints) \
                if hasattr(self,'_kwargs') else dict(changepoints=changepoints)

        super(HSMMStatesPossibleChangepoints,self).__init__(
                model,T=len(changepoints),data=data,**kwargs)

    def clear_caches(self):
        self._aBBl = self._mf_aBBl = None
        super(HSMMStatesPossibleChangepoints,self).clear_caches()

    ### properties for the outside world

    @property
    def stateseq(self):
        return self.blockstateseq.repeat(self.segmentlens)

    @stateseq.setter
    def stateseq(self,stateseq):
        self._stateseq_norep = None
        self._durations_censored = None

        assert len(stateseq) == self.Tblock or len(stateseq) == self.Tfull
        if len(stateseq) == self.Tblock:
            self.blockstateseq = stateseq
        else:
            self.blockstateseq = stateseq[self.segmentstarts]

    # @property
    # def stateseq_norep(self):
    #     if self._stateseq_norep is None:
    #         self._stateseq_norep, self._repeats_censored = rle(self.stateseq)
    #     self._durations_censored = self._repeats_censored.repeat(self.segmentlens)
    #     return self._stateseq_norep

    # @property
    # def durations_censored(self):
    #     if self._durations_censored is None:
    #         self._stateseq_norep, self._repeats_censored = rle(self.stateseq)
    #     self._durations_censored = self._repeats_censored.repeat(self.segmentlens)
    #     return self._durations_censored

    ### model parameter properties

    @property
    def Tblock(self):
        return len(self.changepoints)

    @property
    def Tfull(self):
        return self.data.shape[0]

    @property
    def aBBl(self):
        if self._aBBl is None:
            aBl = self.aBl
            aBBl = self._aBBl = np.empty((self.Tblock,self.num_states))
            for idx, (start,stop) in enumerate(self.changepoints):
                aBBl[idx] = aBl[start:stop].sum(0)
        return self._aBBl

    @property
    def mf_aBBl(self):
        if self._mf_aBBl is None:
            aBl = self.mf_aBl
            aBBl = self._mf_aBBl = np.empty((self.Tblock,self.num_states))
            for idx, (start,stop) in enumerate(self.changepoints):
                aBBl[idx] = aBl[start:stop].sum(0)
        return self._mf_aBBl

    # TODO reduce repetition with parent in next 4 props

    @property
    def aDl(self):
        # just like parent aDl, except we use Tfull
        if self._aDl is None:
            aDl = np.empty((self.Tfull,self.num_states))
            possible_durations = np.arange(1,self.Tfull + 1,dtype=np.float64)
            for idx, dur_distn in enumerate(self.dur_distns):
                aDl[:,idx] = dur_distn.log_pmf(possible_durations)
            self._aDl = aDl
        return self._aDl

    @property
    def aDsl(self):
        # just like parent aDl, except we use Tfull
        if self._aDsl is None:
            aDsl = np.empty((self.Tfull,self.num_states))
            possible_durations = np.arange(1,self.Tfull + 1,dtype=np.float64)
            for idx, dur_distn in enumerate(self.dur_distns):
                aDsl[:,idx] = dur_distn.log_sf(possible_durations)
            self._aDsl = aDsl
        return self._aDsl

    @property
    def mf_aDl(self):
        # just like parent aDl, except we use Tfull
        if self._aDl is None:
            aDl = np.empty((self.Tfull,self.num_states))
            possible_durations = np.arange(1,self.Tfull + 1,dtype=np.float64)
            for idx, dur_distn in enumerate(self.dur_distns):
                aDl[:,idx] = dur_distn.expected_log_pmf(possible_durations)
            self._aDl = aDl
        return self._aDl

    @property
    def mf_aDsl(self):
        # just like parent aDl, except we use Tfull
        if self._aDsl is None:
            aDsl = np.empty((self.Tfull,self.num_states))
            possible_durations = np.arange(1,self.Tfull + 1,dtype=np.float64)
            for idx, dur_distn in enumerate(self.dur_distns):
                aDsl[:,idx] = dur_distn.expected_log_sf(possible_durations)
            self._aDsl = aDsl
        return self._aDsl

    # @property
    # def betal(self):
    #     if self._betal is None:
    #         self._betal = np.empty((self.Tblock,self.num_states))
    #     return self._betal

    # @property
    # def betastarl(self):
    #     if self._betastarl is None:
    #         self._betastarl = np.empty((self.Tblock,self.num_states))
    #     return self._betastarl

    # @property
    # def alphal(self):
    #     if self._alphal is None:
    #         self._alphal = np.empty((self.Tblock,self.num_states))
    #     return self._alphal

    # @property
    # def alphastarl(self):
    #     if self._alphastarl is None:
    #         self._alphastarl = np.empty((self.Tblock,self.num_states))
    #     return self._alphastarl

    ### message passing

    # TODO caching
    # TODO trunc

    # TODO wrap the duration stuff into single functions. reduces passing
    # around, reduces re-computation in this case

    # backwards messages potentials

    def cumulative_obs_potentials(self,tblock):
        return self.aBBl[tblock:].cumsum(0)[:self.trunc]

    def dur_potentials(self,tblock):
        possible_durations = self.segmentlens[tblock:].cumsum()[:self.trunc]
        return self.aDl[possible_durations -1]

    def dur_survival_potentials(self,tblock):
        # return -np.inf # for testing against other implementation
        max_dur = self.segmentlens[tblock:].cumsum()[:self.trunc][-1]
        return self.aDsl[max_dur -1]

    # forwards messages potentials

    def reverse_cumulative_obs_potentials(self,tblock):
        return rcumsum(self.aBBl[:tblock+1])\
                [-self.trunc if self.trunc is not None else None:]

    def reverse_dur_potentials(self,tblock):
        possible_durations = rcumsum(self.segmentlens[:tblock+1])\
                [-self.trunc if self.trunc is not None else None:]
        return self.aDl[possible_durations -1]

    def reverse_dur_survival_potentials(self,tblock):
        # NOTE: untested, unused
        max_dur = rcumsum(self.segmentlens[:tblock+1])\
                [-self.trunc if self.trunc is not None else None:][0]
        return self.aDsl[max_dur -1]

    # mean field messages potentials

    def mf_cumulative_obs_potentials(self,tblock):
        return self.mf_aBBl[tblock:].cumsum(0)[:self.trunc]

    def mf_reverse_cumulative_obs_potentials(self,tblock):
        return rcumsum(self.mf_aBBl[:tblock+1])\
                [-self.trunc if self.trunc is not None else None:]

    def mf_dur_potentials(self,tblock):
        possible_durations = self.segmentlens[tblock:].cumsum()[:self.trunc]
        return self.mf_aDl[possible_durations -1]

    def mf_reverse_dur_potentials(self,tblock):
        possible_durations = rcumsum(self.segmentlens[:tblock+1])\
                [-self.trunc if self.trunc is not None else None:]
        return self.mf_aDl[possible_durations -1]

    def mf_dur_survival_potentials(self,tblock):
        max_dur = self.segmentlens[tblock:].cumsum()[:self.trunc][-1]
        return self.mf_aDsl[max_dur -1]

    def mf_reverse_dur_survival_potentials(self,tblock):
        max_dur = rcumsum(self.segmentlens[:tblock+1])\
                [-self.trunc if self.trunc is not None else None:][0]
        return self.mf_aDsl[max_dur -1]


    ### generation

    def generate_states(self):
        if self.left_censoring:
            raise NotImplementedError
        Tblock = len(self.changepoints)
        blockstateseq = self.blockstateseq = np.zeros(Tblock,dtype=np.int32)

        tblock = 0
        nextstate_distr = self.pi_0
        A = self.trans_matrix

        while tblock < Tblock:
            # sample the state
            state = sample_discrete(nextstate_distr)

            # compute possible duration info (indep. of state)
            possible_durations = self.segmentlens[tblock:].cumsum()

            # compute the pmf over those steps
            durprobs = self.dur_distns[state].pmf(possible_durations)
            # TODO censoring: the last possible duration isn't quite right
            durprobs /= durprobs.sum()

            # sample it
            blockdur = sample_discrete(durprobs) + 1

            # set block sequence
            blockstateseq[tblock:tblock+blockdur] = state

            # set up next iteration
            tblock += blockdur
            nextstate_distr = A[state]

        self._stateseq_norep = None
        self._durations_censored = None

    def generate(self):
        raise NotImplementedError

    def plot(self,*args,**kwargs):
        super(HSMMStatesPossibleChangepoints,self).plot(*args,**kwargs)
        plt.xlim((0,self.Tfull))

    # TODO E step refactor
    # TODO trunc

    def _expected_states(self,*args,**kwargs):
        expected_states = super(HSMMStatesPossibleChangepoints,self)._expected_states(*args,**kwargs)
        return expected_states.repeat(self.segmentlens,axis=0)

    def _expected_durations(self,
            dur_potentials,cumulative_obs_potentials,
            alphastarl,betal,normalizer):
        logpmfs = -np.inf*np.ones((self.Tfull,alphastarl.shape[1]))
        errs = np.seterr(invalid='ignore') # logaddexp(-inf,-inf)
        # TODO censoring not handled correctly here
        for tblock in xrange(self.Tblock):
            possible_durations = self.segmentlens[tblock:].cumsum()[:self.trunc]
            logpmfs[possible_durations -1] = np.logaddexp(
                    dur_potentials(tblock) + alphastarl[tblock]
                    + betal[tblock:tblock+self.trunc if self.trunc is not None else None]
                    + cumulative_obs_potentials(tblock) - normalizer,
                    logpmfs[possible_durations -1])
        np.seterr(**errs)
        return np.exp(logpmfs.T)

class HSMMStatesPossibleChangepointsSeparateTrans(
        _SeparateTransMixin,
        HSMMStatesPossibleChangepoints):
    pass


# NOTE: this class is purely for testing HSMM messages
class _HSMMStatesEmbedding(HSMMStatesPython,HMMStatesPython):

    @property
    def hmm_aBl(self):
        return np.repeat(self.aBl,self.T,axis=1)

    @property
    def hmm_backwards_pi_0(self):
        if not self.left_censoring:
            aD = np.exp(self.aDl)
            aD[-1] = [np.exp(distn.log_sf(self.T-1)) for distn in self.dur_distns]
            assert np.allclose(aD.sum(0),1.)
            pi_0 = (self.pi_0 *  aD[::-1,:]).T.ravel()
            assert np.isclose(pi_0.sum(),1.)
            return pi_0
        else:
            raise NotImplementedError

    @property
    def hmm_backwards_trans_matrix(self):
        # TODO construct this as a csr
        blockcols = []
        aD = np.exp(self.aDl)
        aDs = np.array([np.exp(distn.log_sf(self.T-1)) for distn in self.dur_distns])
        for j in xrange(self.num_states):
            block = np.zeros((self.T,self.T))
            block[-1,0] = aDs[j]
            block[-1,1:] = aD[self.T-2::-1,j]
            blockcol = np.kron(self.trans_matrix[:,na,j],block)
            blockcol[j*self.T:(j+1)*self.T] = np.eye(self.T,k=1)
            blockcols.append(blockcol)
        return np.hstack(blockcols)

    @property
    def hmm_forwards_pi_0(self):
        if not self.left_censoring:
            out = np.zeros((self.num_states,self.T))
            out[:,0] = self.pi_0
            return out.ravel()
        else:
            raise NotImplementedError

    @property
    def hmm_forwards_trans_matrix(self):
        # TODO construct this as a csc
        blockrows = []
        aD = np.exp(self.aDl)
        aDs = np.hstack([np.exp(distn.log_sf(np.arange(self.T)))[:,na]
            for distn in self.dur_distns])
        for i in xrange(self.num_states):
            block = np.zeros((self.T,self.T))
            block[:,0] = aD[:self.T,i] / aDs[:self.T,i]
            blockrow = np.kron(self.trans_matrix[i],block)
            blockrow[:self.T,i*self.T:(i+1)*self.T] = \
                    np.diag(1-aD[:self.T-1,i]/aDs[:self.T-1,i],k=1)
            blockrow[-1,(i+1)*self.T-1] = 1-aD[self.T-1,i]/aDs[self.T-1,i]
            blockrows.append(blockrow)
        return np.vstack(blockrows)


    def messages_forwards_normalized_hmm(self):
        return HMMStatesPython._messages_forwards_normalized(
                self.hmm_forwards_trans_matrix,self.hmm_forwards_pi_0,self.hmm_aBl)

    def messages_backwards_normalized_hmm(self):
        return HMMStatesPython._messages_backwards_normalized(
                self.hmm_backwards_trans_matrix,self.hmm_backwards_pi_0,self.hmm_aBl)


    def messages_forwards_log_hmm(self):
        return HMMStatesPython._messages_forwards_log(
                self.hmm_forwards_trans_matrix,self.hmm_forwards_pi_0,self.hmm_aBl)

    def log_likelihood_forwards_hmm(self):
        alphal = self.messages_forwards_log_hmm()
        if self.right_censoring:
            return np.logaddexp.reduce(alphal[-1])
        else:
            # TODO should dot against deltas instead of ones
            raise NotImplementedError


    def messages_backwards_log_hmm(self):
        return HMMStatesPython._messages_backwards_log(
            self.hmm_backwards_trans_matrix,self.hmm_aBl)

    def log_likelihood_backwards_hmm(self):
        betal = self.messages_backwards_log_hmm()
        return np.logaddexp.reduce(np.log(self.hmm_backwards_pi_0) + self.hmm_aBl[0] + betal[0])


    def messages_backwards_log(self,*args,**kwargs):
        raise NotImplementedError # NOTE: this hmm method shouldn't be called this way

    def messages_fowrards_log(self,*args,**kwargs):
        raise NotImplementedError # NOTE: this hmm method shouldn't be called this way

    def messages_backwards_normalized(self,*args,**kwargs):
        raise NotImplementedError # NOTE: this hmm method shouldn't be called this way

    def messages_forwards_normalized(self,*args,**kwargs):
        raise NotImplementedError # NOTE: this hmm method shouldn't be called this way


### HSMM messages

def hsmm_messages_backwards_log(
    trans_potentials, initial_state_potential,
    cumulative_obs_potentials, dur_potentials, dur_survival_potentials,
    betal, betastarl,
    left_censoring=False, right_censoring=True):

    T, _ = betal.shape

    betal[-1] = 0.
    for t in xrange(T-1,-1,-1):
        cB = cumulative_obs_potentials(t)
        np.logaddexp.reduce(betal[t:t+cB.shape[0]] + cB + dur_potentials(t),
                axis=0, out=betastarl[t])
        if right_censoring:
            np.logaddexp(betastarl[t], cB[-1] + dur_survival_potentials(t),
                    out=betastarl[t])
        np.logaddexp.reduce(betastarl[t] + trans_potentials(t-1),
                axis=1, out=betal[t-1])
    betal[-1] = 0. # overwritten on last iteration

    if not left_censoring:
        normalizer = np.logaddexp.reduce(initial_state_potential + betastarl[0])
    else:
        raise NotImplementedError

    return betal, betastarl, normalizer

def hsmm_messages_forwards_log(
    trans_potential, initial_state_potential,
    reverse_cumulative_obs_potentials, reverse_dur_potentials, reverse_dur_survival_potentials,
    alphal, alphastarl,
    left_censoring=False, right_censoring=True):

    T, _ = alphal.shape

    alphastarl[0] = initial_state_potential
    for t in xrange(T-1):
        cB = reverse_cumulative_obs_potentials(t)
        np.logaddexp.reduce(alphastarl[t+1-cB.shape[0]:t+1] + cB + reverse_dur_potentials(t),
                axis=0, out=alphal[t])
        if left_censoring:
            raise NotImplementedError
        np.logaddexp.reduce(alphal[t][:,na] + trans_potential(t),
                axis=0, out=alphastarl[t+1])
    t = T-1
    cB = reverse_cumulative_obs_potentials(t)
    np.logaddexp.reduce(alphastarl[t+1-cB.shape[0]:t+1] + cB + reverse_dur_potentials(t),
            axis=0, out=alphal[t])

    if not right_censoring:
        normalizer = np.logaddexp.reduce(alphal[t])
    else:
        normalizer = None # TODO

    return alphal, alphastarl, normalizer


# TODO test with trunc
def hsmm_sample_forwards_log(
    trans_potentials, initial_state_potential,
    cumulative_obs_potentials, dur_potentials, dur_survival_potentails,
    betal, betastarl,
    left_censoring=False, right_censoring=True):

    T, _ = betal.shape
    stateseq = np.empty(T,dtype=np.int)
    durations = []

    t = 0

    if left_censoring:
        raise NotImplementedError
    else:
        nextstate_unsmoothed = initial_state_potential

    while t < T:
        ## sample the state
        nextstate_distn_log = nextstate_unsmoothed + betastarl[t]
        nextstate_distn = np.exp(nextstate_distn_log - np.logaddexp.reduce(nextstate_distn_log))
        assert nextstate_distn.sum() > 0
        state = sample_discrete(nextstate_distn)

        ## sample the duration
        dur_logpmf = dur_potentials(t)[:,state]
        obs = cumulative_obs_potentials(t)[:,state]
        durprob = np.random.random()

        dur = 0 # NOTE: always incremented at least once
        while durprob > 0 and dur < dur_logpmf.shape[0] and t+dur < T:
            p_d = np.exp(dur_logpmf[dur] + obs[dur]
                    + betal[t+dur,state] - betastarl[t,state])

            assert not np.isnan(p_d)
            durprob -= p_d
            dur += 1

        stateseq[t:t+dur] = state
        durations.append(dur)

        t += dur
        nextstate_log_distn = trans_potentials(t)[state]

    return stateseq, durations

def hsmm_maximizing_assignment(
    N, T,
    trans_potentials, initial_state_potential,
    cumulative_obs_potentials, reverse_cumulative_obs_potentials,
    dur_potentials, dur_survival_potentials,
    left_censoring=False, right_censoring=True):

    beta_scores, beta_args = np.empty((T,N)), np.empty((T,N),dtype=np.int)
    betastar_scores, betastar_args = np.empty((T,N)), np.empty((T,N),dtype=np.int)

    beta_scores[-1] = 0.
    for t in xrange(T-1,-1,-1):
        cB = cumulative_obs_potentials(t)

        vals = beta_scores[t:t+cB.shape[0]] + cB + dur_potentials(t)
        if right_censoring:
            vals = np.vstack((vals,cB[-1] + dur_survival_potentials(t)))

        vals.max(axis=0,out=betastar_scores[t])
        vals.argmax(axis=0,out=betastar_args[t])

        vals = betastar_scores[t] + trans_potentials(t-1)

        vals.max(axis=1,out=beta_scores[t-1])
        vals.argmax(axis=1,out=beta_args[t-1])
    beta_scores[-1] = 0.

    stateseq = np.empty(T,dtype=np.int)

    t = 0
    state = (betastar_scores[t] + initial_state_potential).argmax()
    dur = betastar_args[t,state]
    stateseq[t:t+dur] = state
    t += dur
    while t < T:
        state = beta_args[t-1,state]
        dur = betastar_args[t,state] + 1
        stateseq[t:t+dur] = state
        t += dur

    return stateseq

