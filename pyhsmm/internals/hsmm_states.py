from __future__ import division
from builtins import range, map
import numpy as np
from numpy import newaxis as na
from scipy.special import logsumexp

from pyhsmm.util.stats import sample_discrete
from pyhsmm.util.general import rle, rcumsum, cumsum

from . import hmm_states
from .hmm_states import _StatesBase, _SeparateTransMixin, \
    HMMStatesPython, HMMStatesEigen


class HSMMStatesPython(_StatesBase):
    def __init__(self,model,right_censoring=True,left_censoring=False,trunc=None,
            stateseq=None,**kwargs):
        self.right_censoring = right_censoring
        self.left_censoring = left_censoring
        self.trunc = trunc

        self._kwargs = dict(
            self._kwargs,trunc=trunc,
            left_censoring=left_censoring,right_censoring=right_censoring)

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

    # TODO make this generic, just call hsmm_sample_forwards_log with zero
    # potentials?
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
            T = self.data.shape[0]
            self._mf_aBl = aBl = np.empty((self.data.shape[0],self.num_states))
            for idx, o in enumerate(self.obs_distns):
                aBl[:,idx] = o.expected_log_likelihood(self.data).reshape((T,))
            aBl[np.isnan(aBl).any(1)] = 0.
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

    def get_vlb(self, states_last_updated=True):
        # TODO like HMM.get_vlb, allow computing vlb even when this factor isn't
        # the most recently updated
        assert states_last_updated
        if self._normalizer is None:
            self.meanfieldupdate()  # a bit excessive...
        return self._normalizer

    # forwards messages potentials

    def trans_potentials(self,t):
        return self.log_trans_matrix

    def cumulative_obs_potentials(self,t):
        stop = None if self.trunc is None else min(self.T,t+self.trunc)
        return np.cumsum(self.aBl[t:stop],axis=0), 0.

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
        return np.cumsum(self.mf_aBl[t:stop],axis=0), 0.

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

    def meanfieldupdate(self):
        self.clear_caches()
        self.all_expected_stats = self._expected_statistics(
                self.mf_trans_potentials, np.log(self.mf_pi_0),
                self.mf_cumulative_obs_potentials, self.mf_reverse_cumulative_obs_potentials,
                self.mf_dur_potentials, self.mf_reverse_dur_potentials,
                self.mf_dur_survival_potentials, self.mf_reverse_dur_survival_potentials)

    @property
    def all_expected_stats(self):
        return self.expected_states, self.expected_transcounts, \
                self.expected_durations, self._normalizer

    @all_expected_stats.setter
    def all_expected_stats(self,vals):
        self.expected_states, self.expected_transcounts, \
                self.expected_durations, self._normalizer = vals
        self.stateseq = self.expected_states.argmax(1).astype('int32') # for plotting

    def init_meanfield_from_sample(self):
        self.expected_states = \
            np.hstack([(self.stateseq == i).astype('float64')[:,na]
                for i in range(self.num_states)])

        from pyhsmm.util.general import count_transitions
        self.expected_transcounts = \
            count_transitions(self.stateseq_norep,minlength=self.num_states)

        self.expected_durations = expected_durations = \
                np.zeros((self.num_states,self.T))
        for state in range(self.num_states):
            expected_durations[state] += \
                np.bincount(
                    self.durations_censored[self.stateseq_norep == state],
                    minlength=self.T)[:self.T]

    # here's the real work

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
            raise NotImplementedError("_expected_durations can't handle trunc")
        T = self.T
        logpmfs = -np.inf*np.ones_like(alphastarl)
        errs = np.seterr(invalid='ignore')
        for t in range(T):
            cB, offset = cumulative_obs_potentials(t)
            np.logaddexp(dur_potentials(t) + alphastarl[t] + betal[t:] +
                    cB - (normalizer + offset),
                    logpmfs[:T-t], out=logpmfs[:T-t])
        np.seterr(**errs)
        expected_durations = np.exp(logpmfs.T)

        return expected_durations


# TODO call this 'time homog'
class HSMMStatesEigen(HSMMStatesPython):
    # NOTE: the methods in this class only work with iid emissions (i.e. without
    # overriding methods like cumulative_likelihood_block)

    def messages_backwards(self):
        # NOTE: np.maximum calls are because the C++ code doesn't do
        # np.logaddexp(-inf,-inf) = -inf, it likes nans instead
        from pyhsmm.internals.hsmm_messages_interface import messages_backwards_log
        betal, betastarl = messages_backwards_log(
                np.maximum(self.trans_matrix,1e-50),self.aBl,np.maximum(self.aDl,-1000000),
                self.aDsl,np.empty_like(self.aBl),np.empty_like(self.aBl),
                self.right_censoring,self.trunc if self.trunc is not None else self.T)
        assert not np.isnan(betal).any()
        assert not np.isnan(betastarl).any()

        if not self.left_censoring:
            self._normalizer = logsumexp(np.log(self.pi_0) + betastarl[0])
        else:
            raise NotImplementedError

        return betal, betastarl

    def messages_backwards_python(self):
        return super(HSMMStatesEigen,self).messages_backwards()

    def sample_forwards(self,betal,betastarl):
        from pyhsmm.internals.hsmm_messages_interface import sample_forwards_log
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
        from pyhsmm.internals.hsmm_messages_interface import resample_log_multiple
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

#################################
#  geometric / HMM-like models  #
#################################

class GeoHSMMStates(HSMMStatesPython):
    def resample(self):
        alphan, self._normalizer = HMMStatesEigen._messages_forwards_normalized(
                self.hmm_trans_matrix,self.pi_0,self.aBl)
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

    def E_step(self):
        alphal = HMMStatesEigen._messages_forwards_log(
                self.hmm_trans_matrix,
                self.pi_0,
                self.aBl)
        betal = HMMStatesEigen._messages_backwards_log(
                self.hmm_trans_matrix,
                self.aBl)
        self.expected_states, self.expected_transcounts, self._normalizer = \
                HMMStatesPython._expected_statistics_from_messages(
                        self.hmm_trans_matrix,
                        self.aBl,
                        alphal,
                        betal)

        # using these is untested!
        self._expected_ns = np.diag(self.expected_transcounts).copy()
        self._expected_tots = self.expected_transcounts.sum(1)

        self.expected_transcounts.flat[::self.expected_transcounts.shape[0]+1] = 0.

    @property
    def expected_durations(self):
        raise NotImplementedError

    @expected_durations.setter
    def expected_durations(self,val):
        raise NotImplementedError

    # TODO viterbi!

class DelayedGeoHSMMStates(HSMMStatesPython):
    def clear_caches(self):
        super(DelayedGeoHSMMStates,self).clear_caches()
        self._hmm_aBl = None
        self._hmm_trans_matrix = None

    def resample(self):
        alphan, self._normalizer = HMMStatesEigen._messages_forwards_normalized(
                self.hmm_trans_matrix,self.hmm_pi_0,self.hmm_aBl)
        self.stateseq = HMMStatesEigen._sample_backwards_normalized(
                alphan,self.hmm_trans_matrix.T.copy())

    @property
    def delays(self):
        return np.array([d.delay for d in self.dur_distns])

    @property
    def hmm_trans_matrix(self):
        # NOTE: more general version, allows different delays, o/w we could
        # construct with np.kron
        if self._hmm_trans_matrix is None:
            ps, delays = map(np.array,zip(*[(d.p,d.delay) for d in self.dur_distns]))
            starts, ends = cumsum(delays,strict=True), cumsum(delays,strict=False)
            trans_matrix = self._hmm_trans_matrix = np.zeros((ends[-1],ends[-1]))

            for (i,j), Aij in np.ndenumerate(self.trans_matrix):
                block = trans_matrix[starts[i]:ends[i],starts[j]:ends[j]]
                if i == j:
                    block[:-1,1:] = np.eye(block.shape[0]-1)
                    block[-1,-1] = 1-ps[i]
                else:
                    block[-1,0] = ps[j]*Aij

        return self._hmm_trans_matrix

    @property
    def hmm_aBl(self):
        if self._hmm_aBl is None:
            self._hmm_aBl = self.aBl.repeat(self.delays,axis=1)
        return self._hmm_aBl

    @property
    def hmm_pi_0(self):
        delays = self.delays
        starts = cumsum(delays,strict=True)

        pi_0 = np.zeros(delays.sum())
        pi_0[starts] = self.pi_0
        return pi_0

    @property
    def delays(self):
        return np.array([d.delay for d in self.dur_distns])

##################
#  changepoints  #
##################

class _PossibleChangepointsMixin(hmm_states._PossibleChangepointsMixin,HSMMStatesPython):
    @property
    def stateseq(self):
        return super(_PossibleChangepointsMixin,self).stateseq

    @stateseq.setter
    def stateseq(self,stateseq):
        hmm_states._PossibleChangepointsMixin.stateseq.fset(self,stateseq)
        HSMMStatesPython.stateseq.fset(self,self.stateseq)

    def init_meanfield_from_sample(self):
        # NOTE: only durations is different here; uses Tfull
        self.expected_states = \
            np.hstack([(self.stateseq == i).astype('float64')[:,na]
                for i in range(self.num_states)])

        from pyhsmm.util.general import count_transitions
        self.expected_transcounts = \
            count_transitions(self.stateseq_norep,minlength=self.num_states)

        self.expected_durations = expected_durations = \
                np.zeros((self.num_states,self.Tfull))
        for state in range(self.num_states):
            expected_durations[state] += \
                np.bincount(
                    self.durations_censored[self.stateseq_norep == state],
                    minlength=self.Tfull)[:self.Tfull]

class GeoHSMMStatesPossibleChangepoints(_PossibleChangepointsMixin,GeoHSMMStates):
    pass

class HSMMStatesPossibleChangepoints(_PossibleChangepointsMixin,HSMMStatesPython):
    def clear_caches(self):
        self._caBl = None
        super(HSMMStatesPossibleChangepoints,self).clear_caches()

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

    ### message passing

    # TODO caching

    # TODO wrap the duration stuff into single functions. reduces passing
    # around, reduces re-computation in this case

    # backwards messages potentials

    @property
    def caBl(self):
        if self._caBl is None:
            self._caBl = np.vstack((np.zeros(self.num_states),self.aBl.cumsum(0)))
        return self._caBl


    def cumulative_obs_potentials(self,tblock):
        return self.caBl[tblock+1:][:self.trunc], self.caBl[tblock]
        # return self.aBl[tblock:].cumsum(0)[:self.trunc]

    def dur_potentials(self,tblock):
        possible_durations = self.segmentlens[tblock:].cumsum()[:self.trunc].astype('int32')
        return self.aDl[possible_durations -1]

    def dur_survival_potentials(self,tblock):
        # return -np.inf # for testing against other implementation
        max_dur = self.segmentlens[tblock:].cumsum()[:self.trunc][-1]
        return self.aDsl[max_dur -1]

    # forwards messages potentials

    def reverse_cumulative_obs_potentials(self,tblock):
        return rcumsum(self.aBl[:tblock+1])\
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
        return self.mf_aBl[tblock:].cumsum(0)[:self.trunc], 0.

    def mf_reverse_cumulative_obs_potentials(self,tblock):
        return rcumsum(self.mf_aBl[:tblock+1])\
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
            durprobssum = durprobs.sum()
            durprobs /= durprobssum

            # If no duration is possible, then pick the first duration
            if durprobssum == 0:
                durprobs[0] = 1.0
                durprobs[1:] = 0.0

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

    # TODO E step refactor
    # TODO trunc

    def _expected_durations(self,
            dur_potentials,cumulative_obs_potentials,
            alphastarl,betal,normalizer):
        logpmfs = -np.inf*np.ones((self.Tfull,alphastarl.shape[1]))
        errs = np.seterr(invalid='ignore') # logaddexp(-inf,-inf)
        # TODO censoring not handled correctly here
        for tblock in range(self.Tblock):
            possible_durations = self.segmentlens[tblock:].cumsum()[:self.trunc]
            cB, offset = cumulative_obs_potentials(tblock)
            logpmfs[possible_durations -1] = np.logaddexp(
                    dur_potentials(tblock) + alphastarl[tblock]
                    + betal[tblock:tblock+self.trunc if self.trunc is not None else None]
                    + cB - (offset + normalizer),
                    logpmfs[possible_durations -1])
        np.seterr(**errs)
        return np.exp(logpmfs.T)


###################
#  sparate trans  #
###################

class HSMMStatesSeparateTrans(_SeparateTransMixin,HSMMStatesEigen):
    pass

class HSMMStatesPossibleChangepointsSeparateTrans(
        _SeparateTransMixin,
        HSMMStatesPossibleChangepoints):
    pass

##########
#  temp  #
##########

class DiagGaussStates(HSMMStatesPossibleChangepointsSeparateTrans):
    @property
    def aBl(self):
        if self._aBl is None:
            sigmas = np.array([d.sigmas for d in self.obs_distns])
            Js = -1./(2*sigmas)
            mus = np.array([d.mu for d in self.obs_distns])
            aBl = (np.einsum('td,td,nd->tn',self.data,self.data,Js)
                    - np.einsum('td,nd,nd->tn',self.data,2*mus,Js)) \
                  + (mus**2*Js - 1./2*np.log(2*np.pi*sigmas)).sum(1)
            aBl[np.isnan(aBl).any(1)] = 0.

            aBBl = np.empty((self.Tblock,self.num_states))
            for idx, (start,stop) in enumerate(self.changepoints):
                aBBl[idx] = aBl[start:stop].sum(0)

            self._aBl = aBl
            self._aBBl = aBBl
        return self._aBBl

    @property
    def aBl_slow(self):
        return super(DiagGaussStates,self).aBl

class DiagGaussGMMStates(HSMMStatesPossibleChangepointsSeparateTrans):
    @property
    def aBl(self):
        return self.aBl_eigen

    @property
    def aBl_einsum(self):
        if self._aBBl is None:
            sigmas = np.array([[c.sigmas for c in d.components] for d in self.obs_distns])
            Js = -1./(2*sigmas)
            mus = np.array([[c.mu for c in d.components] for d in self.obs_distns])

            # all_likes is T x Nstates x Ncomponents
            all_likes = \
                    (np.einsum('td,td,nkd->tnk',self.data,self.data,Js)
                        - np.einsum('td,nkd,nkd->tnk',self.data,2*mus,Js))
            all_likes += (mus**2*Js - 1./2*np.log(2*np.pi*sigmas)).sum(2)

            # weights is Nstates x Ncomponents
            weights = np.log(np.array([d.weights.weights for d in self.obs_distns]))
            all_likes += weights[na,...]

            # aBl is T x Nstates
            aBl = self._aBl = logsumexp(all_likes, axis=2)
            aBl[np.isnan(aBl).any(1)] = 0.

            aBBl = self._aBBl = np.empty((self.Tblock,self.num_states))
            for idx, (start,stop) in enumerate(self.changepoints):
                aBBl[idx] = aBl[start:stop].sum(0)

        return self._aBBl

    @property
    def aBl_eigen(self):
        if self._aBBl is None:
            sigmas = np.array([[c.sigmas for c in d.components] for d in self.obs_distns])
            mus = np.array([[c.mu for c in d.components] for d in self.obs_distns])
            weights = np.array([d.weights.weights for d in self.obs_distns])
            changepoints = np.array(self.changepoints).astype('int32')

            if self.model.temperature is not None:
                sigmas *= self.model.temperature

            from pyhsmm.util.temp import gmm_likes
            self._aBBl = np.empty((self.Tblock,self.num_states))
            gmm_likes(self.data,sigmas,mus,weights,changepoints,self._aBBl)
        return self._aBBl

    @property
    def aBl_slow(self):
        self.clear_caches()
        return super(DiagGaussGMMStates,self).aBl

############################
#  HSMM message functions  #
############################

def hsmm_messages_backwards_log(
    trans_potentials, initial_state_potential,
    cumulative_obs_potentials, dur_potentials, dur_survival_potentials,
    betal, betastarl,
    left_censoring=False, right_censoring=True):
    errs = np.seterr(invalid='ignore') # logaddexp(-inf,-inf)

    T, _ = betal.shape

    betal[-1] = 0.
    for t in range(T-1,-1,-1):
        cB, offset = cumulative_obs_potentials(t)
        dp = dur_potentials(t)
        betastarl[t] = logsumexp(
            betal[t:t+cB.shape[0]] + cB + dur_potentials(t), axis=0)
        betastarl[t] -= offset
        if right_censoring:
            np.logaddexp(betastarl[t], cB[-1] - offset + dur_survival_potentials(t),
                    out=betastarl[t])
        betal[t-1] = logsumexp(betastarl[t] + trans_potentials(t-1), axis=1)
    betal[-1] = 0. # overwritten on last iteration

    if not left_censoring:
        normalizer = logsumexp(initial_state_potential + betastarl[0])
    else:
        raise NotImplementedError

    np.seterr(**errs)
    return betal, betastarl, normalizer

def hsmm_messages_forwards_log(
    trans_potential, initial_state_potential,
    reverse_cumulative_obs_potentials, reverse_dur_potentials, reverse_dur_survival_potentials,
    alphal, alphastarl,
    left_censoring=False, right_censoring=True):

    T, _ = alphal.shape

    alphastarl[0] = initial_state_potential
    for t in range(T-1):
        cB = reverse_cumulative_obs_potentials(t)
        alphal[t] = logsumexp(
            alphastarl[t+1-cB.shape[0]:t+1] + cB + reverse_dur_potentials(t), axis=0)
        if left_censoring:
            raise NotImplementedError
        alphastarl[t+1] = logsumexp(
            alphal[t][:,na] + trans_potential(t), axis=0)
    t = T-1
    cB = reverse_cumulative_obs_potentials(t)
    alphal[t] = logsumexp(
        alphastarl[t+1-cB.shape[0]:t+1] + cB + reverse_dur_potentials(t), axis=0)

    if not right_censoring:
        normalizer = logsumexp(alphal[t])
    else:
        normalizer = None # TODO

    return alphal, alphastarl, normalizer

def hsmm_sample_forwards_log(
    trans_potentials, initial_state_potential,
    cumulative_obs_potentials, dur_potentials, dur_survival_potentails,
    betal, betastarl,
    left_censoring=False, right_censoring=True):

    T, _ = betal.shape
    stateseq = np.empty(T,dtype=np.int32)
    durations = []

    t = 0

    if left_censoring:
        raise NotImplementedError
    else:
        nextstate_unsmoothed = initial_state_potential

    while t < T:
        ## sample the state
        nextstate_distn_log = nextstate_unsmoothed + betastarl[t]
        nextstate_distn = np.exp(nextstate_distn_log - logsumexp(nextstate_distn_log))
        assert nextstate_distn.sum() > 0
        state = sample_discrete(nextstate_distn)

        ## sample the duration
        dur_logpmf = dur_potentials(t)[:,state]
        obs, offset = cumulative_obs_potentials(t)
        obs, offset = obs[:,state], offset[state]
        durprob = np.random.random()

        dur = 0 # NOTE: always incremented at least once
        while durprob > 0 and dur < dur_logpmf.shape[0] and t+dur < T:
            p_d = np.exp(dur_logpmf[dur] + obs[dur] - offset
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
    for t in range(T-1,-1,-1):
        cB, offset = cumulative_obs_potentials(t)

        vals = beta_scores[t:t+cB.shape[0]] + cB + dur_potentials(t)
        if right_censoring:
            vals = np.vstack((vals,cB[-1] + dur_survival_potentials(t)))
        vals -= offset

        vals.max(axis=0,out=betastar_scores[t])
        vals.argmax(axis=0,out=betastar_args[t])

        vals = betastar_scores[t] + trans_potentials(t-1)

        vals.max(axis=1,out=beta_scores[t-1])
        vals.argmax(axis=1,out=beta_args[t-1])
    beta_scores[-1] = 0.

    stateseq = np.empty(T,dtype='int32')

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

