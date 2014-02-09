import numpy as np
from numpy import newaxis as na
from numpy.random import random
import scipy.weave
import abc, copy, warnings
import scipy.stats as stats
import scipy.sparse as sparse

np.seterr(invalid='raise')

from ..util.stats import sample_discrete, sample_discrete_from_log, sample_markov
from ..util.general import rle, top_eigenvector, rcumsum, cumsum
from ..util.profiling import line_profiled

# TODO change HSMM message methods to be called messages_log

class _StatesBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,model,T=None,data=None,stateseq=None,
            initialize_from_prior=True,**kwargs):
        self.model = model

        assert (data is None) ^ (T is None)
        self.T = data.shape[0] if data is not None else T
        self.data = data

        self.clear_caches()

        if stateseq is not None:
            self.stateseq = np.array(stateseq,dtype=np.int32)
        else:
            if data is not None and not initialize_from_prior:
                self.resample(**kwargs)
            else:
                self.generate_states()

    def copy_sample(self,newmodel):
        new = copy.copy(self)
        new.clear_caches() # saves space, though may recompute later for likelihoods
        new.model = newmodel
        new.stateseq = self.stateseq.copy()
        return new

    ### model properties

    @property
    def obs_distns(self):
        return self.model.obs_distns

    @property
    def trans_matrix(self):
        return self.model.trans_distn.trans_matrix

    @property
    def pi_0(self):
        return self.model.init_state_distn.pi_0

    @property
    def num_states(self):
        return self.model.num_states

    ### generation

    def generate(self):
        self.generate_states()
        return self.generate_obs()

    @abc.abstractmethod
    def generate_states(self):
        pass

    def generate_obs(self):
        obs = []
        for state,dur in zip(*rle(self.stateseq)):
            obs.append(self.obs_distns[state].rvs(int(dur)))
        self.data = np.concatenate(obs)
        return self.data

    ### messages and likelihoods

    # some cached things depends on model parameters, so caches should be
    # cleared when the model changes (e.g. when parameters are updated)

    def clear_caches(self):
        self._aBl = self._mf_aBl = None
        self._loglike = None
        self._vlb = None

    @property
    def aBl(self):
        if self._aBl is None:
            data = self.data
            aBl = self._aBl = np.empty((data.shape[0],self.num_states))
            for idx, obs_distn in enumerate(self.obs_distns):
                aBl[:,idx] = np.nan_to_num(obs_distn.log_likelihood(data))
        return self._aBl

    @abc.abstractmethod
    def log_likelihood(self):
        pass


class HMMStatesPython(_StatesBase):
    ### generation

    def generate_states(self):
        T = self.T
        nextstate_distn = self.pi_0
        A = self.trans_matrix

        stateseq = np.zeros(T,dtype=np.int32)
        for idx in xrange(T):
            stateseq[idx] = sample_discrete(nextstate_distn)
            nextstate_distn = A[stateseq[idx]]

        self.stateseq = stateseq
        return stateseq

    ### message passing

    def log_likelihood(self):
        if self._loglike is None:
            self.messages_forwards_normalized() # NOTE: sets self._loglike
        return self._loglike

    @staticmethod
    def _messages_backwards_log(trans_matrix,log_likelihoods):
        errs = np.seterr(over='ignore')
        Al = np.log(trans_matrix)
        aBl = log_likelihoods

        betal = np.zeros_like(aBl)

        for t in xrange(betal.shape[0]-2,-1,-1):
            np.logaddexp.reduce(Al + betal[t+1] + aBl[t+1],axis=1,out=betal[t])

        np.seterr(**errs)
        return betal

    def messages_backwards_log(self):
        betal = self._messages_backwards_log(self.trans_matrix,self.aBl)
        assert not np.isnan(betal).any()
        self._loglike = np.logaddexp.reduce(np.log(self.pi_0) + betal[0] + self.aBl[0])
        return betal

    @staticmethod
    def _messages_forwards_log(trans_matrix,init_state_distn,log_likelihoods):
        errs = np.seterr(over='ignore')
        Al = np.log(trans_matrix)
        aBl = log_likelihoods

        alphal = np.zeros_like(aBl)

        alphal[0] = np.log(init_state_distn) + aBl[0]
        for t in xrange(alphal.shape[0]-1):
            alphal[t+1] = np.logaddexp.reduce(alphal[t] + Al.T,axis=1) + aBl[t+1]

        np.seterr(**errs)
        return alphal

    def messages_forwards_log(self):
        alphal = self._messages_forwards_log(self.trans_matrix,self.pi_0,self.aBl)
        assert not np.any(np.isnan(alphal))
        self._loglike = np.logaddexp.reduce(alphal[-1])
        return alphal

    @staticmethod
    def _messages_backwards_normalized(trans_matrix,init_state_distn,log_likelihoods):
        aBl = log_likelihoods
        A = trans_matrix
        T = aBl.shape[0]

        betan = np.empty_like(aBl)
        logtot = 0.

        betan[-1] = 1.
        for t in xrange(T-2,-1,-1):
            cmax = aBl[t+1].max()
            betan[t] = A.dot(betan[t+1] * np.exp(aBl[t+1] - cmax))
            norm = betan[t].sum()
            logtot += cmax + np.log(norm)
            betan[t] /= norm

        cmax = aBl[0].max()
        logtot += cmax + np.log((np.exp(aBl[0] - cmax) * init_state_distn * betan[0]).sum())

        return betan, logtot

    def messages_backwards_normalized(self):
        betan, self._loglike = \
                self._messages_backwards_normalized(self.trans_matrix,self.pi_0,self.aBl)
        return betan

    @staticmethod
    def _messages_forwards_normalized(trans_matrix,init_state_distn,log_likelihoods):
        aBl = log_likelihoods
        A = trans_matrix
        T = aBl.shape[0]

        alphan = np.empty_like(aBl)
        logtot = 0.

        in_potential = init_state_distn
        for t in xrange(T):
            cmax = aBl[t].max()
            alphan[t] = in_potential * np.exp(aBl[t] - cmax)
            norm = alphan[t].sum()
            if norm != 0:
                alphan[t] /= norm
                logtot += np.log(norm) + cmax
            else:
                alphan[t:] = 0.
                return alphan, np.log(0.)
            in_potential = alphan[t].dot(A)

        return alphan, logtot

    def messages_forwards_normalized(self):
        alphan, self._loglike = \
                self._messages_forwards_normalized(self.trans_matrix,self.pi_0,self.aBl)
        return alphan

    ### Gibbs sampling

    def resample_log(self,temp=None):
        self.temp = temp
        betal = self.messages_backwards_log()
        self.sample_forwards_log(betal)

    def resample_normalized(self,temp=None):
        self.temp = temp
        alphan = self.messages_forwards_normalized()
        self.sample_backwards_normalized(alphan)

    def resample(self,temp=None):
        return self.resample_normalized(temp=temp)

    @staticmethod
    def _sample_forwards_log(betal,trans_matrix,init_state_distn,log_likelihoods):
        A = trans_matrix
        aBl = log_likelihoods
        T = aBl.shape[0]

        stateseq = np.empty(T,dtype=np.int32)

        nextstate_unsmoothed = init_state_distn
        for idx in xrange(T):
            logdomain = betal[idx] + aBl[idx]
            logdomain[nextstate_unsmoothed == 0] = -np.inf
            if np.any(np.isfinite(logdomain)):
                stateseq[idx] = sample_discrete(nextstate_unsmoothed * np.exp(logdomain - np.amax(logdomain)))
            else:
                stateseq[idx] = sample_discrete(nextstate_unsmoothed)
            nextstate_unsmoothed = A[stateseq[idx]]

        return stateseq

    def sample_forwards_log(self,betal):
        self.stateseq = self._sample_forwards_log(betal,self.trans_matrix,self.pi_0,self.aBl)

    @staticmethod
    def _sample_forwards_normalized(betan,trans_matrix,init_state_distn,log_likelihoods):
        A = trans_matrix
        aBl = log_likelihoods
        T = aBl.shape[0]

        stateseq = np.empty(T,dtype=np.int32)

        nextstate_unsmoothed = init_state_distn
        for idx in xrange(T):
            logdomain = aBl[idx]
            logdomain[nextstate_unsmoothed == 0] = -np.inf
            stateseq[idx] = sample_discrete(nextstate_unsmoothed * betan * np.exp(logdomain - np.amax(logdomain)))
            nextstate_unsmoothed = A[stateseq[idx]]

        return stateseq

    def sample_forwards_normalized(self,betan):
        self.stateseq = self._sample_forwards_normalized(
                betan,self.trans_matrix,self.pi_0,self.aBl)

    @staticmethod
    def _sample_backwards_normalized(alphan,trans_matrix_transpose):
        AT = trans_matrix_transpose
        T = alphan.shape[0]

        stateseq = np.empty(T,dtype=np.int32)

        next_potential = np.ones(AT.shape[0])
        for t in xrange(T-1,-1,-1):
            stateseq[t] = sample_discrete(next_potential * alphan[t])
            next_potential = AT[stateseq[t]]

        return stateseq

    def sample_backwards_normalized(self,alphan):
        self.stateseq = self._sample_backwards_normalized(alphan,self.trans_matrix.T.copy())

    ### Mean Field

    @property
    def mf_aBl(self):
        aBl = np.empty((self.data.shape[0],self.num_states))
        for idx, o in enumerate(self.obs_distns):
            aBl[:,idx] = o.expected_log_likelihood(self.data)
        return aBl

    @property
    def mf_trans_matrix(self):
        return self.model.trans_distn.exp_expected_log_trans_matrix

    @property
    def mf_pi_0(self):
        return self.model.init_state_distn.exp_expected_log_init_state_distn

    def meanfieldupdate(self):
        # TODO this is like E_step, I should abstract it
        mf_aBl = self.mf_aBl
        mf_trans_matrix = self.mf_trans_matrix
        mf_pi_0 = self.mf_pi_0
        assert not np.isnan(mf_aBl).any()
        mf_alphal = self._messages_forwards_log(mf_trans_matrix,mf_pi_0,mf_aBl)
        mf_betal = self._messages_backwards_log(mf_trans_matrix,mf_aBl)
        assert not np.isnan(mf_alphal).any()
        assert not np.isnan(mf_betal).any()

        # calculate the expected node stats (responsibilities)
        expectations = self.mf_expectations = mf_alphal + mf_betal
        expectations -= expectations.max(1)[:,na]
        np.exp(expectations,out=expectations)
        expectations /= expectations.sum(1)[:,na]

        # compute the expected pairwise suff. stats for transitions
        mf_Al = np.log(self.mf_trans_matrix)
        log_joints = mf_alphal[:-1,:,na] + (mf_betal[1:,na,:] + mf_aBl[1:,na,:]) + mf_Al[na,...]
        log_joints -= log_joints.max((1,2))[:,na,na]
        joints = np.exp(log_joints)
        joints /= joints.sum((1,2))[:,na,na] # NOTE: renormalizing each isnt really necessary
        self.mf_expected_transcounts = joints.sum(0)

        # cache the vlb
        self._vlb = np.logaddexp.reduce(mf_alphal[0] + mf_betal[0])
        self._loglike = None # message passing set this to the vlb

        # for plotting
        self.stateseq = expectations.argmax(1)

    def get_vlb(self):
        if self._vlb is None:
            self.meanfieldupdate() # NOTE: sets self._vlb
        return self._vlb

    ### EM

    def E_step(self):
        aBl = self.aBl
        errs = np.seterr(divide='ignore')
        Al = np.log(self.trans_matrix)
        np.seterr(**errs)
        alphal = self.alphal = self.messages_forwards_log()
        betal = self.betal = self.messages_backwards_log()

        expectations = self.expectations = alphal + betal
        expectations -= expectations.max(1)[:,na]
        np.exp(expectations,out=expectations)
        expectations /= expectations.sum(1)[:,na]

        pairwise_expectations = alphal[:-1,:,na] + (betal[1:,na,:] + aBl[1:,na,:]) + Al[na,...]
        pairwise_expectations -= pairwise_expectations.max()
        np.exp(pairwise_expectations,out=pairwise_expectations)
        self.expected_transcounts = pairwise_expectations.sum(0)
        self.expected_transcounts *= (self.T-1) / self.expected_transcounts.sum()

        self._loglike = np.logaddexp.reduce(alphal[0] + betal[0])

        self.stateseq = expectations.argmax(1) # plotting

    ### Viterbi

    def Viterbi(self):
        scores, args = self.maxsum_messages_backwards()
        self.maximize_forwards(scores,args)

    @staticmethod
    def _maxsum_messages_backwards(trans_matrix, log_likelihoods):
        Al = np.log(trans_matrix)
        aBl = log_likelihoods

        scores = np.zeros_like(aBl)
        args = np.zeros(aBl.shape,dtype=np.int32)

        for t in xrange(scores.shape[0]-2,-1,-1):
            vals = Al + scores[t+1] + aBl[t+1]
            vals.argmax(axis=1,out=args[t+1])
            vals.max(axis=1,out=scores[t])

        return scores, args

    def maxsum_messages_backwards(self):
        return self._maxsum_messages_backwards(self.trans_matrix,self.aBl)

    @staticmethod
    def _maximize_forwards(scores,args,init_state_distn,log_likelihoods):
        aBl = log_likelihoods
        T = aBl.shape[0]

        stateseq = np.empty(T,dtype=np.int32)

        stateseq[0] = (scores[0] + np.log(init_state_distn) + aBl[0]).argmax()
        for idx in xrange(1,T):
            stateseq[idx] = args[idx,stateseq[idx-1]]

        return stateseq

    def maximize_forwards(self,scores,args):
        self.stateseq = self._maximize_forwards(scores,args,self.pi_0,self.aBl)

    ### plotting

    def plot(self,colors_dict=None,vertical_extent=(0,1),**kwargs):
        from matplotlib import pyplot as plt
        states,durations = rle(self.stateseq)
        X,Y = np.meshgrid(np.hstack((0,durations.cumsum())),vertical_extent)

        if colors_dict is not None:
            C = np.array([[colors_dict[state] for state in states]])
        else:
            C = states[na,:]

        plt.pcolor(X,Y,C,vmin=0,vmax=1,**kwargs)
        plt.ylim(vertical_extent)
        plt.xlim((0,durations.sum()))
        plt.yticks([])

class HMMStatesEigen(HMMStatesPython):
    def generate_states(self):
        self.stateseq = sample_markov(
                T=self.T,
                trans_matrix=self.trans_matrix,
                init_state_distn=self.pi_0)

    ### common messages (Gibbs, EM, likelihood calculation)

    @staticmethod
    def _messages_backwards_log(trans_matrix,log_likelihoods):
        from hmm_messages_interface import messages_backwards_log
        return messages_backwards_log(
                trans_matrix,log_likelihoods,
                np.empty_like(log_likelihoods))

    @staticmethod
    def _messages_forwards_log(trans_matrix,init_state_distn,log_likelihoods):
        from hmm_messages_interface import messages_forwards_log
        return messages_forwards_log(trans_matrix,log_likelihoods,
                init_state_distn,np.empty_like(log_likelihoods))

    @staticmethod
    def _messages_forwards_normalized(trans_matrix,init_state_distn,log_likelihoods):
        from hmm_messages_interface import messages_forwards_normalized
        return messages_forwards_normalized(trans_matrix,log_likelihoods,
                init_state_distn,np.empty_like(log_likelihoods))

    # next three methods are just for convenient testing

    def messages_backwards_log_python(self):
        return super(HMMStatesEigen,self)._messages_backwards_log(
                self.trans_matrix,self.aBl)

    def messages_forwards_log_python(self):
        return super(HMMStatesEigen,self)._messages_forwards_log(
                self.trans_matrix,self.pi_0,self.aBl)

    def messages_forwards_normalized_python(self):
        return super(HMMStatesEigen,self)._messages_forwards_normalized(
                self.trans_matrix,self.pi_0,self.aBl)

    ### sampling

    @staticmethod
    def _sample_forwards_log(betal,trans_matrix,init_state_distn,log_likelihoods):
        from hmm_messages_interface import sample_forwards_log
        return sample_forwards_log(trans_matrix,log_likelihoods,
                init_state_distn,betal,np.empty(log_likelihoods.shape[0],dtype='int32'))

    @staticmethod
    def _sample_backwards_normalized(alphan,trans_matrix_transpose):
        from hmm_messages_interface import sample_backwards_normalized
        return sample_backwards_normalized(trans_matrix_transpose,alphan,
                np.empty(alphan.shape[0],dtype='int32'))

    @staticmethod
    def _resample_multiple(states_list):
        from hmm_messages_interface import resample_normalized_multiple
        if len(states_list) > 0:
            loglikes = resample_normalized_multiple(
                    states_list[0].trans_matrix,states_list[0].pi_0,
                    [s.aBl for s in states_list],[s.stateseq for s in states_list])
            for s, loglike in zip(states_list,loglikes):
                s._loglike = loglike

    ### EM

    # TODO E_step

    ### Vitberbi

    def Viterbi(self):
        from hmm_messages_interface import viterbi
        return viterbi(self.trans_matrix,self.aBl,self.pi_0,
                np.empty(self.aBl.shape[0],dtype='int32'))


class HSMMStatesPython(_StatesBase):
    def __init__(self,model,right_censoring=True,left_censoring=False,trunc=None,
            stateseq=None,**kwargs):
        self.right_censoring = right_censoring
        self.left_censoring = left_censoring
        self.trunc = trunc

        super(HSMMStatesPython,self).__init__(model,stateseq=stateseq,**kwargs)

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
        # not really a slice, but this can be passed in to an ndarray
        trunced = []
        if self.left_censoring:
            trunced.append(0)
        if self.right_censoring:
            trunced.append(-1)
        return trunced

    @property
    def pi_0(self):
        if not self.left_censoring:
            return self.model.init_state_distn.pi_0
        else:
            return self.model.left_censoring_init_state_distn.pi_0

    @property
    def dur_distns(self):
        return self.model.dur_distns

    ### generation

    def generate_states(self):
        if self.left_censoring:
            raise NotImplementedError # TODO
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
        self._aDl = self._mf_aDl = None
        self._aDsl = self._mf_aDsl = None
        super(HSMMStatesPython,self).clear_caches()

    @property
    def aDl(self):
        aDl = np.empty((self.T,self.num_states))
        possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
        for idx, dur_distn in enumerate(self.dur_distns):
            aDl[:,idx] = dur_distn.log_pmf(possible_durations)
        return aDl

    @property
    def aD(self):
        return np.exp(self.aDl)

    @property
    def aDsl(self):
        aDsl = np.empty((self.T,self.num_states))
        possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
        for idx, dur_distn in enumerate(self.dur_distns):
            aDsl[:,idx] = dur_distn.log_sf(possible_durations)
        return aDsl

    ### message passing

    def log_likelihood(self):
        if self._loglike is None:
            betal, betastarl = self.messages_backwards()
            if not self.left_censoring:
                self._loglike = np.logaddexp.reduce(np.log(self.pi_0) + betastarl[0])
            else:
                raise NotImplementedError
        return self._loglike

    def messages_backwards(self):
        errs = np.seterr(divide='ignore')
        aDl, aDsl, Al = self.aDl, self.aDsl, np.log(self.trans_matrix)
        np.seterr(**errs)
        T,num_states = aDl.shape
        trunc = self.trunc if self.trunc is not None else T

        betal = np.zeros((T,num_states),dtype=np.float64)
        betastarl = np.zeros((T,num_states),dtype=np.float64)

        for t in xrange(T-1,-1,-1):
            np.logaddexp.reduce(betal[t:t+trunc] + self.cumulative_likelihoods(t,t+trunc)
                    + aDl[:min(trunc,T-t)],axis=0, out=betastarl[t])
            if T-t-1 < trunc and self.right_censoring:
                np.logaddexp(betastarl[t], self.likelihood_block(t,None) + aDsl[T-t -1],
                        betastarl[t])
            np.logaddexp.reduce(betastarl[t] + Al,axis=1,out=betal[t-1])
        betal[-1] = 0. # overwritten on last iteration

        self._loglike = np.logaddexp.reduce(np.log(self.pi_0) + betastarl[0])

        return betal, betastarl

    def cumulative_likelihoods(self,start,stop):
        return np.cumsum(self.aBl[start:stop],axis=0)

    def cumulative_likelihood_state(self,start,stop,state):
        return np.cumsum(self.aBl[start:stop,state])

    def likelihood_block(self,start,stop):
        return np.sum(self.aBl[start:stop],axis=0)

    def likelihood_block_state(self,start,stop,state):
        return np.sum(self.aBl[start:stop,state])

    def messages_forwards(self):
        if self.left_censoring:
            raise NotImplementedError # TODO
        errs = np.seterr(divide='ignore')
        aDl, aDsl, Al = self.aDl, self.aDsl, np.log(self.trans_matrix)
        T, num_states = aDl.shape
        trunc = self.trunc if self.trunc is not None else T # TODO actually use trunc here

        alphal = np.zeros((T,num_states),dtype=np.float64)
        alphastarl = np.zeros((T,num_states),dtype=np.float64)

        alphastarl[0] = np.log(self.pi_0)
        np.seterr(**errs)
        for t in xrange(T-1):
            np.logaddexp.reduce(alphastarl[:t+1] + self.reverse_cumulative_likelihoods(None,t+1)
                    + aDl[:t+1][::-1],axis=0,out=alphal[t])
            np.logaddexp.reduce(alphal[t][:,na] + Al,axis=0,out=alphastarl[t+1])
        t = T-1
        np.logaddexp.reduce(alphastarl[:t+1] + self.reverse_cumulative_likelihoods(None,t+1)
                + aDl[:t+1][::-1],axis=0,out=alphal[t])

        if self.right_censoring:
            pass # TODO
        else:
            self._loglike = np.logaddexp.reduce(alphal[t])

        return alphal, alphastarl

    def reverse_cumulative_likelihoods(self,start,stop):
        return rcumsum(self.aBl[start:stop])

    ### EM

    def E_step(self):
        # NOTE: here be dragons
        alphal, alphastarl = self.messages_forwards()
        betal, betastarl = self.messages_backwards()
        log_p_y = self.log_likelihood()

        # posterior state probabilities (self.expectations)
        gammal = alphal + betal
        gammastarl = alphastarl + betastarl

        gamma = np.exp(gammal - log_p_y)
        gammastar = np.exp(gammastarl - log_p_y)

        self.expectations = expectations = \
                (gammastar - np.vstack((np.zeros(self.num_states),gamma[:-1]))).cumsum(0)

        assert not np.isnan(expectations).any()
        assert np.isclose(expectations.min(),0.,atol=1e-2)
        assert np.isclose(expectations.max(),1.,atol=1e-2)
        assert np.allclose(expectations.sum(1),1.,atol=1e-2)
        expectations = np.maximum(0.,expectations)
        expectations /= expectations.sum(1)[:,na]

        # expected transitions (self.expected_transcounts)
        transl = alphal[:-1,:,na] + betastarl[1:,na,:] + np.log(self.trans_matrix[na,...])
        transl -= log_p_y
        self.expected_transcounts = np.exp(transl).sum(0)

        # expected durations (self.expected_durations)
        # TODO this won't handle right-truncation exactly right...
        logpmfs = np.zeros((self.T,self.num_states))
        caBl = np.vstack((np.zeros(self.num_states),self.aBl.cumsum(0)))
        T, aDl = self.T, self.aDl

        for d in xrange(1,T+1):
            np.logaddexp.reduce(aDl[d-1] + alphastarl[:T-d+1] + betal[d-1:]
                    + caBl[d:] - caBl[:T-d+1] - log_p_y,axis=0,out=logpmfs[d-1])

        self.expected_durations = np.exp(logpmfs.T)

        # for plotting
        self.stateseq = expectations.argmax(1)

    ### Mean Field

    # TODO these methods are repeats from HMMStates; factor out mixins in
    # this file!

    @property
    def mf_aBl(self):
        aBl = np.empty((self.data.shape[0],self.num_states))
        for idx, o in enumerate(self.obs_distns):
            aBl[:,idx] = o.expected_log_likelihood(self.data)
        return aBl

    @property
    def mf_trans_matrix(self):
        return self.model.trans_distn.exp_expected_log_trans_matrix

    @property
    def mf_pi_0(self):
        return self.model.init_state_distn.exp_expected_log_init_state_distn

    # this method is new though

    @property
    def mf_aDl(self):
        aDl = np.empty((self.T,self.num_states))
        possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
        for idx, dur_distn in enumerate(self.dur_distns):
            aDl[:,idx] = dur_distn.expected_log_pmf(possible_durations)
        return aDl

    @property
    def mf_aD(self):
        return np.exp(self.mf_aDl)

    @property
    def mf_aDsl(self):
        aDsl = np.empty((self.T,self.num_states))
        possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
        for idx, dur_distn in enumerate(self.dur_distns):
            aDsl[:,idx] = dur_distn.expected_log_sf(possible_durations)
        return aDsl

    def _mf_param_swap(self):
        # this is pretty shady code-wise: we want to call the same
        # self.messages_fowards() and self.messages_backwards() methods and have
        # them use the mf versions of the parameters, so we just swap out the
        # 'normal' parameters. most pythonic code ever.
        cls = self.__class__
        cls.aBl, cls.mf_aBl = cls.mf_aBl, cls.aBl
        cls.aDl, cls.mf_aDl = cls.mf_aDl, cls.aDl
        cls.aDsl, cls.mf_aDsl = cls.mf_aDsl, cls.aDsl
        cls.aD, cls.mf_aD = cls.mf_aD, cls.mf_aD
        cls.trans_matrix, cls.mf_trans_matrix = cls.mf_trans_matrix, cls.trans_matrix
        cls.pi_0, cls.mf_pi_0 = cls.mf_pi_0, cls.pi_0

    def meanfieldupdate(self):
        # do the E step with mean field parameters
        self._mf_param_swap()
        self.E_step()
        self._mf_param_swap()

        # swap out all the computed stuff
        self._vlb = self._loglike
        self._loglike = None

        self.mf_expectations = self.expectations
        del self.expectations

        self.mf_expected_transcounts = self.expected_transcounts
        del self.expected_transcounts

        self.mf_expected_durations = self.expected_durations
        del self.expected_durations

    def get_vlb(self):
        if self._vlb is None:
            self.meanfieldupdate()
        return self._vlb

    ### Gibbs sampling

    def resample(self,temp=None):
        self.temp = temp
        betal, betastarl = self.messages_backwards()
        self.sample_forwards(betal,betastarl)

    def copy_sample(self,newmodel):
        new = super(HSMMStatesPython,self).copy_sample(newmodel)
        return new

    def sample_forwards(self,betal,betastarl):
        if self.left_censoring:
            raise NotImplementedError # TODO

        A = self.trans_matrix
        apmf = self.aD
        T, num_states = betal.shape

        stateseq = self.stateseq = np.zeros(T,dtype=np.int32)

        idx = 0
        nextstate_unsmoothed = self.pi_0

        while idx < T:
            logdomain = betastarl[idx] - np.amax(betastarl[idx])
            nextstate_distr = np.exp(logdomain) * nextstate_unsmoothed
            if (nextstate_distr == 0.).all():
                # this is a numerical issue; no good answer, so we'll just follow the messages.
                nextstate_distr = np.exp(logdomain)
            state = sample_discrete(nextstate_distr)

            durprob = random()
            dur = 0 # always incremented at least once
            prob_so_far = 0.0
            while durprob > 0:
                # NOTE: funny indexing: dur variable is 1 less than actual dur
                # we're considering, i.e. if dur=5 at this point and we break
                # out of the loop in this iteration, that corresponds to
                # sampling a duration of 6
                p_d_prior = apmf[dur,state] if dur < T else 1.
                assert not np.isnan(p_d_prior)
                assert p_d_prior >= 0

                if p_d_prior == 0:
                    dur += 1
                    continue

                if idx+dur < T:
                    mess_term = np.exp(self.likelihood_block_state(idx,idx+dur+1,state) \
                            + betal[idx+dur,state] - betastarl[idx,state])
                    p_d = mess_term * p_d_prior
                    prob_so_far += p_d

                    assert not np.isnan(p_d)
                    durprob -= p_d
                    dur += 1
                else:
                    if self.right_censoring:
                        dur = self.dur_distns[state].rvs_given_greater_than(dur)
                    else:
                        dur += 1

                    break

            assert dur > 0

            stateseq[idx:idx+dur] = state
            # stateseq_norep.append(state)
            # assert len(stateseq_norep) < 2 or stateseq_norep[-1] != stateseq_norep[-2]
            # durations.append(dur)

            nextstate_unsmoothed = A[state,:]

            idx += dur

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

class HSMMStatesEmbedding(HSMMStatesPython,HMMStatesPython):
    # NOTE: this class is purely for testing HSMM messages

    @property
    def hmm_aBl(self):
        return np.repeat(self.aBl,self.T,axis=1)

    @property
    def hmm_backwards_pi_0(self):
        if not self.left_censoring:
            aD = self.aD.copy()
            aD[-1] = [np.exp(distn.log_sf(self.T-1)) for distn in self.dur_distns]
            assert np.allclose(aD.sum(0),1.)
            pi_0 = (self.pi_0 *  aD[::-1,:]).T.ravel()
            assert np.isclose(pi_0.sum(),1.)
            return pi_0
        else:
            # TODO
            raise NotImplementedError

    @property
    def hmm_backwards_trans_matrix(self):
        # TODO construct this as a csr
        blockcols = []
        aD = self.aD
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
            # TODO
            raise NotImplementedError

    @property
    def hmm_forwards_trans_matrix(self):
        # TODO construct this as a csc
        blockrows = []
        aD = self.aD
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
            self._loglike = np.logaddexp.reduce(np.log(self.pi_0) + betastarl[0])
        else:
            raise NotImplementedError

        return betal, betastarl

    def messages_backwards_python(self):
        return super(HSMMStatesEigen,self).messages_backwards()

    def sample_forwards(self,betal,betastarl):
        from hsmm_messages_interface import sample_forwards_log
        if self.left_censoring:
            raise NotImplementedError # TODO
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
                s._loglike = loglike
                s.stateseq = stateseq

class _HSMMStatesIntegerNegativeBinomialBase(HMMStatesEigen, HSMMStatesPython):
    # TODO this inheritance is confusing; factor out base classes that make it
    # clear what's being inherited
    __metaclass__ = abc.ABCMeta

    def clear_caches(self):
        HSMMStatesPython.clear_caches(self)
        self._hmm_trans = None
        self._rs = None
        self._hmm_aBl = None
        # note: we never use aDl or aDsl in this class

    def copy_sample(self,*args,**kwargs):
        return HSMMStatesPython.copy_sample(self,*args,**kwargs)

    @property
    def hsmm_aBl(self):
        return super(_HSMMStatesIntegerNegativeBinomialBase,self).aBl

    @property
    def hsmm_trans_matrix(self):
        return super(_HSMMStatesIntegerNegativeBinomialBase,self).trans_matrix

    @property
    def hsmm_pi_0(self):
        return super(_HSMMStatesIntegerNegativeBinomialBase,self).pi_0

    @property
    def rs(self):
        if True or self._rs is None:
            self._rs = np.array([d.r for d in self.dur_distns],dtype=np.int32)
        return self._rs

    @property
    def ps(self):
        if True or self._ps is none:
            self._ps = np.array([d.p for d in self.dur_distns])
        return self._ps

    ###

    @property
    def pi_0(self):
        if not self.left_censoring:
            rs = self.rs
            starts = np.concatenate(((0,),rs.cumsum()[:-1]))
            pi_0 = np.zeros(rs.sum())
            pi_0[starts] = self.hsmm_pi_0
            return pi_0
        else:
            return top_eigenvector(self.trans_matrix)

    @property
    def aBl(self):
        if self._hmm_aBl is None or True:
            self._hmm_aBl = self.hsmm_aBl.repeat(self.rs,axis=1)
        return self._hmm_aBl

    @abc.abstractproperty
    def trans_matrix(self):
        pass

    def resample(self,temp=None):
        self.temp = temp
        betal = self.messages_backwards_log()
        self.sample_forwards_log(betal)

    # generic implementation, these could be overridden for efficiency
    # they act like HMMs, and they're probably called from an HMMStates method

    def generate_states(self):
        return self.generate_states_hmm()

    def sample_forwards_log(self,betal):
        return self.sample_forwards_log_hmm(betal)

    def sample_backwards_normalized(self,alphan):
        return self.sample_backwards_normalized_hmm(alphan)

    def maximize_forwards(self,scores,args):
        return self.maximize_forwards_hmm(scores,args)

    ### for testing, ensures calling parent HMM methods

    def generate_states_hmm(self):
        ret = HMMStatesEigen.generate_states(self)
        self._map_states()
        return ret

    def resample_hmm(self):
        alphan = self.messages_forwards_normalized_hmm()
        self.sample_backwards_normalized_hmm(alphan)

    def messages_backwards_log_hmm(self):
        return HMMStatesEigen.messages_backwards_log(self)

    def sample_forwards_log_hmm(self,betal):
        ret = HMMStatesEigen.sample_forwards_log(self,betal)
        self._map_states()
        return ret

    def messages_forwards_normalized_hmm(self):
        return HMMStatesEigen.messages_forwards_normalized(self)

    def sample_backwards_normalized_hmm(self,alphan):
        ret = HMMStatesEigen.sample_backwards_normalized(self,alphan)
        self._map_states()
        return ret

    def Viterbi_hmm(self):
        scores, args = self.maxsum_messages_backwards_hmm()
        return self.maximize_forwards_hmm(scores,args)

    def maxsum_messages_backwards_hmm(self):
        return HMMStatesEigen.maxsum_messages_backwards_log(self)

    def maximize_forwards_hmm(self,scores,args):
        ret = HMMStatesEigen.maximize_forwards(self,scores,args)
        self._map_states()
        return ret

    def _map_states(self):
        themap = np.arange(self.num_states).repeat(self.rs)
        assert themap.shape[0] == self.trans_matrix.shape[0]
        self.stateseq = themap[self.stateseq]

class HSMMStatesIntegerNegativeBinomial(_HSMMStatesIntegerNegativeBinomialBase):
    def clear_caches(self):
        super(HSMMStatesIntegerNegativeBinomial,self).clear_caches()
        self._binoms = None
        self._hmm_trans = None

    @property
    def trans_matrix(self):
        if self._hmm_trans is None or True: # TODO put back caching
            rs = self.rs
            ps = np.array([d.p for d in self.dur_distns])

            trans_matrix = np.zeros((rs.sum(),rs.sum()))
            trans_matrix += np.diag(np.repeat(ps,rs))
            trans_matrix += np.diag(np.repeat(1.-ps,rs)[:-1],k=1)
            for z in rs[:-1].cumsum():
                trans_matrix[z-1,z] = 0

            ends = rs.cumsum()
            starts = np.concatenate(((0,),rs.cumsum()[:-1]))
            binoms = self.binoms
            for (i,j), v in np.ndenumerate(self.hsmm_trans_matrix * (1-ps)[:,na]):
                if i != j:
                    trans_matrix[ends[i]-1,starts[j]:ends[j]] = v * binoms[j]

            self._hmm_trans = trans_matrix
        return self._hmm_trans

    @property
    def binoms(self):
        if self._binoms is None or True: # TODO put back caching
            self._binoms = []
            for D in self.dur_distns:
                if 0 < D.p < 1:
                    arr = stats.binom.pmf(np.arange(D.r),D.r,1.-D.p)
                    arr[-1] += stats.binom.pmf(D.r,D.r,1.-D.p)
                else:
                    arr = np.zeros(D.r)
                    arr[0] = 1
                self._binoms.append(arr)
        return self._binoms

    def resample(self):
        alphan = self.messages_forwards_normalized()
        self.sample_backwards_normalized(alphan)

    def sample_forwards_normalized(self):
        # TODO fast messages_forwards_normalized
        HMMStatesEigen.sample_forwards_normalized(self)

    # TODO sample_backwards_normalized using sparse

    def meanfieldupdate(self):
        from ..util.general import count_transitions
        nsamples = self.mf_nsamples if hasattr(self,'mf_nsamples') else 25

        self.mf_expectations = np.zeros((self.T,self.num_states))
        self.mf_expected_transcounts = np.zeros((self.num_states,self.num_states))
        self.mf_expected_durations = np.zeros((self.num_states,self.T))

        eye = np.eye(self.num_states)/nsamples
        for i in xrange(nsamples):
            self.model._resample_from_mf()
            self.resample()
            self.mf_expectations += eye[self.stateseq]
            self.mf_expected_transcounts += \
                    count_transitions(self.stateseq_norep,minlength=self.num_states)/nsamples
            for state in xrange(self.num_states):
                self.mf_expected_durations[state] += \
                np.bincount(self.durations_censored[self.stateseq_norep == state],
                        minlength=self.T)[:self.T].astype(np.double)/nsamples # TODO remove hack

    def _resample_params_from_mf(self):
        self.model.trans_distn._resample_from_mf()
        self.model.init_state_distn._resample_from_mf()
        for o in self.model.obs_distns:
            o._resample_from_mf()
        for d in self.model.dur_distns:
            d._resample_from_mf()


class HSMMStatesIntegerNegativeBinomialVariant(_HSMMStatesIntegerNegativeBinomialBase):
    def clear_caches(self):
        super(HSMMStatesIntegerNegativeBinomialVariant,self).clear_caches()
        self._hmm_trans = None

    @property
    def trans_matrix(self):
        if self._hmm_trans is None or True:
            rs = self.rs
            ps = np.array([d.p for d in self.dur_distns])

            trans_matrix = np.zeros((rs.sum(),rs.sum()))
            trans_matrix += np.diag(np.repeat(ps,rs))
            trans_matrix += np.diag(np.repeat(1.-ps,rs)[:-1],k=1)
            for z in rs[:-1].cumsum():
                trans_matrix[z-1,z] = 0

            ends = rs.cumsum()
            starts = np.concatenate(((0,),rs.cumsum()[:-1]))
            for (i,j), v in np.ndenumerate(self.hsmm_trans_matrix * (1-ps)[:,na]):
                if i != j:
                    trans_matrix[ends[i]-1,starts[j]] = v

            self._hmm_trans = trans_matrix

        return self._hmm_trans

    ### structure-exploiting methods

    def resample(self):
        alphan = self.messages_forwards_normalized()
        self.sample_backwards_normalized(alphan)

    def messages_forwards_normalized(self):
        from hsmm_intnegbin_messages_interface import messages_forwards_normalized
        alphan, self._loglike = messages_forwards_normalized(
                self.hsmm_trans_matrix,self.hsmm_aBl,self.pi_0,self.rs,self.ps,
                np.empty((self.T,self.rs.sum())))
        return alphan

    # TODO a sample_backwards_normalized method using a sparse array repr

    @staticmethod
    def _resample_multiple(states_list):
        from hsmm_intnegbin_messages_interface import resample_normalized_multiple
        if len(states_list) > 0:
            Ts = [s.T for s in states_list]
            stateseqs = [np.empty(T,dtype=np.int32) for T in Ts]
            self = states_list[0]
            loglikes = resample_normalized_multiple(
                    self.trans_matrix,self.hsmm_trans_matrix,self.pi_0,
                    self.rs,self.ps,[s.hsmm_aBl for s in states_list],
                    stateseqs)
            for s, loglike, stateseq in zip(states_list,loglikes,stateseqs):
                s._loglike = loglike
                s.stateseq = stateseq

    ### methods that exist because of insufficient mixinification

    def E_step(self):
        raise NotImplementedError


class HSMMStatesPossibleChangepoints(HSMMStatesPython):
    def __init__(self,model,changepoints,**kwargs):
        self.changepoints = changepoints
        self.segmentstarts = np.array([start for start,stop in changepoints],dtype=np.int32)
        self.segmentlens = np.array([stop-start for start,stop in changepoints],dtype=np.int32)
        self.Tblock = len(changepoints) # number of blocks
        super(HSMMStatesPossibleChangepoints,self).__init__(model,**kwargs)

    def clear_caches(self):
        super(HSMMStatesPossibleChangepoints,self).clear_caches()
        self._aBBl = None

    @property
    def aBBl(self):
        if self._aBBl is None:
            aBl = self.aBl
            aBBl = self._aBBl = np.empty((self.Tblock,self.num_states))
            for idx, (start,stop) in enumerate(self.changepoints):
                aBBl[idx] = aBl[start:stop].sum(0)
        return self._aBBl

    ### generation

    def generate_states(self):
        if self.left_censoring:
            raise NotImplementedError # TODO
        Tblock = self.Tblock
        assert Tblock == len(self.changepoints)
        blockstateseq = np.zeros(Tblock,dtype=np.int32)

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

        # convert block state sequence to full stateseq and stateseq_norep and
        # durations
        stateseq = np.zeros(self.T,dtype=np.int32)
        for state, (start,stop) in zip(blockstateseq,self.changepoints):
            stateseq[start:stop] = state
        self.stateseq = stateseq

        return stateseq

    def generate(self): # TODO
        raise NotImplementedError

    ### message passing

    def log_likelihood(self):
        raise NotImplementedError

    def messages_backwards(self):
        errs = np.seterr(divide='ignore')
        aDl, Al = self.aDl, np.log(self.trans_matrix)
        np.seterr(**errs)
        Tblock = self.Tblock
        num_states = Al.shape[0]
        trunc = self.trunc if self.trunc is not None else self.T

        betal = np.zeros((Tblock,num_states),dtype=np.float64)
        betastarl = np.zeros_like(betal)

        for tblock in range(Tblock-1,-1,-1):
            possible_durations = self.segmentlens[tblock:].cumsum() # could precompute these
            possible_durations = possible_durations[possible_durations < max(trunc,possible_durations[0]+1)]
            truncblock = len(possible_durations)
            normalizer = np.logaddexp.reduce(aDl[possible_durations-1],axis=0)

            np.logaddexp.reduce(betal[tblock:tblock+truncblock]
                    + self.block_cumulative_likelihoods(tblock,tblock+truncblock,possible_durations)
                    + aDl[possible_durations-1] - normalizer,axis=0,out=betastarl[tblock])
            # TODO TODO put censoring here, must implement likelihood_block
            np.logaddexp.reduce(betastarl[tblock] + Al, axis=1, out=betal[tblock-1])
        betal[-1] = 0.

        assert not np.isnan(betal).any()
        assert not np.isnan(betastarl).any()

        return betal, betastarl

    def block_cumulative_likelihoods(self,startblock,stopblock,possible_durations):
        return self.aBBl[startblock:stopblock].cumsum(0)[:possible_durations.shape[0]]

    def block_cumulative_likelihood_state(self,startblock,stopblock,state,possible_durations):
        return self.aBBl[startblock:stopblock,state].cumsum(0)[:possible_durations.shape[0]]

    ### Gibbs sampling

    def sample_forwards(self,betal,betastarl):
        aDl = self.aDl
        trunc = self.trunc

        Tblock = betal.shape[0]
        assert Tblock == len(self.changepoints)
        blockstateseq = np.zeros(Tblock,dtype=np.int32)

        tblock = 0
        nextstate_unsmoothed = self.pi_0
        A = self.trans_matrix
        trunc = trunc if trunc is not None else self.T

        while tblock < Tblock:
            # sample the state
            logdomain = betastarl[tblock] - np.amax(betastarl[tblock])
            nextstate_distr = np.exp(logdomain) * nextstate_unsmoothed
            state = sample_discrete(nextstate_distr)

            # compute possible duration info (indep. of state)
            # TODO TODO doesn't handle censoring quite correctly
            possible_durations = self.segmentlens[tblock:].cumsum()
            possible_durations = possible_durations[possible_durations < max(trunc,possible_durations[0]+1)]
            truncblock = len(possible_durations)

            if truncblock > 1:
                # compute the next few log likelihoods
                loglikelihoods = self.block_cumulative_likelihood_state(tblock,tblock+truncblock,state,possible_durations)

                # compute pmf over those steps
                logpmf = aDl[possible_durations-1,state] + loglikelihoods + betal[tblock:tblock+truncblock,state] - betastarl[tblock,state]

                # sample from it
                blockdur = sample_discrete_from_log(logpmf)+1
            else:
                blockdur = 1

            # set block sequence
            blockstateseq[tblock:tblock+blockdur] = state

            # set up next iteration
            tblock += blockdur
            nextstate_unsmoothed = A[state]

        # convert block state sequence to full stateseq and stateseq_norep and
        # durations
        stateseq = np.zeros(self.T,dtype=np.int32)
        for state, (start,stop) in zip(blockstateseq,self.changepoints):
            stateseq[start:stop] = state
        self.stateseq = stateseq

        return stateseq

