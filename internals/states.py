import numpy as np
from numpy import newaxis as na
from numpy.random import random
import scipy.weave
import abc, copy, warnings
import scipy.stats as stats
import scipy.sparse as sparse

np.seterr(invalid='raise')

from ..util.stats import sample_discrete, sample_discrete_from_log, sample_markov
from ..util.general import rle, top_eigenvector

# TODO change HSMM message methods to be called messages_log

# TODO temperature

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
        return np.concatenate(obs)

    ### messages and likelihoods

    # some cached things depends on model parameters, so caches should be
    # cleared when the model changes (e.g. when parameters are updated)

    def clear_caches(self):
        self._aBl = None
        self._loglike = None

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

    ### EM

    def E_step(self):
        alphal = self.alphal = self.messages_forwards()
        betal = self.betal = self.messages_backwards()
        aBl = self.aBl
        Al = np.log(self.trans_matrix)

        expectations = alphal + betal
        expectations -= expectations.max(1)[:,na]
        np.exp(expectations,out=expectations)
        expectations /= expectations.sum(1)[:,na]
        self.expectations = expectations

        pairwise_expectations = alphal[:-1,:,na] + (betal[1:,na,:] + aBl[1:,na,:]) + Al[na,...]
        pairwise_expectations -= pairwise_expectations.max()
        np.exp(pairwise_expectations,out=pairwise_expectations)
        self.expected_transcounts = pairwise_expectations.sum(0)
        self.expected_transcounts *= (self.T-1) / self.expected_transcounts.sum()

        self.stateseq = expectations.argmax(1)

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

    def messages_forwards_normalized_python(self):
        return super(HMMStatesEigen,self).messages_forwards_normalized()

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

    def _get_stateseq(self):
        return self._stateseq

    def _set_stateseq(self,stateseq):
        self._stateseq = stateseq
        self._stateseq_norep = None
        self._durations_censored = None

    stateseq = property(_get_stateseq,_set_stateseq)

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
        self._aDl = None
        self._aDsl = None
        self._betal, self._betastarl = None, None
        super(HSMMStatesPython,self).clear_caches()

    @property
    def aDl(self):
        if self._aDl is None:
            self._aDl = aDl = np.empty((self.T,self.num_states))
            possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
            for idx, dur_distn in enumerate(self.dur_distns):
                aDl[:,idx] = dur_distn.log_pmf(possible_durations)
        return self._aDl

    @property
    def aD(self):
        return np.exp(self.aDl)

    @property
    def aDsl(self):
        if self._aDsl is None:
            self._aDsl = aDsl = np.empty((self.T,self.num_states))
            possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
            for idx, dur_distn in enumerate(self.dur_distns):
                aDsl[:,idx] = dur_distn.log_sf(possible_durations)
        return self._aDsl

    ### message passing

    def log_likelihood(self):
        if self._loglike is None:
            betal, _ = self.messages_backwards()
            self._loglike = np.logaddexp.reduce(np.log(self.pi_0) + betal[0] + self.aBl[0])
        return self._loglike

    def messages_backwards(self):
        if self._betal is not None and self._betastarl is not None:
            return self._betal, self._betastarl

        errs = np.seterr(divide='ignore')
        aDl, aDsl, Al = self.aDl, self.aDsl, np.log(self.trans_matrix)
        np.seterr(**errs)
        T,state_dim = aDl.shape
        trunc = self.trunc if self.trunc is not None else T

        betal = np.zeros((T,num_states),dtype=np.float64)
        betastarl = np.zeros((T,num_states),dtype=np.float64)

        for t in xrange(T-1,-1,-1):
            np.logaddexp.reduce(betal[t:t+trunc] + self.cumulative_likelihoods(t,t+trunc) + aDl[:min(trunc,T-t)],axis=0, out=betastarl[t])
            if T-t < trunc and self.right_censoring:
                np.logaddexp(betastarl[t], self.likelihood_block(t,None) + aDsl[T-t -1], betastarl[t])
            np.logaddexp.reduce(betastarl[t] + Al,axis=1,out=betal[t-1])
        betal[-1] = 0.

        self._betal, self._betastarl = betal, betastarl

        return betal, betastarl

    def cumulative_likelihoods(self,start,stop):
        return np.cumsum(self.aBl[start:stop],axis=0)

    def cumulative_likelihood_state(self,start,stop,state):
        return np.cumsum(self.aBl[start:stop,state])

    def likelihood_block(self,start,stop):
        return np.sum(self.aBl[start:stop],axis=0)

    def likelihood_block_state(self,start,stop,state):
        return np.sum(self.aBl[start:stop,state])

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

class HSMMStatesEigen(HSMMStatesPython):
    def sample_forwards(self,betal,betastarl):
        if self.left_censoring:
            raise NotImplementedError # TODO

        global eigen_path
        hsmm_sample_forwards_codestr = _get_codestr('hsmm_sample_forwards')

        A = self.trans_matrix
        apmf = self.aD
        T,M = betal.shape
        pi0 = self.pi_0
        aBl = self.aBl / self.temp if self.temp is not None else self.aBl

        stateseq = np.zeros(T,dtype=np.int32)

        scipy.weave.inline(hsmm_sample_forwards_codestr,
                ['betal','betastarl','aBl','stateseq','A','pi0','apmf','M','T'],
                headers=['<Eigen/Core>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])

        self.stateseq = stateseq # must have this line at end; it triggers stateseq_norep


class _HSMMStatesIntegerNegativeBinomialBase(HMMStatesEigen, HSMMStatesPython):
    # NOTE: I'm secretly an HMMStates first! I just shh

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
    def dur_distns(self):
        return self.model.dur_distns

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


    # the next methods are to override the calls that the parents' methods would
    # make so that the parent can view us as the effective HMM we are!

    @property
    def aBl(self):
        if self._hmm_aBl is None or True:
            self._hmm_aBl = self.hsmm_aBl.repeat(self.rs,axis=1)
        return self._hmm_aBl

    @abc.abstractproperty
    def pi_0(self):
        pass

    @abc.abstractproperty
    def trans_matrix(self):
        pass

    def resample(self,temp=None):
        self.temp = temp
        betal, superbetal = self.messages_backwards_log()
        self.sample_forwards_log(betal,superbetal)

    # generic implementation, these could be overridden for efficiency
    # they act like HMMs, and they're probably called from an HMMStates method

    def generate_states(self):
        ret = HMMStatesEigen.generate_states(self)
        self._map_states()
        return ret

    def messages_backwards_log(self):
        return self.messages_backwards_hmm(), None # 2nd is a dummy, see sample_forwards

    def messages_forwards_log(self):
        return HMMStatesEigen.messages_forwards_log(self)

    def sample_forwards_log(self,betal,dummy):
        return self.sample_forwards_hmm(betal)

    def maxsum_messages_backwards(self):
        return self.maxsum_messages_backwards_hmm()

    def maximize_forwards(self,scores,args):
        return self.maximize_forwards_hmm(scores,args)

    def _map_states(self):
        themap = np.arange(self.num_states).repeat(self.rs)
        self.stateseq = themap[self.stateseq]

    ### for testing, ensures calling parent HMM methods

    def Viterbi_hmm(self):
        scores, args = self.maxsum_messages_backwards_hmm()
        return self.maximize_forwards_hmm(scores,args)

    def messages_backwards_log_hmm(self):
        return HMMStatesEigen.messages_backwards(self)

    def sample_forwards_log_hmm(self,betal):
        ret = HMMStatesEigen.sample_forwards(self,betal)
        self._map_states()
        return ret

    def maxsum_messages_backwards_hmm(self):
        return HMMStatesEigen.maxsum_messages_backwards(self)

    def maximize_forwards_hmm(self,scores,args):
        ret = HMMStatesEigen.maximize_forwards(self,scores,args)
        self._map_states()
        return ret

class HSMMStatesIntegerNegativeBinomialVariant(_HSMMStatesIntegerNegativeBinomialBase):
    def clear_caches(self):
        super(HSMMStatesIntegerNegativeBinomialVariant,self).clear_caches()
        self._hmm_trans = None
        self._betal, self._superbetal = None, None

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

    @property
    def _trans_matrix_terms(self):
        # returns diag part and off-diag part of (hmm) trans matrix, along with
        # lengths
        rs = self.rs
        ps = np.array([d.p for d in self.dur_distns])
        return np.repeat(ps,rs), (1-ps)[:,na] * self.hsmm_trans_matrix, rs

    def E_step(self):
        raise NotImplementedError

    ### structure-exploiting methods

    def messages_backwards_log(self):
        if self._betal is not None and self._superbetal is not None:
            return self._betal, self._superbetal

        global eigen_path
        hsmm_intnegbin_messages_backwards_codestr = _get_codestr('hsmm_intnegbinvariant_messages_backwards')

        aBl = self.hsmm_aBl
        T,M = aBl.shape

        rs = self.rs
        crs = rs.cumsum()
        start_indices = np.concatenate(((0,),crs[:-1]))
        end_indices = crs-1
        rtot = int(crs[-1])
        ps = np.array([d.p for d in self.dur_distns])
        AT = self.hsmm_trans_matrix.T.copy() * (1-ps)

        superbetal = np.zeros((T,M))
        betal = np.zeros((T,rtot))

        scipy.weave.inline(hsmm_intnegbin_messages_backwards_codestr,
                ['start_indices','end_indices','rtot','AT','ps','superbetal','betal','aBl','T','M'],
                headers=['<Eigen/Core>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])

        assert not np.isnan(betal).any() and not np.isnan(superbetal).any()

        self._betal, self._superbetal = betal, superbetal

        return betal, superbetal

    def sample_forwards_log(self,betal,superbetal):
        global eigen_path
        hsmm_intnegbin_sample_forwards_codestr = _get_codestr('hsmm_intnegbinvariant_sample_forwards')

        aBl = self.hsmm_aBl
        T,M = aBl.shape
        rs = self.rs
        crs = rs.cumsum()
        start_indices = np.concatenate(((0,),crs[:-1]))
        end_indices = crs-1
        rtot = int(crs[-1])
        ps = np.array([d.p for d in self.dur_distns])
        A = self.hsmm_trans_matrix * (1-ps[:,na])
        A.flat[::A.shape[0]+1] = ps

        if self.left_censoring:
            initial_substate = sample_discrete_from_log(np.log(self.pi_0) + betal[0] + aBl[0].repeat(rs))
            initial_superstate = np.arange(self.num_states).repeat(self.rs)[initial_substate]
        else:
            initial_superstate = sample_discrete_from_log(np.log(self.hsmm_pi_0) + superbetal[0] + aBl[0])
            initial_substate = start_indices[initial_superstate]

        stateseq = np.zeros(T,dtype=np.int32)

        scipy.weave.inline(hsmm_intnegbin_sample_forwards_codestr,
                ['betal','superbetal','aBl','stateseq','A','initial_superstate','initial_substate','M','T','ps','rtot','start_indices','end_indices'],
                headers=['<Eigen/Core>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])

        self.stateseq = stateseq # must have this line at end; it triggers stateseq_norep

    def maxsum_messages_backwards(self):
        global eigen_path
        # these names are dumb
        hsmm_intnegbin_maxsum_messages_backwards_codestr = _get_codestr('hsmm_intnegbinvariant_maxsum_messages_backwards')

        aBl = self.hsmm_aBl
        T,M = aBl.shape

        rs = self.rs
        crs = rs.cumsum()
        start_indices = np.concatenate(((0,),crs[:-1]))
        end_indices = crs-1
        rtot = int(crs[-1])
        ps = np.array([d.p for d in self.dur_distns])
        logps = np.log(ps)
        log1mps = np.log1p(-ps)
        Al = np.log(self.hsmm_trans_matrix) + log1mps[:,na]

        scores = np.zeros((T,rtot))
        args = np.zeros((T,rtot),dtype=np.int32)

        scipy.weave.inline(hsmm_intnegbin_maxsum_messages_backwards_codestr,
                ['start_indices','end_indices','rtot','Al','aBl',
                    'logps','log1mps','scores','args','T','M'],
                headers=['<Eigen/Core>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])

        return scores, args

    def maximize_forwards(self,scores,args):
        global eigen_path
        hsmm_intnegbin_maximize_forwards_codestr = _get_codestr('hsmm_intnegbin_maximize_forwards') # same code as intnegbin

        T,rtot = scores.shape

        rs = self.rs
        crs = rs.cumsum()
        start_indices = np.concatenate(((0,),crs[:-1]))
        themap = np.arange(self.num_states).repeat(rs)

        stateseq = np.empty(T,dtype=np.int32)
        stateseq[0] = (scores[0,start_indices] + np.log(self.hsmm_pi_0) + self.hsmm_aBl[0]).argmax()
        initial_hmm_state = start_indices[stateseq[0]]

        scipy.weave.inline(hsmm_intnegbin_maximize_forwards_codestr,
                ['T','rtot','themap','scores','args','stateseq','initial_hmm_state'],
                headers=['<Eigen/Core>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])

        self.stateseq = stateseq # must have this line at end; it triggers stateseq_norep

class HSMMStatesIntegerNegativeBinomial(_HSMMStatesIntegerNegativeBinomialBase):
    def clear_caches(self):
        super(HSMMStatesIntegerNegativeBinomial,self).clear_caches()
        self._binoms = None

    # TODO test
    @property
    def binoms(self):
        raise NotImplementedError, 'theres a bug here' # TODO
        if self._binoms is None:
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

    # TODO test
    @property
    def pi_0(self):
        if not self.left_censoring:
            rs = self.rs
            return self.hsmm_pi_0.repeat(rs) * np.concatenate(self.binoms)
        else:
            return top_eigenvector(self.trans_matrix)

    # TODO test
    @property
    def trans_matrix(self):
        if self._hmm_trans is None:
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

    # matrix structure-exploiting methods

    def maxsum_messages_backwards_log(self):
        global eigen_path
        # these names are dumb
        hsmm_intnegbin_nonvariant_maxsum_messages_backwards_codestr = _get_codestr('hsmm_intnegbin_maxsum_messages_backwards')

        aBl = self.hsmm_aBl
        T,M = aBl.shape

        rs = self.rs
        crs = rs.cumsum()
        start_indices = np.concatenate(((0,),crs[:-1]))
        end_indices = crs-1
        rtot = int(crs[-1])
        ps = np.array([d.p for d in self.dur_distns])
        logps = np.log(ps)
        log1mps = np.log1p(-ps)
        Al = np.log(self.hsmm_trans_matrix) + log1mps[:,na]
        binoms = np.log(np.concatenate(self.binoms))

        scores = np.zeros((T,rtot))
        args = np.zeros((T,rtot),dtype=np.int32)

        scipy.weave.inline(hsmm_intnegbin_nonvariant_maxsum_messages_backwards_codestr,
                ['start_indices','end_indices','rtot','Al','aBl',
                    'logps','log1mps','scores','args','T','M','binoms'],
                headers=['<Eigen/Core>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])

        return scores, args

    def maximize_forwards(self,scores,args):
        global eigen_path
        hsmm_intnegbin_maximize_forwards_codestr = _get_codestr('hsmm_intnegbin_maximize_forwards')

        T,rtot = scores.shape

        rs = self.rs
        crs = rs.cumsum()
        start_indices = np.concatenate(((0,),crs[:-1]))
        themap = np.arange(self.num_states).repeat(rs)

        stateseq = np.empty(T,dtype=np.int32)
        initial_hmm_state = (np.concatenate(self.binoms) + scores[0] + np.log(self.pi_0) + self.aBl[0]).argmax()
        stateseq[0] = themap[initial_hmm_state]

        scipy.weave.inline(hsmm_intnegbin_maximize_forwards_codestr,
                ['T','rtot','themap','scores','args','stateseq','initial_hmm_state'],
                headers=['<Eigen/Core>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])

        self.stateseq = stateseq # must have this line at end; it triggers stateseq_norep

#################
#  eigen stuff  #
#################

# TODO move away from weave, which is not maintained. numba? ctypes? cffi?
# cython? probably cython or ctypes.

import os
eigen_path = os.path.join(os.path.dirname(__file__),'../deps/Eigen3/')
eigen_code_dir = os.path.join(os.path.dirname(__file__),'cpp_eigen_code/')

codestrs = {}
def _get_codestr(name):
    if name not in codestrs:
        with open(os.path.join(eigen_code_dir,name+'.cpp')) as infile:
            codestrs[name] = infile.read()
    return codestrs[name]

