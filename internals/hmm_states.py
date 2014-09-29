from __future__ import division
import numpy as np
from numpy import newaxis as na
import abc, copy, warnings

from ..util.stats import sample_discrete, sample_discrete_from_log, sample_markov
from ..util.general import rle, top_eigenvector, rcumsum, cumsum
from ..util.profiling import line_profiled

######################
#  Mixins and bases  #
######################

class _StatesBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,model,T=None,data=None,stateseq=None,
            generate=True,initialize_from_prior=True):
        self.model = model

        self.T = T if T is not None else data.shape[0]
        self.data = data

        self.clear_caches()

        if stateseq is not None:
            self.stateseq = np.array(stateseq,dtype=np.int32)
        elif generate:
            if data is not None and not initialize_from_prior:
                self.resample()
            else:
                self.generate_states()

    def copy_sample(self,newmodel):
        new = copy.copy(self)
        new.clear_caches() # saves space, though may recompute later for likelihoods
        new.model = newmodel
        new.stateseq = self.stateseq.copy()
        return new

    _kwargs = {} # used in subclasses for joblib stuff

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
        counts = np.bincount(self.stateseq,minlength=self.num_states)
        obs = [iter(o.rvs(count)) for o, count in zip(self.obs_distns,counts)]
        self.data = np.squeeze(np.vstack([obs[state].next() for state in self.stateseq]))
        return self.data

    ### messages and likelihoods

    # some cached things depends on model parameters, so caches should be
    # cleared when the model changes (e.g. when parameters are updated)

    def clear_caches(self):
        self._aBl = self._mf_aBl = None
        self._normalizer = None

    @property
    def aBl(self):
        if self._aBl is None:
            data = self.data

            aBl = self._aBl = np.empty((data.shape[0],self.num_states))
            for idx, obs_distn in enumerate(self.obs_distns):
                aBl[:,idx] = obs_distn.log_likelihood(data).ravel()
            aBl[np.isnan(aBl).any(1)] = 0.

        return self._aBl

    @abc.abstractmethod
    def log_likelihood(self):
        pass

class _SeparateTransMixin(object):
    def __init__(self,group_id,**kwargs):
        assert not isinstance(group_id,np.ndarray)
        self.group_id = group_id

        self._kwargs = dict(self._kwargs,group_id=group_id) \
                if hasattr(self,'_kwargs') else dict(group_id=group_id)
        super(_SeparateTransMixin,self).__init__(**kwargs)

        # access these to be sure they're instantiated
        self.trans_matrix
        self.pi_0

    @property
    def trans_matrix(self):
        return self.model.trans_distns[self.group_id].trans_matrix

    @property
    def pi_0(self):
        return self.model.init_state_distns[self.group_id].pi_0

    @property
    def mf_trans_matrix(self):
        return np.maximum(
                self.model.trans_distns[self.group_id].exp_expected_log_trans_matrix,
                1e-3)

    @property
    def mf_pi_0(self):
        return self.model.init_state_distns[self.group_id].exp_expected_log_init_state_distn

class _PossibleChangepointsMixin(object):
    def __init__(self,model,data,changepoints=None,**kwargs):
        changepoints = changepoints if changepoints is not None \
                else [(t,t+1) for t in xrange(data.shape[0])]

        self.changepoints = changepoints
        self.segmentstarts = np.array([start for start,stop in changepoints],dtype=np.int32)
        self.segmentlens = np.array([stop-start for start,stop in changepoints],dtype=np.int32)

        assert all(l > 0 for l in self.segmentlens)
        assert sum(self.segmentlens) == data.shape[0]
        assert self.changepoints[0][0] == 0 and self.changepoints[-1][-1] == data.shape[0]

        self._kwargs = dict(self._kwargs,changepoints=changepoints)

        super(_PossibleChangepointsMixin,self).__init__(
                model,T=len(changepoints),data=data,**kwargs)

    def clear_caches(self):
        self._aBBl = self._mf_aBBl = None
        self._stateseq = None
        super(_PossibleChangepointsMixin,self).clear_caches()

    @property
    def Tblock(self):
        return len(self.changepoints)

    @property
    def Tfull(self):
        return self.data.shape[0]

    @property
    def stateseq(self):
        if self._stateseq is None:
            self._stateseq = self.blockstateseq.repeat(self.segmentlens)
        return self._stateseq

    @stateseq.setter
    def stateseq(self,stateseq):
        assert len(stateseq) == self.Tblock or len(stateseq) == self.Tfull
        if len(stateseq) == self.Tblock:
            self.blockstateseq = stateseq
        else:
            self.blockstateseq = stateseq[self.segmentstarts]
        self._stateseq = None

    def _expected_states(self,*args,**kwargs):
        expected_states = \
            super(_PossibleChangepointsMixin,self)._expected_states(*args,**kwargs)
        return expected_states.repeat(self.segmentlens,axis=0)

    @property
    def aBl(self):
        if self._aBBl is None:
            aBl = super(_PossibleChangepointsMixin,self).aBl
            aBBl = self._aBBl = np.empty((self.Tblock,self.num_states))
            for idx, (start,stop) in enumerate(self.changepoints):
                aBBl[idx] = aBl[start:stop].sum(0)
        return self._aBBl

    @property
    def mf_aBl(self):
        if self._mf_aBBl is None:
            aBl = super(_PossibleChangepointsMixin,self).mf_aBl
            aBBl = self._mf_aBBl = np.empty((self.Tblock,self.num_states))
            for idx, (start,stop) in enumerate(self.changepoints):
                aBBl[idx] = aBl[start:stop].sum(0)
        return self._mf_aBBl

    def plot(self,*args,**kwargs):
        from matplotlib import pyplot as plt
        super(_PossibleChangepointsMixin,self).plot(*args,**kwargs)
        plt.xlim((0,self.Tfull))

    # TODO do generate() and generate_states() actually work?

####################
#  States classes  #
####################

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
        if self._normalizer is None:
            self.messages_forwards_normalized() # NOTE: sets self._normalizer
        return self._normalizer

    def _messages_log(self,trans_matrix,init_state_distn,log_likelihoods):
        alphal = self._messages_forwards_log(trans_matrix,init_state_distn,log_likelihoods)
        betal = self._messages_backwards_log(trans_matrix,log_likelihoods)
        return alphal, betal

    def messages_log(self):
        return self._messages_log(self.trans_matrix,self.pi_0,self.aBl)

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
        self._normalizer = np.logaddexp.reduce(np.log(self.pi_0) + betal[0] + self.aBl[0])
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
        self._normalizer = np.logaddexp.reduce(alphal[-1])
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
        betan, self._normalizer = \
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
        alphan, self._normalizer = \
                self._messages_forwards_normalized(self.trans_matrix,self.pi_0,self.aBl)
        return alphan

    ### Gibbs sampling

    def resample_log(self):
        betal = self.messages_backwards_log()
        self.sample_forwards_log(betal)

    @line_profiled
    def resample_normalized(self):
        alphan = self.messages_forwards_normalized()
        self.sample_backwards_normalized(alphan)

    def resample(self):
        return self.resample_normalized()

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
        if self._mf_aBl is None:
            T = self.data.shape[0]
            self._mf_aBl = aBl = np.empty((T,self.num_states))

            for idx, o in enumerate(self.obs_distns):
                aBl[:,idx] = o.expected_log_likelihood(self.data).ravel()
            aBl[np.isnan(aBl).any(1)] = 0.

        return self._mf_aBl

    @property
    def mf_trans_matrix(self):
        return self.model.trans_distn.exp_expected_log_trans_matrix
        # return np.maximum(self.model.trans_distn.exp_expected_log_trans_matrix,1e-5)

    @property
    def mf_pi_0(self):
        return self.model.init_state_distn.exp_expected_log_init_state_distn

    @property
    def all_expected_stats(self):
        return self.expected_states, self.expected_transcounts, self._normalizer

    @all_expected_stats.setter
    def all_expected_stats(self,vals):
        self.expected_states, self.expected_transcounts, self._normalizer = vals
        self.stateseq = self.expected_states.argmax(1).astype('int32') # for plotting

    def meanfieldupdate(self):
        self.clear_caches()
        self.all_expected_stats = self._expected_statistics(
                self.mf_trans_matrix,self.mf_pi_0,self.mf_aBl)

    def get_vlb(self):
        if self._normalizer is None:
            self.meanfieldupdate() # NOTE: sets self._normalizer
        return self._normalizer

    def _expected_statistics(self,trans_potential,init_potential,likelihood_log_potential):
        alphal = self._messages_forwards_log(trans_potential,init_potential,
                likelihood_log_potential)
        betal = self._messages_backwards_log(trans_potential,likelihood_log_potential)
        expected_states, expected_transcounts, normalizer = \
                self._expected_statistics_from_messages(trans_potential,likelihood_log_potential,alphal,betal)
        assert not np.isinf(expected_states).any()
        return expected_states, expected_transcounts, normalizer

    @staticmethod
    def _expected_statistics_from_messages(trans_potential,likelihood_log_potential,alphal,betal):
        expected_states = alphal + betal
        expected_states -= expected_states.max(1)[:,na]
        np.exp(expected_states,out=expected_states)
        expected_states /= expected_states.sum(1)[:,na]

        Al = np.log(trans_potential)
        log_joints = alphal[:-1,:,na] + (betal[1:,na,:] + likelihood_log_potential[1:,na,:]) + Al[na,...]
        log_joints -= log_joints.max((1,2))[:,na,na]
        joints = np.exp(log_joints)
        joints /= joints.sum((1,2))[:,na,na] # NOTE: renormalizing each isnt really necessary
        expected_transcounts = joints.sum(0)

        normalizer = np.logaddexp.reduce(alphal[0] + betal[0])

        return expected_states, expected_transcounts, normalizer

    ### EM

    def E_step(self):
        self.clear_caches()
        self.all_expected_stats = self._expected_statistics(
                self.trans_matrix,self.pi_0,self.aBl)

    ### Viterbi

    def Viterbi(self):
        scores, args = self.maxsum_messages_backwards()
        self.maximize_forwards(scores,args)

    def maxsum_messages_backwards(self):
        return self._maxsum_messages_backwards(self.trans_matrix,self.aBl)

    def maximize_forwards(self,scores,args):
        self.stateseq = self._maximize_forwards(scores,args,self.pi_0,self.aBl)


    def mf_Viterbi(self):
        scores, args = self.mf_maxsum_messages_backwards()
        self.mf_maximize_forwards(scores,args)

    def mf_maxsum_messages_backwards(self):
        return self._maxsum_messages_backwards(self.mf_trans_matrix,self.mf_aBl)

    def mf_maximize_forwards(self,scores,args):
        self.stateseq = self._maximize_forwards(scores,args,self.mf_pi_0,self.mf_aBl)


    @staticmethod
    def _maxsum_messages_backwards(trans_matrix, log_likelihoods):
        errs = np.seterr(divide='ignore')
        Al = np.log(trans_matrix)
        np.seterr(**errs)
        aBl = log_likelihoods

        scores = np.zeros_like(aBl)
        args = np.zeros(aBl.shape,dtype=np.int32)

        for t in xrange(scores.shape[0]-2,-1,-1):
            vals = Al + scores[t+1] + aBl[t+1]
            vals.argmax(axis=1,out=args[t+1])
            vals.max(axis=1,out=scores[t])

        return scores, args

    @staticmethod
    def _maximize_forwards(scores,args,init_state_distn,log_likelihoods):
        aBl = log_likelihoods
        T = aBl.shape[0]

        stateseq = np.empty(T,dtype=np.int32)

        stateseq[0] = (scores[0] + np.log(init_state_distn) + aBl[0]).argmax()
        for idx in xrange(1,T):
            stateseq[idx] = args[idx,stateseq[idx-1]]

        return stateseq

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
                s._normalizer = loglike

    ### EM

    @staticmethod
    def _expected_statistics_from_messages(
            trans_potential,likelihood_log_potential,alphal,betal,
            expected_states=None,expected_transcounts=None):
        from hmm_messages_interface import expected_statistics_log
        expected_states = np.zeros_like(alphal) \
                if expected_states is None else expected_states
        expected_transcounts = np.zeros_like(trans_potential) \
                if expected_transcounts is None else expected_transcounts
        return expected_statistics_log(
                np.log(trans_potential),likelihood_log_potential,alphal,betal,
                expected_states,expected_transcounts)

    ### Vitberbi

    def Viterbi(self):
        from hmm_messages_interface import viterbi
        self.stateseq = viterbi(self.trans_matrix,self.aBl,self.pi_0,
                np.empty(self.aBl.shape[0],dtype='int32'))

class HMMStatesEigenSeparateTrans(_SeparateTransMixin,HMMStatesEigen):
    pass

class HMMStatesPossibleChangepoints(_PossibleChangepointsMixin,HMMStatesEigen):
    pass

class HMMStatesPossibleChangepointsSeparateTrans(
        _SeparateTransMixin,
        HMMStatesPossibleChangepoints):
    pass

