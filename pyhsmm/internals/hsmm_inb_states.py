from __future__ import division
from builtins import zip, range
from future.utils import with_metaclass
import numpy as np
import abc
import scipy.stats as stats
import scipy.special as special
from scipy.special import logsumexp

try:
    from ..util.cstats import sample_markov
except ImportError:
    from ..util.stats import sample_markov
from ..util.general import top_eigenvector, cumsum

from .hmm_states import HMMStatesPython, HMMStatesEigen, _SeparateTransMixin
from .hsmm_states import HSMMStatesEigen

# TODO these classes are currently backed by HMM message passing, but they can
# be made much more time and memory efficient. i have the code to do it in some
# other branches, but dense matrix multiplies are actually competitive.


class _HSMMStatesIntegerNegativeBinomialBase(with_metaclass(abc.ABCMeta, HSMMStatesEigen, HMMStatesEigen)):

    @property
    def rs(self):
        return np.array([d.r for d in self.dur_distns])

    @property
    def ps(self):
        return np.array([d.p for d in self.dur_distns])

    ### HMM embedding parameters

    @abc.abstractproperty
    def hmm_trans_matrix(self):
        pass

    @property
    def hmm_aBl(self):
        if self._hmm_aBl is None:
            self._hmm_aBl = self.aBl.repeat(self.rs,axis=1)
        return self._hmm_aBl

    @property
    def hmm_pi_0(self):
        if not self.left_censoring:
            rs = self.rs
            starts = np.concatenate(((0,),rs.cumsum()[:-1]))
            pi_0 = np.zeros(rs.sum())
            pi_0[starts] = self.pi_0
            return pi_0
        else:
            return top_eigenvector(self.hmm_trans_matrix)

    def clear_caches(self):
        super(_HSMMStatesIntegerNegativeBinomialBase,self).clear_caches()
        self._hmm_aBl = None

    def _map_states(self):
        themap = np.arange(self.num_states).repeat(self.rs).astype('int32')
        self.stateseq = themap[self.stateseq]

    def generate_states(self):
        self.stateseq = sample_markov(
                T=self.T,trans_matrix=self.hmm_trans_matrix,
                init_state_distn=self.hmm_pi_0)
        self._map_states()

    def Viterbi_hmm(self):
        from hmm_messages_interface import viterbi
        self.stateseq = viterbi(
                self.hmm_trans_matrix,self.hmm_aBl,self.hmm_pi_0,
                np.empty(self.hmm_aBl.shape[0],dtype='int32'))
        self._map_states()

    def resample_hmm(self):
        alphan, self._normalizer = \
                HMMStatesEigen._messages_forwards_normalized(
                        self.hmm_trans_matrix,self.hmm_pi_0,self.hmm_aBl)
        self.stateseq = HMMStatesEigen._sample_backwards_normalized(
                alphan,self.hmm_trans_matrix.T.copy())
        self._map_states()

        self.alphan = alphan  # TODO remove

    def resample_hsmm(self):
        betal, betastarl = HSMMStatesEigen.messages_backwards(self)
        HMMStatesEigen.sample_forwards(betal,betastarl)

    def resample(self):
        self.resample_hmm()

    def Viterbi(self):
        self.Viterbi_hmm()

    def hmm_messages_forwards_log(self):
        return HMMStatesEigen._messages_forwards_log(
                self.hmm_trans_matrix,self.hmm_pi_0,self.hmm_aBl)


class HSMMStatesIntegerNegativeBinomial(_HSMMStatesIntegerNegativeBinomialBase):
    @property
    def hmm_trans_matrix(self):
        return self.hmm_bwd_trans_matrix

    @property
    def hmm_bwd_trans_matrix(self):
        rs, ps = self.rs, self.ps
        starts, ends = cumsum(rs,strict=True), cumsum(rs,strict=False)
        trans_matrix = np.zeros((ends[-1],ends[-1]))

        enters = self.bwd_enter_rows
        for (i,j), Aij in np.ndenumerate(self.trans_matrix):
            block = trans_matrix[starts[i]:ends[i],starts[j]:ends[j]]
            block[-1,:] = Aij * (1-ps[i]) * enters[j]
            if i == j:
                block[...] += np.diag(np.repeat(ps[i],rs[i])) \
                        + np.diag(np.repeat(1-ps[i],rs[i]-1),k=1)

        assert np.allclose(trans_matrix.sum(1),1) or self.trans_matrix.shape == (1,1)
        return trans_matrix

    @property
    def bwd_enter_rows(self):
        return [stats.binom.pmf(np.arange(r)[::-1],r-1,p) for r,p in zip(self.rs,self.ps)]

    @property
    def hmm_fwd_trans_matrix(self):
        rs, ps = self.rs, self.ps
        starts, ends = cumsum(rs,strict=True), cumsum(rs,strict=False)
        trans_matrix = np.zeros((ends[-1],ends[-1]))

        exits = self.fwd_exit_cols
        for (i,j), Aij in np.ndenumerate(self.trans_matrix):
            block = trans_matrix[starts[i]:ends[i],starts[j]:ends[j]]
            block[:,0] = Aij * exits[i] * (1-ps[i])
            if i == j:
                block[...] += \
                        np.diag(np.repeat(ps[i],rs[i])) \
                        + np.diag(np.repeat(1-ps[i],rs[i]-1) * (1-exits[i][:-1]),k=1)

        assert np.allclose(trans_matrix.sum(1),1)
        assert (0 <= trans_matrix).all() and (trans_matrix <= 1.).all()
        return trans_matrix

    @property
    def fwd_exit_cols(self):
        return [(1-p)**(np.arange(r)[::-1]) for r,p in zip(self.rs,self.ps)]

    def messages_backwards2(self):
        # this method is just for numerical testing
        # returns HSMM messages using HMM embedding. the way of the future!
        Al = np.log(self.trans_matrix)
        T, num_states = self.T, self.num_states

        betal = np.zeros((T,num_states))
        betastarl = np.zeros((T,num_states))

        starts = cumsum(self.rs,strict=True)
        ends = cumsum(self.rs,strict=False)
        foo = np.zeros((num_states,ends[-1]))
        for idx, row in enumerate(self.bwd_enter_rows):
            foo[idx,starts[idx]:ends[idx]] = row
        bar = np.zeros_like(self.hmm_bwd_trans_matrix)
        for start, end in zip(starts,ends):
            bar[start:end,start:end] = self.hmm_bwd_trans_matrix[start:end,start:end]

        pmess = np.zeros(ends[-1])

        # betal[-1] is 0
        for t in range(T-1,-1,-1):
            pmess += self.hmm_aBl[t]
            betastarl[t] = logsumexp(np.log(foo) + pmess, axis=1)
            betal[t-1] = logsumexp(Al + betastarl[t], axis=1)

            pmess = logsumexp(np.log(bar) + pmess, axis=1)
            pmess[ends-1] = np.logaddexp(pmess[ends-1],betal[t-1] + np.log(1-self.ps))
        betal[-1] = 0.

        return betal, betastarl

    ### NEW

    def meanfieldupdate(self):
        return self.meanfieldupdate_sampling()
        # return self.meanfieldupdate_Estep()

    def meanfieldupdate_sampling(self):
        from ..util.general import count_transitions
        num_r_samples = self.model.mf_num_samples \
                if hasattr(self.model,'mf_num_samples') else 10

        self.expected_states = np.zeros((self.T,self.num_states))
        self.expected_transcounts = np.zeros((self.num_states,self.num_states))
        self.expected_durations = np.zeros((self.num_states,self.T))

        eye = np.eye(self.num_states)/num_r_samples
        for i in range(num_r_samples):
            self.model._resample_from_mf()
            self.clear_caches()

            self.resample()

            self.expected_states += eye[self.stateseq]
            self.expected_transcounts += \
                count_transitions(self.stateseq_norep,minlength=self.num_states)\
                / num_r_samples
            for state in range(self.num_states):
                self.expected_durations[state] += \
                    np.bincount(
                            self.durations_censored[self.stateseq_norep == state],
                            minlength=self.T)[:self.T].astype(np.double)/num_r_samples

    def meanfieldupdate_Estep(self):
        # TODO bug in here? it's not as good as sampling
        num_r_samples = self.model.mf_num_samples \
                if hasattr(self.model,'mf_num_samples') else 10
        num_stateseq_samples_per_r = self.model.mf_num_stateseq_samples_per_r \
                if hasattr(self.model,'mf_num_stateseq_samples_per_r') else 1

        self.expected_states = np.zeros((self.T,self.num_states))
        self.expected_transcounts = np.zeros((self.num_states,self.num_states))
        self.expected_durations = np.zeros((self.num_states,self.T))

        mf_aBl = self.mf_aBl

        for i in range(num_r_samples):
            for d in self.dur_distns:
                d._resample_r_from_mf()
            self.clear_caches()

            trans = self.mf_bwd_trans_matrix  # TODO check this
            init = self.hmm_mf_bwd_pi_0
            aBl = mf_aBl.repeat(self.rs,axis=1)

            hmm_alphal, hmm_betal = HMMStatesEigen._messages_log(self,trans,init,aBl)

            # collect stateseq and transitions statistics from messages
            hmm_expected_states, hmm_expected_transcounts, normalizer = \
                    HMMStatesPython._expected_statistics_from_messages(
                            trans,aBl,hmm_alphal,hmm_betal)
            expected_states, expected_transcounts, _ \
                    = self._hmm_stats_to_hsmm_stats(
                            hmm_expected_states, hmm_expected_transcounts, normalizer)

            self.expected_states += expected_states / num_r_samples
            self.expected_transcounts += expected_transcounts / num_r_samples

            # collect duration statistics by sampling from messages
            for j in range(num_stateseq_samples_per_r):
                self._resample_from_mf(trans,init,aBl,hmm_alphal,hmm_betal)
                for state in range(self.num_states):
                    self.expected_durations[state] += \
                        np.bincount(
                                self.durations_censored[self.stateseq_norep == state],
                                minlength=self.T)[:self.T].astype(np.double) \
                            /(num_r_samples*num_stateseq_samples_per_r)

    def _hmm_stats_to_hsmm_stats(self,hmm_expected_states,hmm_expected_transcounts,normalizer):
        rs = self.rs
        starts = np.concatenate(((0,),np.cumsum(rs[:-1])))
        dotter = np.zeros((rs.sum(),len(rs)))
        for idx, (start, length) in enumerate(zip(starts,rs)):
            dotter[start:start+length,idx] = 1.

        expected_states = hmm_expected_states.dot(dotter)
        expected_transcounts = dotter.T.dot(hmm_expected_transcounts).dot(dotter)
        expected_transcounts.flat[::expected_transcounts.shape[0]+1] = 0

        return expected_states, expected_transcounts, normalizer

    def _resample_from_mf(self,trans,init,aBl,hmm_alphal,hmm_betal):
        self.stateseq = HMMStatesEigen._sample_forwards_log(
                hmm_betal,trans,init,aBl)
        self._map_states()

    @property
    def hmm_mf_bwd_pi_0(self):
        rs = self.rs
        starts = np.concatenate(((0,),rs.cumsum()[:-1]))
        mf_pi_0 = np.zeros(rs.sum())
        mf_pi_0[starts] = self.mf_pi_0
        return mf_pi_0

    @property
    def mf_bwd_trans_matrix(self):
        rs = self.rs
        starts, ends = cumsum(rs,strict=True), cumsum(rs,strict=False)
        trans_matrix = np.zeros((ends[-1],ends[-1]))

        Elnps, Eln1mps = zip(*[d._fixedr_distns[d.ridx]._mf_expected_statistics() for d in self.dur_distns])
        Eps, E1mps = np.exp(Elnps), np.exp(Eln1mps) # NOTE: actually exp(E[ln(p)]) etc

        enters = self.mf_bwd_enter_rows(rs,Eps,E1mps)
        for (i,j), Aij in np.ndenumerate(self.mf_trans_matrix):
            block = trans_matrix[starts[i]:ends[i],starts[j]:ends[j]]
            block[-1,:] = Aij * eE1mps[i] * enters[j]
            if i == j:
                block[...] += np.diag(np.repeat(eEps[i],rs[i])) \
                        + np.diag(np.repeat(eE1mps[i],rs[i]-1),k=1)

        assert np.all(trans_matrix >= 0)
        return trans_matrix

    def mf_bwd_enter_rows(self,rs,Elnps,Eln1mps):
        return [self._mf_binom(np.arange(r)[::-1],r-1,Ep,E1mp)
            for r,Ep,E1mp in zip(rs,Eps,E1mps)]

    @staticmethod
    def _mf_binom(k,n,p1,p2):
        return np.exp(special.gammaln(n+1) - special.gammaln(k+1) - special.gammaln(n-k+1) \
                + k*p1 + (n-k)*p2)

class HSMMStatesIntegerNegativeBinomialVariant(_HSMMStatesIntegerNegativeBinomialBase):
    @property
    def hmm_trans_matrix(self):
        return self.hmm_bwd_trans_matrix

    @property
    def hmm_bwd_trans_matrix(self):
        rs, ps = self.rs, self.ps
        starts, ends = cumsum(rs,strict=True), cumsum(rs,strict=False)
        trans_matrix = np.zeros((rs.sum(),rs.sum()))

        for (i,j), Aij in np.ndenumerate(self.trans_matrix):
            block = trans_matrix[starts[i]:ends[i],starts[j]:ends[j]]
            block[-1,0] = Aij * (1-ps[i])
            if i == j:
                block[...] += np.diag(np.repeat(ps[i],rs[i])) \
                        + np.diag(np.repeat(1-ps[i],rs[i]-1),k=1)

        assert np.allclose(trans_matrix.sum(1),1)
        return trans_matrix


class HSMMStatesIntegerNegativeBinomialSeparateTrans(
        _SeparateTransMixin,
        HSMMStatesIntegerNegativeBinomial):
    pass


class HSMMStatesDelayedIntegerNegativeBinomial(HSMMStatesIntegerNegativeBinomial):
    @property
    def hmm_trans_matrix(self):
        # return self.hmm_trans_matrix_orig
        return self.hmm_trans_matrix_2

    @property
    def hmm_trans_matrix_orig(self):
        rs, ps, delays = self.rs, self.ps, self.delays
        starts, ends = cumsum(rs+delays,strict=True), cumsum(rs+delays,strict=False)
        trans_matrix = np.zeros((ends[-1],ends[-1]))

        enters = self.bwd_enter_rows
        for (i,j), Aij in np.ndenumerate(self.trans_matrix):
            block = trans_matrix[starts[i]:ends[i],starts[j]:ends[j]]

            if delays[i] == 0:
                block[-1,:rs[j]] = Aij * enters[j] * (1-ps[i])
            else:
                block[-1,:rs[j]] = Aij * enters[j]

            if i == j:
                block[:rs[i],:rs[i]] += \
                    np.diag(np.repeat(ps[i],rs[i])) + np.diag(np.repeat(1-ps[i],rs[i]-1),k=1)
                if delays[i] > 0:
                    block[rs[i]-1,rs[i]] = (1-ps[i])
                    block[rs[i]:,rs[i]:] = np.eye(delays[i],k=1)

        assert np.allclose(trans_matrix.sum(1),1.)
        return trans_matrix

    @property
    def hmm_trans_matrix_1(self):
        rs, ps, delays = self.rs, self.ps, self.delays
        starts, ends = cumsum(rs+delays,strict=True), cumsum(rs+delays,strict=False)
        trans_matrix = np.zeros((ends[-1],ends[-1]))

        enters = self.bwd_enter_rows
        for (i,j), Aij in np.ndenumerate(self.trans_matrix):
            block = trans_matrix[starts[i]:ends[i],starts[j]:ends[j]]

            block[-1,:rs[j]] = Aij * enters[j] * (1-ps[i])

            if i == j:
                block[-rs[i]:,-rs[i]:] += \
                    np.diag(np.repeat(ps[i],rs[i])) + np.diag(np.repeat(1-ps[i],rs[i]-1),k=1)
                if delays[i] > 0:
                    block[:delays[i]:,:delays[i]] = np.eye(delays[i],k=1)
                    block[delays[i]-1,delays[i]] = 1

        assert np.allclose(trans_matrix.sum(1),1.)
        return trans_matrix

    @property
    def hmm_trans_matrix_2(self):
        rs, ps, delays = self.rs, self.ps, self.delays
        starts, ends = cumsum(rs+delays,strict=True), cumsum(rs+delays,strict=False)
        trans_matrix = np.zeros((ends[-1],ends[-1]))

        enters = self.bwd_enter_rows
        for (i,j), Aij in np.ndenumerate(self.trans_matrix):
            block = trans_matrix[starts[i]:ends[i],starts[j]:ends[j]]

            block[-1,0] = Aij * (1-ps[i])

            if i == j:
                block[-rs[i]:,-rs[i]:] += \
                    np.diag(np.repeat(ps[i],rs[i])) + np.diag(np.repeat(1-ps[i],rs[i]-1),k=1)
                if delays[i] > 0:
                    block[:delays[i]:,:delays[i]] = np.eye(delays[i],k=1)
                    block[delays[i]-1,-rs[i]:] = enters[i]

        assert np.allclose(trans_matrix.sum(1),1.)
        return trans_matrix

    @property
    def hmm_aBl(self):
        if self._hmm_aBl is None:
            self._hmm_aBl = self.aBl.repeat(self.rs+self.delays,axis=1)
        return self._hmm_aBl

    @property
    def hmm_pi_0(self):
        if self.left_censoring:
            raise NotImplementedError
        else:
            rs, delays = self.rs, self.delays
            starts = np.concatenate(((0,),(rs+delays).cumsum()[:-1]))
            pi_0 = np.zeros((rs+delays).sum())
            pi_0[starts] = self.pi_0
            return pi_0

    @property
    def delays(self):
        return np.array([d.delay for d in self.dur_distns])

    def _map_states(self):
        themap = np.arange(self.num_states).repeat(self.rs+self.delays).astype('int32')
        self.stateseq = themap[self.stateseq]

class HSMMStatesTruncatedIntegerNegativeBinomial(HSMMStatesDelayedIntegerNegativeBinomial):
    @property
    def bwd_enter_rows(self):
        As = [np.diag(np.repeat(p,r)) + np.diag(np.repeat(1-p,r-1),k=1) for r,p in zip(self.rs,self.ps)]
        enters = [stats.binom.pmf(np.arange(r)[::-1],r-1,p) for A,r,p in zip(As,self.rs,self.ps)]
        # norms = [sum(v.dot(np.linalg.matrix_power(A,d))[-1]*(1-p) for d in range(delay))
        #         for A,v,p,delay in zip(As,enters,self.ps,self.delays)]
        # enters = [v.dot(np.linalg.matrix_power(A,self.delays[state])) / (1.-norm)
        enters = [v.dot(np.linalg.matrix_power(A,self.delays[state]))
                for state, (A,v) in enumerate(zip(As,enters))]
        return [v / v.sum() for v in enters] # this should just be for numerical purposes

class HSMMStatesDelayedIntegerNegativeBinomialSeparateTrans(
        _SeparateTransMixin,
        HSMMStatesDelayedIntegerNegativeBinomial):
    pass

class HSMMStatesTruncatedIntegerNegativeBinomialSeparateTrans(
        _SeparateTransMixin,
        HSMMStatesTruncatedIntegerNegativeBinomial):
    pass

