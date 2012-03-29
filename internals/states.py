import numpy as np
from numpy import newaxis as na
from numpy.random import random
import scipy.weave

from pyhsmm.util.stats import sample_discrete
from pyhsmm.util import general as util # perhaps a confusing name :P

import os
eigen_code_dir = os.path.join(os.path.dirname(__file__),'cpp_eigen_code/')

# TODO move eigen code loading into global code, string interpolation can still
# be local
with open(eigen_code_dir + 'hsmm_sample_forwards.cpp') as infile:
    hsmm_sample_forwards_codestr = infile.read()

with open(eigen_code_dir + 'hsmm_messages_backwards.cpp') as infile:
    hsmm_messages_backwards_codestr = infile.read()

with open(eigen_code_dir + 'hmm_messages_forwards.cpp') as infile:
    hmm_messages_forwards_codestr = infile.read()

class hsmm_states_python(object):
    '''
    HSMM states distribution class. Connects the whole model.

    Parameters include:

    T
    state_dim
    obs_distns
    dur_distns
    transition_distn
    initial_distn
    trunc

    stateseq
    durations
    stateseq_norep
    '''

    # these are convenient
    durations = None
    stateseq_norep = None

    def __init__(self,T,state_dim,obs_distns,dur_distns,transition_distn,initial_distn,
            stateseq=None,trunc=None,data=None,**kwargs):
        # TODO T parameter only makes sense with censoring. it should be
        # removed.
        self.T = T
        self.state_dim = state_dim
        self.obs_distns = obs_distns
        self.dur_distns = dur_distns
        self.transition_distn = transition_distn
        self.initial_distn = initial_distn
        self.trunc = T if trunc is None else trunc
        self.data = data

        # this arg is for initialization heuristics which may pre-determine the
        # state sequence
        if stateseq is not None:
            self.stateseq = stateseq
            # gather durations and stateseq_norep
            self.stateseq_norep, self.durations = util.rle(stateseq)
        else:
            if data is not None:
                self.resample()
            else:
                self.generate_states()

    def generate(self):
        self.generate_states()
        return self.generate_obs()

    def generate_obs(self):
        obs = []
        for state,dur in zip(self.stateseq_norep,self.durations):
            obs.append(self.obs_distns[state].rvs(size=int(dur)))
        return np.concatenate(obs)[:self.T] # censoring

    def generate_states(self):
        idx = 0
        nextstate_distr = self.initial_distn.pi_0
        A = self.transition_distn.A

        stateseq = -1*np.ones(self.T,dtype=np.int32)
        stateseq_norep = []
        durations = []

        while idx < self.T:
            # sample a state
            state = sample_discrete(nextstate_distr)
            # sample a duration for that state
            duration = self.dur_distns[state].rvs()
            # save everything
            stateseq_norep.append(state)
            durations.append(duration)
            stateseq[idx:idx+duration] = state
            # set up next state distribution
            nextstate_distr = A[state,]
            # update index
            idx += duration

        self.stateseq_norep = np.array(stateseq_norep,dtype=np.int32)
        self.durations = np.array(durations,dtype=np.int32)
        self.stateseq = stateseq

        # NOTE self.durations.sum() >= self.T since self.T is the censored
        # length

        assert len(self.stateseq_norep) == len(self.durations)
        assert (self.stateseq >= 0).all()

    def resample(self):
        if self.data is not None:
            data = self.data
        else:
            print 'ERROR: can only call resample on %s instances with data' % type(self)
            return

        # generate duration pmf and sf values
        # generate and cache iid likelihood values, used in cumulative_likelihood functions
        possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
        aDl = np.zeros((self.T,self.state_dim))
        aDsl = np.zeros((self.T,self.state_dim))
        self.aBl = self.get_aBl(data)
        for idx, dur_distn in enumerate(self.dur_distns):
            aDl[:,idx] = dur_distn.log_pmf(possible_durations)
            aDsl[:,idx] = dur_distn.log_sf(possible_durations)
        # run backwards message passing
        betal, betastarl = self.messages_backwards(np.log(self.transition_distn.A),aDl,aDsl,self.trunc)
        # sample forwards
        self.sample_forwards(betal,betastarl)
        # save these for testing convenience
        self.aDl = aDl
        self.aDsl = aDsl

    def messages_backwards(self,Al,aDl,aDsl,trunc):
        T = aDl.shape[0]
        state_dim = Al.shape[0]
        betal = np.zeros((T,state_dim),dtype=np.float64)
        betastarl = np.zeros((T,state_dim),dtype=np.float64)
        assert betal.shape == aDl.shape == aDsl.shape

        for t in xrange(T-1,-1,-1):
            np.logaddexp.reduce(betal[t:t+trunc] + self.cumulative_likelihoods(t,t+trunc) + aDl[:min(trunc,T-t)],axis=0, out=betastarl[t])
            if T-t < trunc:
                np.logaddexp(betastarl[t], self.likelihood_block(t,None) + aDsl[T-t-1], betastarl[t]) # censoring calc, -1 for zero indexing of aDl compared to arguments to log_sf
            np.logaddexp.reduce(betastarl[t] + Al,axis=1,out=betal[t-1])
        betal[-1] = 0. # overwritten in last loop for loop expression simplicity, set it back to 0 here

        assert not np.isnan(betal).any()
        assert not np.isinf(betal[0]).all()

        return betal, betastarl

    def get_aBl(self,data):
        aBl = np.zeros((data.shape[0],self.state_dim))
        for idx in xrange(self.state_dim):
            aBl[:,idx] = self.obs_distns[idx].log_likelihood(data)
        return aBl

    def cumulative_likelihoods(self,start,stop):
        return np.cumsum(self.aBl[start:stop],axis=0)

    def cumulative_likelihood_state(self,start,stop,state):
        return np.cumsum(self.aBl[start:stop,state])

    def likelihood_block(self,start,stop):
        # does not include the stop index
        return np.sum(self.aBl[start:stop],axis=0)

    def likelihood_block_state(self,start,stop,state):
        return np.sum(self.aBl[start:stop,state])

    def sample_forwards(self,betal,betastarl):
        stateseq = self.stateseq = np.zeros(self.T,dtype=np.int32)
        durations = []
        stateseq_norep = []

        idx = 0
        A = self.transition_distn.A
        nextstate_unsmoothed = self.initial_distn.pi_0

        apmf = np.zeros((self.state_dim,self.T))
        arg = np.arange(1,self.T+1)
        for state_idx, dur_distn in enumerate(self.dur_distns):
            apmf[state_idx] = dur_distn.pmf(arg)

        while idx < self.T:
            logdomain = betastarl[idx] - np.amax(betastarl[idx])
            nextstate_distr = np.exp(logdomain) * nextstate_unsmoothed
            if (nextstate_distr == 0.).all():
                # this is a numerical issue; no good answer, so we'll just follow the messages.
                nextstate_distr = np.exp(logdomain)
            state = sample_discrete(nextstate_distr)
            assert len(stateseq_norep) == 0 or state != stateseq_norep[-1]

            durprob = random()
            dur = 0 # always incremented at least once
            prob_so_far = 0.0
            while durprob > 0:
                assert dur < 2*self.T # hacky infinite loop check
                #assert self.dur_distns[state].pmf(dur+1) == apmf[state,dur]
                p_d_marg = apmf[state,dur] if dur < self.T else 1. # note funny indexing: dur variable is 1 less than actual dur we're considering
                assert not np.isnan(p_d_marg)
                assert p_d_marg >= 0
                if p_d_marg == 0:
                    dur += 1
                    continue
                if idx+dur < self.T:
                    mess_term = np.exp(self.likelihood_block_state(idx,idx+dur+1,state) + betal[idx+dur,state] - betastarl[idx,state]) # TODO unnecessarily slow for subhmms
                    p_d = mess_term * p_d_marg
                    #print 'dur: %d, durprob: %f, p_d_marg: %f, p_d: %f' % (dur+1,durprob,p_d_marg,p_d)
                    prob_so_far += p_d
                else:
                    # we're out of data, so we need to sample a duration
                    # conditioned on having lasted at least this long. the
                    # likelihood contributes the same to all possibilities, so
                    # we can just sample from the prior (conditioned on it being
                    # at least this long).
                    arg = np.arange(dur+1,2*self.T) # 2*T is just a guessed upper bound, +1 because 'dur' is one less than the duration we're actually considering
                    remaining = dur_distn.pmf(arg)
                    therest = sample_discrete(remaining)
                    dur = dur + therest
                    durprob = -1 # just to get us out of loop

                assert not np.isnan(p_d)
                durprob -= p_d
                dur += 1

            assert dur > 0

            stateseq[idx:idx+dur] = state
            stateseq_norep.append(state)
            assert len(stateseq_norep) < 2 or stateseq_norep[-1] != stateseq_norep[-2]
            durations.append(dur)

            nextstate_unsmoothed = A[state,:]

            idx += dur

        self.durations = np.array(durations,dtype=np.int32)
        self.stateseq_norep = np.array(stateseq_norep,dtype=np.int32)

    def plot(self,colors_dict=None):
        from matplotlib import pyplot as plt
        X,Y = np.meshgrid(np.hstack((0,self.durations.cumsum())),(0,1))

        if colors_dict is not None:
            C = np.array([[colors_dict[state] for state in self.stateseq_norep]])
        else:
            C = self.stateseq_norep[na,:]

        plt.pcolor(X,Y,C,vmin=0,vmax=1)
        plt.ylim((0,1))
        plt.xlim((0,self.T))
        plt.yticks([])


class hsmm_states_eigen(hsmm_states_python):
    def __init__(self,T,state_dim,obs_distns,dur_distns,transition_distn,initial_distn,stateseq=None,trunc=None,data=None,**kwargs):
        self.T = T
        self.state_dim = state_dim
        self.obs_distns = obs_distns
        self.dur_distns = dur_distns
        self.transition_distn = transition_distn
        self.initial_distn = initial_distn
        self.trunc = T if trunc is None else trunc
        self.data = data

        self.sample_forwards_codestr = hsmm_sample_forwards_codestr % {'M':state_dim,'T':T}

        self.messages_backwards_codestr = hsmm_messages_backwards_codestr % {'M':state_dim,'T':T}

        # this arg is for initialization heuristics which may pre-determine the state sequence
        if stateseq is not None:
            self.stateseq = stateseq
            # gather durations and stateseq_norep
            self.stateseq_norep, self.durations = util.rle(stateseq)
        else:
            if data is not None:
                self.resample()
            else:
                self.generate_states()

    def sample_forwards(self,betal,betastarl):
        aBl = self.aBl
        # stateseq = np.array(self.stateseq,dtype=np.int32)
        stateseq = np.zeros(betal.shape[0],dtype=np.int32)
        A = self.transition_distn.A
        pi0 = self.initial_distn.pi_0

        apmf = np.zeros((self.state_dim,self.T))
        arg = np.arange(1,self.T+1)
        for state_idx, dur_distn in enumerate(self.dur_distns):
            apmf[state_idx] = dur_distn.pmf(arg)

        scipy.weave.inline(self.sample_forwards_codestr,['betal','betastarl','aBl','stateseq','A','pi0','apmf'],headers=['<Eigen/Core>'],include_dirs=['/usr/local/include/eigen3'],extra_compile_args=['-O3'])#,'-march=native'])

        self.stateseq_norep, self.durations = util.rle(stateseq)
        self.stateseq = stateseq

    def messages_backwards_NOTUSED(self,Al,aDl,aDsl,trunc):
        # this isn't actually used: it is the same speed or slower than the
        # python/numpy version, since the code is very vectorized
        A = np.exp(Al).T.copy()
        mytrunc = trunc;
        aBl = self.aBl
        betal = np.zeros((self.T,self.state_dim))
        betastarl = np.zeros((self.T,self.state_dim))
        scipy.weave.inline(self.messages_backwards_codestr,['A','mytrunc','betal','betastarl','aDl','aBl','aDsl'],headers=['<Eigen/Core>','<limits>'],include_dirs=['/usr/local/include/eigen3'],extra_compile_args=['-O3','-march=native'])
        return betal, betastarl

    # TODO could also write Eigen version of the generate() methods


class hmm_states_python(object): 
    def __init__(self,T,state_dim,obs_distns,transition_distn,initial_distn,stateseq=None,data=None,**kwargs):
        self.state_dim = state_dim
        self.obs_distns = obs_distns
        self.transition_distn = transition_distn
        self.initial_distn = initial_distn

        self.T = T
        self.data = data

        if stateseq is not None:
            self.stateseq = stateseq
        else:
            if data is not None:
                self.resample()
            else:
                self.generate_states()

    def generate_states(self):
        T = self.T
        stateseq = np.zeros(T,dtype=np.int32)
        nextstate_distn = self.initial_distn.pi_0
        A = self.transition_distn.A

        for idx in xrange(T):
            stateseq[idx] = sample_discrete(nextstate_distn)
            nextstate_distn = A[stateseq[idx]]

        self.stateseq = stateseq
        return stateseq

    def generate_obs(self):
        obs = []
        for state in self.stateseq:
            obs.append(self.obs_distns[state].rvs(size=1))
        return np.concatenate(obs)

    def generate(self):
        self.generate_states()
        return self.generate_obs()

    def messages_forwards(self,aBl):
        # note: this method never uses self.T
        # or self.data
        T = aBl.shape[0]
        alphal = np.zeros((T,self.state_dim))
        Al = np.log(self.transition_distn.A)

        alphal[0] = np.log(self.initial_distn.pi_0) + aBl[0]

        for t in xrange(T-1):
            alphal[t+1] = np.logaddexp.reduce(alphal[t] + Al.T,axis=1) + aBl[t+1]

        return alphal

    def messages_backwards(self,aBl):
        betal = np.zeros(aBl.shape)
        Al = np.log(self.transition_distn.A)
        T = aBl.shape[0]

        for t in xrange(T-2,-1,-1):
            np.logaddexp.reduce(Al + betal[t+1] + aBl[t+1],axis=1,out=betal[t])

        return betal

    def sample_forwards(self,aBl,betal):
        T = aBl.shape[0]
        stateseq = np.zeros(T,dtype=np.int32)
        nextstate_unsmoothed = self.initial_distn.pi_0
        A = self.transition_distn.A

        for idx in xrange(T):
            logdomain = betal[idx] + aBl[idx]
            stateseq[idx] = sample_discrete(nextstate_unsmoothed * np.exp(logdomain - np.amax(logdomain)))
            nextstate_unsmoothed = A[stateseq[idx]]

        return stateseq

    def resample(self):
        if self.data is None:
            print 'ERROR: can only call resample on %s instances with data' % type(self)
        else:
            data = self.data

        aBl = self.get_aBl(data)
        betal = self.messages_backwards(aBl)
        self.stateseq = self.sample_forwards(aBl,betal)
        return self.stateseq

    def get_aBl(self,data):
        # note: this method never uses self.T
        aBl = np.zeros((data.shape[0],self.state_dim))
        for idx, obs_distn in enumerate(self.obs_distns):
            aBl[:,idx] = obs_distn.log_likelihood(data)
        return aBl

    def plot(self,colors_dict=None):
        from matplotlib import pyplot as plt
        from pyhsmm.util.general import rle
        states,durations = rle(self.stateseq)
        X,Y = np.meshgrid(np.hstack((0,durations.cumsum())),(0,1))

        if colors_dict is not None:
            C = np.array([[colors_dict[state] for state in states]])
        else:
            C = states

        plt.pcolor(X,Y,C,vmin=0,vmax=1)
        plt.ylim((0,1))
        plt.xlim((0,self.T))
        plt.yticks([])


class hmm_states_eigen(hmm_states_python):
    def __init__(self,T,state_dim,obs_distns,transition_distn,initial_distn,stateseq=None,data=None,**kwargs):
        self.state_dim = state_dim
        self.obs_distns = obs_distns
        self.transition_distn = transition_distn
        self.initial_distn = initial_distn
        self.T = T
        self.data = data

        if stateseq is not None:
            self.stateseq = stateseq
        else:
            if data is not None:
                self.resample()
            else:
                self.generate_states()

    def messages_forwards(self,aBl):
        # note: this method never uses self.T
        T = aBl.shape[0]
        alphal = np.zeros((T,self.state_dim))
        alphal[0] = np.log(self.initial_distn.pi_0) + aBl[0]
        A = self.transition_distn.A # eigen sees this transposed

        scipy.weave.inline(hmm_messages_forwards_codestr % {'M':self.state_dim,'T':T},['A','alphal','aBl','T'],headers=['<Eigen/Core>','<limits>'],include_dirs=['/usr/local/include/eigen3'],extra_compile_args=['-O3'])

        assert not np.isnan(alphal).any()
        return alphal

    def messages_backwards(self,aBl):
        # TODO write Eigen version
        return hmm_states_python.messages_backwards(self,aBl)

    def sample_forwards(self,aBl,betal):
        # TODO TODO write eigen version
        self.stateseq = hmm_states_python.sample_forwards(self,aBl,betal)
        return self.stateseq

    # TODO also write eigen versions of generate and generate_obs


hsmm_states = hsmm_states_python
hmm_states = hmm_states_python

def use_eigen(useit=True):
    global hsmm_states, hmm_states
    if useit:
        hsmm_states = hsmm_states_eigen
        hmm_states = hmm_states_eigen
    else:
        hsmm_states = hsmm_states_python
        hmm_states = hmm_states_python

