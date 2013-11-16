from __future__ import division

# these things are bits of old code or need updating

##################################
#  Geometric tail approximation  #
##################################

# NOTE: last I checked, this code works fine, but it was cluttering things up
# and I didn't use it much

class HSMMGeoApproximation(HSMM):
    _states_class = states.HSMMStatesGeoApproximation

class HSMMStatesGeoApproximation(HSMMStatesPython):
    def _get_hmm_transition_matrix(self):
        trunc = self.trunc if self.trunc is not None else self.T
        state_dim = self.state_dim
        hmm_A = self.trans_matrix.copy()
        hmm_A.flat[::state_dim+1] = 0
        thediag = np.array([np.exp(d.log_pmf(trunc+1)-d.log_pmf(trunc))[0] for d in self.dur_distns])
        assert (thediag < 1).all(), 'truncation is too small!'
        hmm_A *= ((1-thediag)/hmm_A.sum(1))[:,na]
        hmm_A.flat[::state_dim+1] = thediag
        return hmm_A

    def messages_backwards(self):
        aDl, aDsl, Al = self.aDl, self.aDsl, np.log(self.trans_matrix)
        trunc = self.trunc if self.trunc is not None else self.T
        T,state_dim = aDl.shape

        assert trunc > 1

        aBl = self.aBl/self.temp if self.temp is not None else self.aBl
        hmm_betal = HMMStatesEigen._messages_backwards(self._get_hmm_transition_matrix(),aBl)
        assert not np.isnan(hmm_betal).any()

        betal = np.zeros((T,state_dim),dtype=np.float64)
        betastarl = np.zeros_like(betal)

        for t in xrange(T-1,-1,-1):
            np.logaddexp.reduce(betal[t:t+trunc] + self.cumulative_likelihoods(t,t+trunc)
                    + aDl[:min(trunc,T-t)],axis=0, out=betastarl[t])
            if t+trunc < T:
                np.logaddexp(betastarl[t], self.likelihood_block(t,t+trunc+1) + aDsl[trunc -1]
                        + hmm_betal[t+trunc], out=betastarl[t])
            if T-t < trunc and self.right_censoring:
                np.logaddexp(betastarl[t], self.likelihood_block(t,None) + aDsl[T-t -1], betastarl[t])
            np.logaddexp.reduce(betastarl[t] + Al,axis=1,out=betal[t-1])
        betal[-1] = 0.

        return betal, betastarl

#################
#  old subhmms  #
#################

### states

import numpy as np
na = np.newaxis

from pyhsmm.internals.states import HSMMStatesPython, HSMMStatesPossibleChangepoints
from pyhsmm.util.general import rle

class HSMMSubHMMStates(HSMMStatesPython):
    # NOTE: can't extend the eigen version because its sample_forwards depends
    # on aBl being iid (doesnt call the sub-methods)
    # TODO due to parent sample_forwards calling likelihood_block_state
    # repeatedly with incremental steps, should implement some caching or
    # something in that function
    def __init__(self,T,hmms,dur_distns,transition_distn,initial_distn,
            initialize_from_prior=True,
            trunc=None,data=None,stateseq=None,substateseqs=None,
            censoring=True,**kwargs):
        self.T = T
        self.state_dim = len(hmms)
        self.hmms = hmms # list of pyhsmm.plugins.subhmms.models.SubHMM instances
        self.censoring = censoring

        self.dur_distns = dur_distns
        self.transition_distn = transition_distn
        self.initial_distn = initial_distn

        self.trunc = trunc if trunc is not None else T

        self.data = data

        self.substates = [] # pyhsmm.internals.states.hmm_states instances

        if stateseq is not None:
            assert len(stateseq) == T
            self.stateseq = stateseq
            self.stateseq_norep, self.durations = map(lambda x: np.array(x,dtype=np.int32),rle(stateseq))
        else:
            if data is not None and not initialize_from_prior:
                self._resample_superstates()
            else:
                self._generate_superstates()

        # TODO this logic seems messy
        # TODO TODO and wrong
        if substateseqs is not None:
            assert data is not None, 'havent implemented this case'
            assert (np.array([len(ss) for ss in substateseqs]) == self.durations).all()
            start_indices = np.concatenate(([0],np.cumsum(self.durations[:-1])))
            datas = [data[startidx:startidx+dur] for startidx,dur in zip(start_indices,self.durations)]
            for substateseq, s, d in zip(substateseqs,self.stateseq_norep,datas):
                self.hmms[s].add_data(data=d,stateseq=substateseq,
                        initialize_from_prior=initialize_from_prior)
                self.substates.append(self.hmms[s].states_list[-1])
        else:
            if data is not None and not initialize_from_prior:
                self._resample_substates()
            else:
                self._generate_substates()
                if data is not None:
                    start_indices = np.concatenate(([0],np.cumsum(self.durations[:-1])))
                    datas = [data[startidx:startidx+dur] for startidx,dur in zip(start_indices,self.durations)]
                    for s,d in zip(self.substates,datas):
                        s.data = d

        assert len(self.substates) == len(self.stateseq_norep) == len(self.durations)
        assert (np.array([len(ss.stateseq) for ss in self.substates[:-1]]) == self.durations[:-1]).all()

    def generate_states(self):
        self._generate_superstates()
        self._generate_substates()

    def _generate_superstates(self):
        super(HSMMSubHMMStates,self).generate_states()

    def _generate_substates(self):
        for state, dur in zip(self.stateseq_norep,self.durations):
            self.hmms[state].generate(dur)
            self.substates.append(self.hmms[state].states_list[-1])
        # because of censoring, must truncate last substates
        assert self.T <= self.durations.sum()
        if self.T < self.durations.sum():
            keep = slice(None,self.T-self.durations.sum())
            self.substates[-1].data = self.substates[-1].data[keep]
            self.substates[-1].stateseq = self.substates[-1].stateseq[keep]
        assert sum(len(ss.data) for ss in self.substates) == self.T

    def generate_obs(self):
        # already generated in HMMs, so this is a little weird
        # just pull out that data here
        obs = []
        for subseq in self.substates:
            obs.append(subseq.data)
        obs = np.concatenate(obs)
        assert len(obs) == self.T
        return obs

    def resample(self):
        # empty current substates
        if len(self.substates) > 0:
            for state, subseq in zip(self.stateseq_norep,self.substates):
                self.hmms[state].states_list.remove(subseq)
            self.substates = []

        self._resample_superstates()
        self._resample_substates()

    def _resample_superstates(self):
        assert self.data is not None
        self.aBls = [hmm.get_aBl(self.data) for hmm in self.hmms]
        super(HSMMSubHMMStates,self).resample()
        del self.aBls

    def _resample_substates(self):
        # TODO don't need to recompute aBls if _resample_superstates did it
        assert len(self.substates) == 0
        indices = np.concatenate(([0],np.cumsum(self.durations[:-1])))
        for state, startidx, dur in zip(self.stateseq_norep,indices,self.durations):
            self.hmms[state].add_data(self.data[startidx:startidx+dur],initialize_from_prior=False)
            self.substates.append(self.hmms[state].states_list[-1])

    def cumulative_likelihood_state(self,start,stop,state):
        return np.logaddexp.reduce(self.hmms[state].messages_forwards(self.aBls[state][start:stop]),axis=1)

    def cumulative_likelihoods(self,start,stop):
        return np.hstack([self.cumulative_likelihood_state(start,stop,state)[:,na]
            for state in range(self.state_dim)])

    def likelihood_block_state(self,start,stop,state):
        return np.logaddexp.reduce(self.hmms[state].messages_forwards(self.aBls[state][start:stop])[-1])

    def likelihood_block(self,start,stop):
        return np.array([self.likelihood_block_state(start,stop,state)
            for state in range(self.state_dim)])

    def get_aBl(self,*args):
        # this method needs to exist because super().resample() will call it
        # but this class sets and uses self.aBls instead
        pass

    def get_states(self):
        return (self.stateseq, [s.stateseq for s in self.substates])

class HSMMSubHMMStatesPossibleChangepoints(HSMMSubHMMStates,HSMMStatesPossibleChangepoints):
    def __init__(self,changepoints,T,hmms,dur_distns,transition_distn,initial_distn,
            trunc=None,data=None,stateseq=None,substateseqs=None,initialize_from_prior=True,
            censoring=True):
        self.changepoints = changepoints
        self.startpoints = np.array([start for start,stop in changepoints],dtype=np.int32)
        self.blocklens = np.array([stop-start for start,stop in changepoints],dtype=np.int32)
        self.Tblock = len(changepoints) # number of blocks

        super(HSMMSubHMMStatesPossibleChangepoints,self).__init__(
                T,hmms,dur_distns,transition_distn,initial_distn,
                initialize_from_prior=initialize_from_prior,trunc=trunc,
                data=data,stateseq=stateseq,substateseqs=substateseqs,censoring=censoring)

    def messages_backwards(self,Al,aDl,aDsl,trunc):
        return HSMMStatesPossibleChangepoints.messages_backwards(self,Al,aDl,aDsl,trunc)

    def sample_forwards(self,betal,betastarl):
        return HSMMStatesPossibleChangepoints.sample_forwards(self,betal,betastarl)

    def block_cumulative_likelihoods(self,startblock,stopblock,possible_durations):
        # could recompute possible_durations given startblock, stopblock,
        # trunc/truncblock, and self.blocklens, but why redo that effort?
        return np.vstack([self.block_cumulative_likelihood_state(startblock,stopblock,state,possible_durations) for state in range(self.state_dim)]).T

    def block_cumulative_likelihood_state(self,startblock,stopblock,state,possible_durations):
        start = self.startpoints[startblock]
        stop = self.startpoints[stopblock] if stopblock < len(self.startpoints) else None
        return np.logaddexp.reduce(self.hmms[state].messages_forwards(self.aBls[state][start:stop])[possible_durations-1],axis=1)

    # TODO TODO does calling likelihood_block_state on this class work? should
    # probably implement it, but since HSMMStatesPossibleChangepoints doesn't
    # handle truncation in messages at the moment, it doesn't try to call
    # likelihod_block_state. i think calling likelihood-block_state may work
    # anyway, since it just uses self.aBl, but it will be unnecessarily slow
    # because it doesn't pay attention to block boundaries

    def generate(self):
        # TODO override generate someday
        raise NotImplementedError


#####################
#  C++/Eigen stuff  #
#####################

import os
this_code_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(this_code_dir,'cpp_eigen_code','hmm_messages_backwards.cpp')) as infile:
    hmm_messages_backwards_codestr = infile.read()
with open(os.path.join(this_code_dir,'cpp_eigen_code','hmm_sample_forwards.cpp')) as infile:
    hmm_sample_forwards_codestr = infile.read()


### models

import numpy as np
import random
from matplotlib import pyplot as plt

import pyhsmm
from pyhsmm.internals.states import HMMStates

from pyhsmm.plugins.subhmms.states import HSMMSubHMMStates
from pyhsmm.plugins.subhmms.states import HSMMSubHMMStatesPossibleChangepoints
from pyhsmm.util.plot import subplot_gridsize

# TODO add annealing to resample and/or resample_parallel
# TODO add a subhmm init method that uses LTR transitions
# TODO add conc. parameter resampling for subHMMs

class HSMMSubHMMs(pyhsmm.models.HSMM):
    def __init__(self,
            subalpha,subgamma,subkappa,subinit_state_concentration,
            obs_distnss,dur_distns,
            **kwargs):
        self.hmms = [
                SubHMM(
                    alpha=subalpha,
                    gamma=subgamma,
                    kappa=subkappa,
                    init_state_concentration=subinit_state_concentration,
                    obs_distns=obs_distns)
                for obs_distns in obs_distnss]

        super(HSMMSubHMMs,self).__init__(obs_distns=range(len(dur_distns)), # obs_distns is faked
                dur_distns=dur_distns,**kwargs)
        self.obs_distns = [] # empty list lets super.resample() work

    def add_data(self,data,stateseq=None,substateseqs=None,**kwargs):
        self.states_list.append(HSMMSubHMMStates(len(data),self.hmms,self.dur_distns,
                self.trans_distn,self.init_state_distn,trunc=self.trunc,data=data,
                stateseq=stateseq,substateseqs=substateseqs,**kwargs))

    def resample_model(self):
        super(HSMMSubHMMs,self).resample_model()
        ### resample subhmm parameters
        # subhmms already have all their observation sequences assigned to them
        # via all the [s.resample() for s in self.states_list] calls in super
        for hmm in self.hmms:
            hmm.resample_model()

    def generate(self,T,keep=True):
        tempstates = HSMMSubHMMStates(T,self.hmms,self.dur_distns,
                self.trans_distn,self.init_state_distn,trunc=self.trunc)

        return self._generate(tempstates,keep)

    def loglike(self,data,trunc=None):
        raise NotImplementedError

    def plot(self):
        self.plot_states()
        self.plot_observations()

    def plot_states(self):
        colors = self._get_colors()
        plt.figure()
        num_rows = len(self.states_list)
        for idx,s in enumerate(self.states_list):
            plt.subplot(num_rows,1,idx+1)
            s.plot(colors_dict = colors)
            plt.title('statesequence %d' % idx)

    def plot_observations(self,*args,**kwargs):
        plt.figure()
        used_superstates = self._get_used_states()
        gridsizerows, gridsizecols = subplot_gridsize(len(used_superstates))
        for idx,stateid in enumerate(used_superstates):
            plt.subplot(gridsizerows,gridsizecols,idx+1)
            self.hmms[stateid].plot_observations()
            plt.title('HMM %d' % stateid)

    def resample_model_parallel(self,numtoresample='all'):
        import pyhsmm.parallel as parallel
        if numtoresample == 'all':
            numtoresample = len(self.states_list)
        elif numtoresample =='engines':
            numtoresample = len(parallel.dv)

        ### resample super-parameters locally (except subhmm parameters)
        self.trans_distn.resample([s.stateseq for s in self.states_list])
        self.init_state_distn.resample([s.stateseq[0] for s in self.states_list])
        for state, distn in enumerate(self.dur_distns):
            distn.resample([s.durations[s.stateseq_norep == state] for s in self.states_list])

        ### choose which data to resample
        states_to_resample = random.sample(self.states_list,numtoresample)

        ### resample superstates and substates in parallel
        self._push_self_parallel(states_to_resample)
        self._build_states_parallel(states_to_resample)

        ### resample hmm parameters locally
        for hmm in self.hmms:
            hmm.resample_model(resample_stateseq=False)

        ### purge
        parallel.c.purge_results('all')

    def _push_self_parallel(self,states_to_resample):
        import pyhsmm.parallel as parallel
        for s in states_to_resample:
            self.states_list.remove(s)
            for superstate, substates in zip(s.stateseq_norep, s.substates):
                self.hmms[superstate].states_list.remove(substates)
        hmm_states_to_restore = []
        for h in self.hmms:
            hmm_states_to_restore.append(h.states_list)
            h.states_list = []
        states_to_restore = self.states_list
        self.states_list =[]

        parallel.dv.push({'global_model':self},block=True)

        self.states_list = states_to_restore
        for h,hlst in zip(self.hmms,hmm_states_to_restore):
            h.states_list = hlst

    def _build_states_parallel(self,states_to_resample):
        import pyhsmm.parallel as parallel
        if len(states_to_resample) > 0:
            raw_stateseq_tuples = parallel.build_states.map([s.data_id for s in states_to_resample])
            for data_id, (superstateseq, substateseqs) in raw_stateseq_tuples:
                self.add_data(data=parallel.alldata[data_id],
                            stateseq=superstateseq,
                            substateseqs=substateseqs)
                self.states_list[-1].data_id = data_id


class HSMMSubHMMsPossibleChangepoints(HSMMSubHMMs, pyhsmm.models.HSMMPossibleChangepoints):
    def add_data(self,data,changepoints,**kwargs):
        self.states_list.append(HSMMSubHMMStatesPossibleChangepoints(
            changepoints=changepoints,T=len(data),data=data,
            hmms=self.hmms,dur_distns=self.dur_distns,
            transition_distn=self.trans_distn,initial_distn=self.init_state_distn,
            **kwargs))

    def _build_states_parallel(self,states_to_resample):
        import pyhsmm.parallel as parallel
        raw_stateseq_tuples = parallel.build_states.map([s.data_id for s in states_to_resample])
        for data_id, (superstateseq, substateseqs) in raw_stateseq_tuples:
            self.add_data(data=parallel.alldata[data_id],
                          changepoints=parallel.allchangepoints[data_id],
                          stateseq=superstateseq,
                          substateseqs=substateseqs)
            self.states_list[-1].data_id = data_id

class SubHMM(pyhsmm.models.StickyHMM):
    # this class 'pulls up' these two methods from the hmm states class so that
    # we don't have to wrap every subsequence we consider into an hmm_states
    # instance
    # TODO dont need to re-write the code... just keep a states object dummy
    # around and call the methods directly
    trunc = None

    def add_data(self,data,stateseq=None,**kwargs):
        self.states_list.append(HMMStates(len(data),self.state_dim,self.obs_distns,self.trans_distn,
                self.init_state_distn,data=data,stateseq=stateseq,**kwargs))

    # this method is the main reason why this class exists
    def get_aBl(self,data):
        aBl = np.zeros((data.shape[0],self.state_dim))
        for idx in xrange(self.state_dim):
            aBl[:,idx] = self.obs_distns[idx].log_likelihood(data)
        return aBl

    # this method is the same as super except it optinally skips resampling the
    # state sequence (used in HSMMSubHMMs.resample_parallel())
    def resample_model(self,resample_stateseq=True):
        if resample_stateseq:
            super(SubHMM,self).resample_model()
        else:
            # resample obsparams
            for state, distn in enumerate(self.obs_distns):
                distn.resample([s.data[s.stateseq == state] for s in self.states_list])

            # resample transitions
            self.trans_distn.resample([s.stateseq for s in self.states_list])

            # resample pi_0
            self.init_state_distn.resample([s.stateseq[0] for s in self.states_list])

    def messages_forwards(self,aBl):
        # note: this method never uses self.T
        T = aBl.shape[0]
        alphal = np.zeros((T,self.state_dim))
        alphal[0] = np.log(self.init_state_distn.pi_0) + aBl[0]
        A = self.trans_distn.A # eigen sees this transposed

        trunc = self.trunc if self.trunc is not None else T

        # note the bad naming: the T in the codestr is replaced with trunc,
        # while the accurately-named T is passed in as a variable.
        scipy.weave.inline(messages_forwards_codestr % {'M':self.state_dim,'T':trunc},
                ['A','alphal','aBl','T'],
                headers=['<Eigen/Core>','<limits>'],
                include_dirs=[pyhsmm.EIGEN_INCLUDE_DIR],
                extra_compile_args=['-O3','-w'],
                verbose=1)

        assert not np.isnan(alphal).any()
        return alphal


### C++/Eigen stuff
import scipy.weave
import os
eigen_code_dir = os.path.join(os.path.dirname(__file__),'cpp_eigen_code/')
with open(os.path.join(eigen_code_dir,'hmm_messages_forwards.cpp'),'r') as infile:
    messages_forwards_codestr = infile.read()

