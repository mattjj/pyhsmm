import numpy as np
from numpy import newaxis as na
from numpy.random import random
import scipy.weave

from util.stats import sample_discrete
from util import general as util

class states(object): # {{{
    '''
    HSMM states distribution class. Connects the whole model.

    Parameters  include:

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

    def __init__(self,T,state_dim,obs_distns,dur_distns,transition_distn,initial_distn,stateseq=None,trunc=None,**kwargs):#{{{
        self.T = T
        self.state_dim = state_dim
        self.obs_distns = obs_distns
        self.dur_distns = dur_distns
        self.transition_distn = transition_distn
        self.initial_distn = initial_distn
        self.trunc = T if trunc is None else trunc

        # TODO put eigen funcs in child class
        # {{{
        self.sample_forwards_codestr = '''
    using namespace Eigen;

    Map<MatrixXd> ebetal(betal,%(M)d,%(T)d);
    Map<MatrixXd> ebetastarl(betastarl,%(M)d,%(T)d);
    Map<MatrixXd> eaBl(aBl,%(M)d,%(T)d);
    Map<MatrixXd> eA(A,%(M)d,%(M)d);
    Map<VectorXd> epi0(pi0,%(M)d);
    Map<MatrixXd> eapmf(apmf,%(T)d,%(M)d);

    //MatrixXd ebetal(%(M)d,%(T)d), ebetastarl(%(M)d,%(T)d), eaBl(%(M)d,%(T)d), eA(%(M)d,%(M)d), eapmf(%(T)d,%(M)d);
    //VectorXd epi0(%(M)d);

    // outputs

    Map<VectorXi> estateseq(stateseq,%(T)d);
    //VectorXi estateseq(%(T)d);
    estateseq.setZero();

    // locals
    // TODO should be stack variables, not dynamic ones
    int idx, state, dur;
    double durprob, p_d_marg, p_d, total;
    VectorXd nextstate_unsmoothed(%(M)d);
    VectorXd logdomain(%(M)d);
    VectorXd nextstate_distr(%(M)d);
    VectorXd cumsum(%(M)d);

    // code!
    // don't think i need to seed... should include sys/time.h for this
    // struct timeval time;
    // gettimeofday(&time,NULL);
    // srandom((time.tv_sec * 1000) + (time.tv_usec / 1000));

    idx = 0;
    nextstate_unsmoothed = epi0;

    while (idx < %(T)d) {
        logdomain = ebetastarl.col(idx).array() - ebetastarl.col(idx).maxCoeff();
        nextstate_distr = logdomain.array().exp() * nextstate_unsmoothed.array();
        if ((nextstate_distr.array() == 0.0).all()) {
            std::cout << "Warning: this is a cryptic error message" << std::endl;
            nextstate_distr = logdomain.array().exp();
        }
        // sample from nextstate_distr
        {
            total = nextstate_distr.sum() * (((double)random())/((double)RAND_MAX));
            for (state = 0; (total -= nextstate_distr(state)) > 0; state++) ;
        }

        durprob = ((double)random())/((double)RAND_MAX);
        dur = 0;
        while (durprob > 0.0) {
            if (dur > 2*%(T)d) {
                std::cout << "FAIL" << std::endl;
            }

            p_d_marg = (dur < %(T)d) ? eapmf(dur,state) : 1.0;
            if (0.0 == p_d_marg) {
                 dur += 1;
                 continue;
            }
            if (idx+dur < %(T)d) {
                 p_d = p_d_marg * (exp(eaBl.row(state).segment(idx,dur+1).sum() + ebetal(state,idx+dur) - ebetastarl(state,idx)));
            } else {
                break; // TODO fix this
            }
            durprob -= p_d;
            dur += 1;
        }

        estateseq.segment(idx,dur).setConstant(state);

        nextstate_unsmoothed = eA.col(state);

        idx += dur;
    }
''' % {'M':state_dim,'T':T}
# }}}

        # {{{
        self.messages_backwards_codestr = '''
        using namespace Eigen;
        using namespace std;
        // inputs
        int etrunc = mytrunc;
        Map<MatrixXd> eaBl(aBl,%(M)d,%(T)d);
        Map<MatrixXd> eA(A,%(M)d,%(M)d);
        Map<MatrixXd> eaDl(aDl,%(M)d,%(T)d);
        Map<MatrixXd> eaDsl(aDsl,%(M)d,%(T)d);

        // outputs
        Map<MatrixXd> ebetal(betal,%(M)d,%(T)d);
        Map<MatrixXd> ebetastarl(betastarl,%(M)d,%(T)d);

        // locals
        VectorXd maxes(%(M)d), result(%(M)d), sumsofar(%(M)d);
        double cmax;

        // computation!
        for (int t = %(T)d-1; t >= 0; t--) {
            sumsofar.setZero();
            ebetastarl.col(t).setConstant(-1.0*numeric_limits<double>::infinity());
            for (int tau = 0; tau < min(etrunc,%(T)d-t); tau++) {
                sumsofar += eaBl.col(t+tau);
                result = ebetal.col(t+tau) + sumsofar + eaDl.col(tau);
                maxes = ebetastarl.col(t).cwiseMax(result);
                ebetastarl.col(t) = ((ebetastarl.col(t) - maxes).array().exp() + (result - maxes).array().exp()).log() + maxes.array();
            }
            // censoring calc
            if (%(T)d - t < etrunc) {
                result = eaBl.block(0,t,%(M)d,%(T)d-t).rowwise().sum() + eaDsl.col(%(T)d-1-t);
                maxes = ebetastarl.col(t).cwiseMax(result);
                ebetastarl.col(t) = ((ebetastarl.col(t) - maxes).array().exp() + (result - maxes).array().exp()).log() + maxes.array();
            }
            // nan issue
            for (int i = 0; i < %(M)d; i++) {
                if (ebetastarl(i,t) != ebetastarl(i,t)) {
                    ebetastarl(i,t) = -1.0*numeric_limits<double>::infinity();
                }
            }
            // betal calc
            if (t > 0) {
                cmax = ebetastarl.col(t).maxCoeff();
                ebetal.col(t-1) = (eA * (ebetastarl.col(t).array() - cmax).array().exp().matrix()).array().log() + cmax;
                for (int i = 0; i < %(M)d; i++) {
                    if (ebetal(i,t-1) != ebetal(i,t-1)) {
                        ebetal(i,t-1) = -1.0*numeric_limits<double>::infinity();
                    }
                }
            }
        }

    ''' % {'M':state_dim,'T':T}
# }}}

        # this arg is for initialization heuristics which may pre-determine the state sequence
        if stateseq is not None:
            self.stateseq = stateseq
            # gather durations and stateseq_norep
            self.stateseq_norep, self.durations = util.rle(stateseq)
        else:
            self.generate_states()#}}}


    def get_aBl(self,data):#{{{
        aBl = np.zeros((data.shape[0],self.state_dim))
        for idx in xrange(self.state_dim):
            aBl[:,idx] = self.obs_distns[idx].log_likelihood(data)
        return aBl#}}}

    def resample(self,data):#{{{
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
        # TODO debug remove this
        self.betal, self.betastarl = betal, betastarl
        # sample forwards
        #self.sample_forwards_eigen(betal,betastarl)
        self.sample_forwards(betal,betastarl)
        # save these for testing convenience
        self.aDl = aDl
        self.aDsl = aDsl#}}}

    def generate(self):#{{{
        self.generate_states()
        return self.generate_obs()#}}}

    def generate_obs(self):#{{{
        obs = []
        for state in self.stateseq:
            obs.append(self.obs_distns[state].rvs(size=1))
        return np.vstack(obs).copy()#}}}

    def generate_states(self,censoring=True):#{{{
        assert censoring
        # with censoring, uses self.T
        # without censoring, overwrites self.T with any extra duration from the last state
        # returns data, sets internal stateseq as truth
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
        self.durations = np.array(durations,dtype=np.int32) # note sum(durations) can exceed len(stateseq) if censoring

        if censoring:
            self.stateseq = stateseq[:self.T]
        else:
            self.stateseq = stateseq

        assert len(self.stateseq_norep) == len(self.durations)
        assert (self.stateseq >= 0).all()#}}}

    def messages_backwards(self,Al,aDl,aDsl,trunc):#{{{
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

        return betal, betastarl#}}}

    def messages_backwards_truncshort(cls,Al,aDl,aDsl,trunc,tol=-np.inf):#{{{
        # a step towards only summing over support
        truncshort = np.where(aDl.max(1) >= tol)[0][0]
        print 'TRUNCSHORT: %d' % truncshort
        assert truncshort < trunc

        T = aDl.shape[0]
        state_dim = Al.shape[0]
        betal = np.zeros((T,state_dim),dtype=np.float64)
        betastarl = np.zeros((T,state_dim),dtype=np.float64)
        assert betal.shape == aDl.shape == aDsl.shape

        for t in xrange(T-1,T-1-truncshort,-1):
            # TODO is this working?
            if T-t < trunc:
                # censoring calc
                betastarl[t] = cls.likelihood_block(t,None) + aDsl[T-1-t]
            np.logaddexp.reduce(betastarl[t] + Al,axis=1,out=betal[t-1])

        for t in xrange(T-1-truncshort,-1,-1):
            np.logaddexp.reduce(betal[t+truncshort:t+trunc] + cls.cumulative_likelihoods(t,t+trunc)[truncshort:] + aDl[truncshort:min(trunc,T-t)],axis=0, out=betastarl[t])
            if T-t < trunc:
                # censoring calc
                np.logaddexp(betastarl[t], cls.likelihood_block(t,None) + aDsl[T-1-t], betastarl[t])
            np.logaddexp.reduce(betastarl[t] + Al,axis=1,out=betal[t-1])

        betal[-1] = 0. # overwritten in last loop for loop expression simplicity, set it back to 0 here

        assert not np.isnan(betal).any()
        assert not np.isinf(betal[0]).all()

        return betal, betastarl#}}}

    def messages_backwards_eigen(self,Al,aDl,aDsl,trunc):#{{{
        # THIS IS SAME SPEED OR SLOWER! that sucks, but it's necessary for a full C++
        # implementation. so i didn't totally waste my time...
        A = np.exp(Al).T.copy()
        mytrunc = trunc;
        aBl = self.aBl
        betal = np.zeros((self.T,self.state_dim))
        betastarl = np.zeros((self.T,self.state_dim))
        scipy.weave.inline(self.messages_backwards_codestr,['A','mytrunc','betal','betastarl','aDl','aBl','aDsl'],headers=['<Eigen/Core>','<limits>'],include_dirs=['/usr/local/include/eigen3'],extra_compile_args=['-O3','-march=native'])
        return betal, betastarl#}}}

    def cumulative_likelihoods(self,start,stop):#{{{
        return np.cumsum(self.aBl[start:stop],axis=0)

    def cumulative_likelihood_state(self,start,stop,state):
        return np.cumsum(self.aBl[start:stop,state])

    def likelihood_block(self,start,stop):
        # does not include the stop index
        return np.sum(self.aBl[start:stop],axis=0)

    def likelihood_block_state(self,start,stop,state):
        return np.sum(self.aBl[start:stop,state])#}}}

    def sample_forwards(self,betal,betastarl):#{{{
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
            logdomain = betastarl[idx,:] - np.amax(betastarl[idx])
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
                    mess_term = np.exp(self.likelihood_block_state(idx,idx+dur+1,state) + betal[idx+dur,state] - betastarl[idx,state])
                    p_d = mess_term * p_d_marg
                    #print 'dur: %d, durprob: %f, p_d_marg: %f, p_d: %f' % (dur+1,durprob,p_d_marg,p_d)
                    prob_so_far += p_d
                else:
                    break # TODO should add in censored sampling here
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
        self.stateseq_norep = np.array(stateseq_norep,dtype=np.int32)#}}}

    def sample_forwards_eigen(self,betal,betastarl):#{{{
        aBl = self.aBl
        #stateseq = self.stateseq
        stateseq = np.array(self.stateseq,dtype=np.int) # TODO weirdness here on ws7
        A = self.transition_distn.A
        pi0 = self.initial_distn.pi_0

        apmf = np.zeros((self.state_dim,self.T))
        arg = np.arange(1,self.T+1)
        for state_idx, dur_distn in enumerate(self.dur_distns):
            apmf[state_idx] = dur_distn.pmf(arg)

        scipy.weave.inline(self.sample_forwards_codestr,['betal','betastarl','aBl','stateseq','A','pi0','apmf'],headers=['<Eigen/Core>'],include_dirs=['/usr/local/include/eigen3'],extra_compile_args=['-O3'])#,'-march=native'])

        self.stateseq_norep, self.durations = util.rle(stateseq)#}}}

    # TODO methods below here may be deprecated...

    def loglike(self,data,trunc=None):
        self.obs = data # since functions within messages_backwards look at self.obs
        T = data.shape[0]
        if trunc is None:
            trunc = T
        possible_durations = np.arange(1,T + 1,dtype=np.float64)
        aDl = np.zeros((T,self.state_dim))
        aDsl = np.zeros((T,self.state_dim))
        for idx, dur_distn in enumerate(self.dur_distns):
            aDl[:,idx] = dur_distn.log_pmf(possible_durations)
            aDsl[:,idx] = dur_distn.log_sf(possible_durations)

        self.aBl = self.get_aBl(data)
        betal, betastarl = self.messages_backwards(np.log(self.transition_distn.A),aDl,aDsl,trunc)
        self.obs = None
        return np.logaddexp.reduce(np.log(self.initial_distn.pi_0) + betastarl[0])

    def resample_full(self,data):
        self.resample(data)
        self.transition_distn.resample(self.stateseq_norep)
        self.initial_distn.resample(self.stateseq[0])
        for state, distn in enumerate(self.dur_distns):
            distn.resample(self.durations[self.stateseq_norep == state])
        for state, distn in enumerate(self.obs_distns):
            distn.resample(data[self.stateseq == state])


    # not this one, though!

    def plot(self,colors_dict=None):
        from matplotlib import pyplot as plt
        X,Y = np.meshgrid(np.hstack((0,self.durations.cumsum())),(0,1))

        if colors_dict is not None:
            C = np.array([[colors_dict[state] for state in self.stateseq_norep]])
        else:
            C = self.stateseq_norep[na,:]

        plt.pcolor(X,Y,C) # NOTE: pcolor normalizes C
        plt.ylim((0,1))
        plt.xlim((0,self.T))
        plt.yticks([])

# }}}

class hmm_states(object): # {{{
    def __init__(self,state_dim,obs_distns,transition_distn,initial_distn,stateseq=None,trunc=None,**kwargs):
        self.state_dim = state_dim
        self.obs_distns = obs_distns
        self.transition_distn = transition_distn
        self.initial_distn = initial_distn
        self.stateseq = stateseq

        if trunc is not None:
            self.messages_forwards_codestr = \
    '''
    using namespace Eigen;

    // inputs
    Map<MatrixXd> eAT(A,%(M)d,%(M)d);
    Map<MatrixXd> eaBl(aBl,%(M)d,%(T)d);

    // outputs
    Map<MatrixXd> ealphal(alphal,%(M)d,%(T)d);

    // locals
    double cmax;

    // computation!
    for (int t=0; t<T-1; t++) {
        cmax = ealphal.col(t).maxCoeff();
        ealphal.col(t+1) = (eAT * (ealphal.col(t).array() - cmax).array().exp().matrix()).array().log() + cmax + eaBl.col(t+1).array();
        /* // nan issue (is there a better way to do this?)
        for (int i=0; i<%(M)d; i++) {
            if (ealphal(i,t+1) != ealphal(i,t+1)) {
                ealphal(i,t+1) = -1.0*numeric_limits<double>::infinity();
            }
        } */
    }
    ''' % {'M':self.state_dim,'T':trunc}
        else:
            self.messages_forwards = self.messages_forwards_numpy


    def generate_states(self,T):
        stateseq = np.zeros(T,dtype=np.int32)
        nextstate_distn = self.initial_distn.pi_0
        A = self.transition_distn.A

        for idx in xrange(T):
            stateseq[idx] = sample_discrete(nextstate_distn)
            nextstate_distn = A[stateseq[idx]]

        self.stateseq = stateseq
        return stateseq

    def generate_obs(self,stateseq):
        obs = []
        for state in stateseq:
            obs.append(self.obs_distns[state].rvs(size=1))
        return np.vstack(obs).copy()

    def generate(self,T):
        stateseq = self.generate_states(T)
        return stateseq, self.generate_obs(stateseq)

    def messages_forwards(self,aBl):
        T = aBl.shape[0]
        alphal = np.zeros((T,self.state_dim))
        alphal[0] = np.log(self.initial_distn.pi_0) + aBl[0]
        A = self.transition_distn.A # eigen sees this transposed

        scipy.weave.inline(self.messages_forwards_codestr,['A','alphal','aBl','T'],headers=['<Eigen/Core>','<limits>'],include_dirs=['/usr/local/include/eigen3'],extra_compile_args=['-O3'])

        assert not np.isnan(alphal).any()
        return alphal

    def messages_forwards_numpy(self,aBl):
        T = aBl.shape[0]
        alphal = np.zeros((T,self.state_dim))
        Al = np.log(self.transition_distn.A)

        alphal[0] = np.log(self.initial_distn.pi_0) + aBl[0]

        for t in xrange(T-1):
            alphal[t+1] = np.logaddexp.reduce(alphal[t] + Al.T,axis=1) + aBl[t+1]

        return alphal

    def messages_backwards(self,aBl):
        # TODO write eigen version
        betal = np.zeros(aBl.shape)
        Al = np.log(self.transition_distn.A)
        T = aBl.shape[0]

        for t in xrange(T-2,-1,-1):
            np.logaddexp.reduce(Al + betal[t+1] + aBl[t+1],axis=1,out=betal[t])

        return betal

    def sample_forwards(self,aBl,betal):
        # TODO write eigen version
        T = aBl.shape[0]
        stateseq = np.zeros(T,dtype=np.int32)
        nextstate_unsmoothed = self.initial_distn.pi_0
        A = self.transition_distn.A

        for idx in xrange(T):
            logdomain = betal[idx] + aBl[idx]
            stateseq[idx] = sample_discrete(nextstate_unsmoothed * np.exp(logdomain - np.amax(logdomain)))
            nextstate_unsmoothed = A[stateseq[idx]]

        return stateseq

    def resample(self,data):
        aBl = self.get_aBl(data)
        betal = self.messages_backwards(aBl)
        self.stateseq = self.sample_forwards(aBl,betal)
        return self.stateseq

    def get_aBl(self,data):
        aBl = np.zeros((data.shape[0],self.state_dim))
        for idx, obs_distn in enumerate(self.obs_distns):
            aBl[:,idx] = obs_distn.log_likelihood(data)
        return aBl

    def loglike(self,data):
        aBl = self.get_aBl(data)
        betal = self.messages_backwards(aBl)
        return np.logaddexp.reduce(np.log(self.initial_distn.pi_0) + betal[0] + aBl[0])
# }}}

