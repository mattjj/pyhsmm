from __future__ import division

# these things are bits of old code or need updating

####################################################
#  possiblechangepoints stuff (power application)  #
####################################################

# NOTE: these only work with iid-like emissions with aBl (that includes
# autoregressive but not fancier things)

class HSMMPossibleChangepoints(HSMM, ModelGibbsSampling):
    _states_class = states.HSMMStatesPossibleChangepoints

    def add_data(self,data,changepoints,**kwargs):
        self.states_list.append(
                self._states_class(model=self,changepoints=changepoints,data=np.asarray(data),**kwargs))

    def add_data_parallel(self,data_id,**kwargs):
        raise NotImplementedError # I broke this!
        from pyhsmm import parallel
        self.add_data(data=parallel.alldata[data_id],changepoints=parallel.allchangepoints[data_id],**kwargs)
        self.states_list[-1].data_id = data_id

    def _build_states_parallel(self,states_to_resample):
        from pyhsmm import parallel
        raw_stateseq_tuples = parallel.hsmm_build_states_changepoints.map([s.data_id for s in states_to_resample])
        for data_id, stateseq, stateseq_norep, durations in raw_stateseq_tuples:
            self.add_data(
                    data=parallel.alldata[data_id],
                    changepoints=parallel.allchangepoints[data_id],
                    stateseq=stateseq,
                    stateseq_norep=stateseq_norep,
                    durations=durations)
            self.states_list[-1].data_id = data_id

    def generate(self,T,changepoints,keep=True):
        raise NotImplementedError

    def log_likelihood(self,data,trunc=None):
        raise NotImplementedError

class HSMMStatesPossibleChangepoints(HSMMStatesPython): # TODO TODO update this class
    def __init__(self,model,changepoints,*args,**kwargs):
        warnings.warn("%s hasn't been used in a while; there may be some bumps"
                % self.__class__.__name__)

        self.changepoints = changepoints
        self.startpoints = np.array([start for start,stop in changepoints],dtype=np.int32)
        self.blocklens = np.array([stop-start for start,stop in changepoints],dtype=np.int32)
        self.Tblock = len(changepoints) # number of blocks
        super(HSMMStatesPossibleChangepoints,self).__init__(model,*args,**kwargs)

    ### generation

    def generate_states(self):
        # TODO TODO this method can probably call sample_forwards with dummy uniform
        # aBl/betal/betastarl, but that's just too complicated!
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
            possible_durations = self.blocklens[tblock:].cumsum()

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
        self.stateseq = np.zeros(self.T,dtype=np.int32)
        for state, (start,stop) in zip(blockstateseq,self.changepoints):
            self.stateseq[start:stop] = state
        self.stateseq_norep, self.durations_censored = util.rle(self.stateseq)

        return self.stateseq

    def generate(self): # TODO
        raise NotImplementedError

    ### caching

    def clear_caches(self):
        self._aBBl = None
        super(HSMMStatesPossibleChangepoints,self).clear_caches()

    @property
    def aBBl(self):
        if self._aBBl is None:
            aBl = self.aBl
            aBBl = self._aBBl = np.empty((self.Tblock,self.state_dim))
            for idx, (start,stop) in enumerate(self.changepoints):
                aBBl[idx] = aBl[start:stop].sum(0)
        return self._aBBl

    ### message passing

    def messages_backwards(self):
        aDl, Al = self.aDl, np.log(self.trans_matrix)
        Tblock = self.Tblock
        state_dim = Al.shape[0]
        trunc = self.trunc if self.trunc is not None else self.T

        betal = np.zeros((Tblock,state_dim),dtype=np.float64)
        betastarl = np.zeros_like(betal)

        for tblock in range(Tblock-1,-1,-1):
            possible_durations = self.blocklens[tblock:].cumsum() # could precompute these
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
            possible_durations = self.blocklens[tblock:].cumsum()
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
        self.stateseq = np.zeros(self.T,dtype=np.int32)
        for state, (start,stop) in zip(blockstateseq,self.changepoints):
            self.stateseq[start:stop] = state
        self.stateseq_norep, self.durations_censored = util.rle(self.stateseq)

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

