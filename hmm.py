from hsmm_internals import states, initial_state, transitions

class hmm(object):
    '''
    The HMM class is a convenient wrapper that provides useful constructors and
    packages all the components.
    '''

    def __init__(self,obs_distns,**kwargs):
        state_dim = len(obs_distns)

        self.obs_distns = obs_distns

        self.trans_distn = transitions.hmm_transitions(state_dim=state_dim,**kwargs)\
                if 'transitions' not in kwargs else kwargs['transitions']

        self.init_state_distn = initial_state.initial_state(state_dim=state_dim,**kwargs)\
                if 'initial_state_distn' not in kwargs else kwargs['initial_state_distn']

        self.states = states.hmm_states(state_dim,obs_distns,self.trans_distn,
                self.init_state_distn,**kwargs)

    def resample(self,obs,niter=1):
        for itr in range(niter):
            # resample obsparams
            for state, distn in enumerate(self.obs_distns):
                distn.resample(obs[self.states.stateseq == state])

            # resample transitions
            self.trans_distn.resample(self.states.stateseq)

            # resample pi_0
            self.init_state_distn.resample(self.states.stateseq[0])

            # resample states
            self.states.resample(obs)

    def generate(self,T):
        return self.states.generate(T)


