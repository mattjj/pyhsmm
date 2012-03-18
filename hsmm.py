from __future__ import division
import numpy as np
from numpy import newaxis as na
from matplotlib import pyplot as plt
from matplotlib import cm
import itertools

import states, initial_state, transitions
import util

class hsmm(object):
    '''
    The HSMM class is a convenient wrapper that provides useful constructors and
    packages all the components.
    '''

    # TODO remove T
    # TODO make trunc a parameter of resample

    def __init__(self,T,obs_distns,dur_distns,trunc=None,**kwargs):
        state_dim = len(obs_distns)
        self.state_dim = state_dim
        self.T = T
        self.trunc = trunc

        self.obs_distns = obs_distns
        self.dur_distns = dur_distns

        self.trans_distn = transitions.transitions(state_dim=state_dim,**kwargs) if 'transitions' not in kwargs else kwargs['transitions']
        self.init_state_distn = initial_state.initial_state(state_dim=state_dim,**kwargs) if 'initial_state_distn' not in kwargs else kwargs['initial_state_distn']

        self.states = states.states(T,state_dim,obs_distns,dur_distns,self.trans_distn,self.init_state_distn,trunc=trunc,**kwargs)


    def resample(self,obs):
        # resample obsparams
        for state, distn in enumerate(self.obs_distns):
            distn.resample(obs[self.states.stateseq == state])

        # resample durparams
        for state, distn in enumerate(self.dur_distns):
            distn.resample(self.states.durations[self.states.stateseq_norep == state])

        # resample transitions
        self.trans_distn.resample(self.states.stateseq_norep)

        # resample pi_0
        self.init_state_distn.resample(self.states.stateseq[0])

        # resample states
        self.states.resample(obs)


    def generate(self):
        return self.states.generate(), self.states.stateseq

    def plot(self):
        means = np.array([obs_distn.mu for obs_distn in self.obs_distns])
        meanseq = means[self.states.stateseq]
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(meanseq,'k.')
        states, durs = util.rle(self.states.stateseq)
        X,Y = np.meshgrid(np.hstack((0,durs.cumsum())),[0,meanseq.max()*1.25]); plt.pcolor(X,Y,states[na,:],edgecolors='k',alpha=0.25); plt.ylim((0,meanseq.max()*1.25)); plt.xlim((0,self.T))
        plt.title('means and stateseq')

        plt.subplot(2,1,2)
        t = np.arange(1,250)
        print ''
        maxstate = states.max()
        cmap = cm.get_cmap()
        for state,d in enumerate(self.dur_distns):
            if np.sum(self.states.stateseq == state) > 1:
                color_index = (float(state)/maxstate)
                plt.plot(t,d.pmf(t),label='state %d, %s' % (state,d),color=cmap(color_index))
                if len(durs[states == state])>1:
                    plt.hist(durs[states == state],normed=True,color=cmap(color_index))
                print 'state %d duration distn: %s' % (state,d)
                print 'state %d observation distn: %s' % (state,self.obs_distns[state])
                print ''
        plt.legend()

    def plot_with_data(self,data):
        assert data.shape == (self.T,1)
        self.plot()
        plt.subplot(2,1,1)
        plt.ylim((0,data.max()))
        plt.plot(data,'y',linewidth=2)

# TODO this stuff is pretty ugly at the moment

class disaggregation(hsmm):
    def __init__(self,T,obs_distns,dur_distns,trunc=None,other_statesobjs=None,**kwargs):
        state_dim = len(obs_distns)
        hsmm.__init__(self,T,obs_distns,dur_distns,trunc=trunc,**kwargs)
        self.states = states.disaggregation_states(T,state_dim,obs_distns,dur_distns,self.trans_distn,self.init_state_distn,other_statesobjs=other_statesobjs,trunc=trunc,**kwargs)

        self.means = np.zeros(state_dim)
        self.vars = np.zeros(state_dim)

        for idx, obs_distn in enumerate(self.obs_distns):
            # ellipses make these slices (zero-dim arrays, references!)
            obs_distn.mubin = self.means[...,idx]
            obs_distn.sigmasqbin = self.vars[...,idx]

            obs_distn.mubin[...] = obs_distn.mu
            obs_distn.sigmasqbin[...] = obs_distn.sigmasq

        self.states.means = self.means
        self.states.vars = self.vars

class block_disaggregation(disaggregation):
    def __init__(self,blocks,T,obs_distns,dur_distns,trunc=None,other_statesobjs=None,**kwargs):
        state_dim = len(obs_distns)
        disaggregation.__init__(self,T,obs_distns,dur_distns,trunc=trunc,other_statesobjs=other_statesobjs,**kwargs)
        self.states = states.disaggregation_block_states(blocks,T,state_dim,obs_distns,dur_distns,self.trans_distn,self.init_state_distn,other_statesobjs=other_statesobjs,trunc=trunc,**kwargs)

        self.means = np.zeros(state_dim)
        self.vars = np.zeros(state_dim)

        for idx, obs_distn in enumerate(self.obs_distns):
            # ellipses make these slices (zero-dim arrays, references!)
            obs_distn.mubin = self.means[...,idx]
            obs_distn.sigmasqbin = self.vars[...,idx]

            obs_distn.mubin[...] = obs_distn.mu
            obs_distn.sigmasqbin[...] = obs_distn.sigmasq

        self.states.means = self.means
        self.states.vars = self.vars

def hsmm_to_disaggregation(chain):
    blah = disaggregation(chain.T,chain.obs_distns,chain.dur_distns,trunc=chain.trunc)
    blah.trans_distn = chain.trans_distn
    blah.init_state_distn = chain.init_state_distn
    blah.states = states_to_dstates(chain.states)
    return blah

def states_to_dstates(statesobj):
    return states.disaggregation_states(statesobj.T,statesobj.state_dim,statesobj.obs_distns,statesobj.dur_distns,statesobj.transition_distn,statesobj.initial_distn,trunc=statesobj.trunc)

class multichain(hsmm):
    def __init__(self,Ts,Nchains,obs_distns,dur_distns,trunc=None,**kwargs):
        self.state_dim = state_dim = len(obs_distns)
        self.trunc = trunc
        self.Nchains = Nchains

        self.obs_distns = obs_distns
        self.dur_distns = dur_distns

        self.trans_distn = transitions.transitions(state_dim=state_dim,**kwargs) if 'transitions' not in kwargs else kwargs['transitions']
        self.init_state_distn = initial_state.initial_state(state_dim=state_dim,**kwargs) if 'initial_state_distn' not in kwargs else kwargs['initial_state_distn']

        self.states_list = [states.states(T,state_dim,obs_distns,dur_distns,self.trans_distn,self.init_state_distn,trunc=trunc,**kwargs) for T in Ts]

    def resample(self,obss):
        # TODO only works for scalar gaussians for now, fix me later!
        for obs in obss:
            assert obs.ndim == 2
            assert obs.shape[1] == 1

        # resample obsparams
        for state, distn in enumerate(self.obs_distns):
            distn.resample(np.vstack([obs[states.stateseq == state] for states,obs in zip(self.states_list,obss)]))

        # resample durparams
        for state, distn in enumerate(self.dur_distns):
            distn.resample(np.concatenate([states.durations[states.stateseq_norep == state] for states in self.states_list]))

        # resample transitions
        self.trans_distn.resample([states.stateseq_norep for states in self.states_list])

        # resample pi_0
        self.init_state_distn.resample([states.stateseq[0] for states in self.states_list])

        # resample states
        for states,obs in zip(self.states_list,obss):
            states.resample(obs)

    def generate(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    @classmethod
    # self should really be cls here
    def copy_from(self,other_hsmm):
        Ts = map(lambda x: x.T,other_hsmm.states_list)
        blah = self(Ts,other_hsmm.Nchains,util.deepcopy(other_hsmm.obs_distns),util.deepcopy(other_hsmm.dur_distns),trunc=other_hsmm.trunc,transitions=util.deepcopy(other_hsmm.trans_distn),initial_state_distn=util.deepcopy(other_hsmm.init_state_distn))
        blah.states_list = util.deepcopy(other_hsmm.states_list)
        return blah

    def plot_with_data(self,datas):
        Nstates = len(self.obs_distns)
        states_used = set(itertools.chain(*map(lambda x: x.stateseq,self.states_list)))
        Nstates_used = len(states_used)
        canon_states_map = np.zeros(Nstates,dtype=np.int32)
        for idx,state in enumerate(states_used):
            canon_states_map[state] = idx
        color_arr = np.linspace(0,1,Nstates_used)
        plt.figure()
        means = np.array([obs_distn.mu for obs_distn in self.obs_distns])
        for idx,(data,states) in enumerate(zip(datas,self.states_list)):
            meanseq = means[states.stateseq]
            plt.subplot(self.Nchains,1,idx+1)
            plt.plot(data,'kx')
            plt.plot(meanseq,'y+')
            states_norep, durs = util.rle(states.stateseq)
            # TODO this is still weird wrt colors...
            plt.pcolor(np.hstack((0,durs.cumsum())),np.array([0,data.max()]),color_arr[canon_states_map[states_norep]][na,:],edgecolor='none',alpha=0.75,vmin=0,vmax=1); plt.ylim((0,data.max())); plt.xlim((0,states.T))

#        plt.figure()
#        t = np.arange(1,250)
#        print ''
#        maxstate = states_norep.max()
#        cmap = cm.get_cmap()
#        for state,d in enumerate(self.dur_distns):
#            if np.sum(self.states.stateseq == state) > 1:
#                color_index = (float(state)/maxstate)
#                plt.plot(t,d.pmf(t),label='state %d, %s' % (state,d),color=cmap(color_index))
#                if len(durs[states_norep == state])>1:
#                    plt.hist(durs[states_norep == state],normed=True,color=cmap(color_index))
#                print 'state %d duration distn: %s' % (state,d)
#                print 'state %d observation distn: %s' % (state,self.obs_distns[state])
#                print ''
#        plt.legend()
