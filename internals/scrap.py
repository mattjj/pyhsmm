from __future__ import division

class LTRHDPHMMTransitions(HDPHMMTransitions):
    def _count_transitions(self,states_list):
        trans_counts = super(LTRHDPHMMTransitions,self)._count_transitions(states_list)
        assert (0==np.tril(trans_counts,-1)).all()
        totalfrom = trans_counts.sum(1)
        totalweight = np.triu(self.fullA).sum(1)
        for i in range(trans_counts.shape[0]):
            tot = np.random.geometric(totalweight[i],size=totalfrom[i]).sum()
            trans_counts[i,:i] = np.random.multinomial(tot,self.fullA[i,:i]/self.fullA[i,:i].sum())
        return trans_counts

    def _resample_A(self,trans_counts):
        super(LTRHDPHMMTransitions,self)._resample_A(trans_counts)
        self.fullA = self.A
        self.A = np.triu(self.A) / np.triu(self.A).sum(1)[:,na]

class StickyLTRHDPHMMTransitions(LTRHDPHMMTransitions,StickyHDPHMMTransitions):
    def _resample_A(self,data):
        aug_data = data + np.diag(self.kappa * np.ones(data.shape[0]))
        super(StickyLTRHDPHMMTransitions,self)._resample_A(aug_data)

class UniformTransitionsFixedSelfTrans(HDPHMMTransitions):
    '''
    All states have the same self-transition probability, i.e. np.diag(A) is a
    constant LMBDA.  The transition-out probabilities are all the same, set according
    to a fixed state weighting PI. Note that transitions out are sampled
    proportional to PI but without the possibility of self-transition.
    '''
    def __init__(self,lmbda,pi):
        assert 0 < lmbda < 1

        self.lmbda = lmbda
        self.pi = pi
        self.state_dim = pi.K
        self._set_A()

    ### Gibbs sampling

    def resample(self,states_list=[],niter=100,trans_counts=None):
        # trans_counts arg is for convenient testing
        if trans_counts is None:
            trans_counts = self._count_transitions(states_list)

        for itr in range(niter):
            aug_trans_counts = self._augment_transitions(trans_counts)
            self.pi.resample(aug_trans_counts.sum(0))

        self._set_A()

    def _augment_transitions(self,trans):
        trans = trans.copy()
        if trans.sum() > 0:
            trans.flat[::trans.shape[0]+1] = 0
            for i, tot in enumerate(trans.sum(1)):
                trans[i,i] = np.random.geometric(1.-self.pi.weights[i],size=tot).sum() - tot
        return trans

    def _set_A(self):
        self.A = np.tile(self.pi.weights,(self.state_dim,1))
        self.A.flat[::self.state_dim+1] = 0
        self.A /= self.A.sum(1)[:,na]
        self.A *= (1.-self.lmbda)
        self.A.flat[::self.state_dim+1] = self.lmbda

    @classmethod
    def test_sampling(cls,N=50,K=10,alpha_0=4.,lmbda=0.95):
        from ..basic.distributions import Categorical
        from matplotlib import pyplot as plt

        true_pi = np.random.dirichlet(np.repeat(alpha_0/K,K))
        counts = np.array([np.random.multinomial(N,true_pi) for i in range(K)]) # diagional ignored

        pi = Categorical(alpha_0=alpha_0,K=K)
        trans = cls(lmbda,pi)

        plt.figure()
        plt.plot(true_pi,'r')
        cmap = plt.cm.get_cmap('Blues')
        for i in range(5):
            trans.resample(trans_counts=counts)
            plt.plot(trans.pi.weights,'--',label=str(i),color=cmap((i+2)/(5+2)))
        plt.legend()
        plt.show()

    ### max likelihood

    def max_likelihood(self,stateseqs,expectations_list=None):
        if expectations_list is not None:
            trans_counts = self._count_weighted_transitions(expectations_list,self.A)
        else:
            trans_counts = self._count_transitions(stateseqs)

        trans_counts = self._E_augment_transitions(trans_counts)
        self.pi.max_likelihood(trans_counts.sum(0))
        self._set_A()

    def _E_augment_transitions(self,trans_softcounts):
        trans_softcounts.flat[::self.state_dim+1] = trans_softcounts.sum(1) * 1./(1.-self.lmbda)
        return trans_softcounts



class UniformTransitions(UniformTransitionsFixedSelfTrans):
    '''
    Like UniformTransitionsFixedSelfTrans except also samples over the
    self-transition probability LMBDA.
    '''
    def __init__(self,lmbda_a_0,lmbda_b_0,pi,lmbda=None):
        self.a_0 = lmbda_a_0
        self.b_0 = lmbda_b_0
        self.pi = pi
        self.state_dim = pi.K

        if lmbda is not None:
            self.lmbda = lmbda
            self._set_A()
        else:
            self.resample()

    def resample(self,states_list=[],niter=100,trans_counts=None):
        if trans_counts is None:
            trans_counts = self._count_transitions(states_list)

        for itr in range(niter):
            # resample lmbda
            self_trans = trans_counts.diagonal().sum()
            total_out = trans_counts.sum() - self_trans
            self.lmbda = np.random.beta(self.a_0 + self_trans, self.b_0 + total_out)

            # resample everything else as usual
            super(UniformTransitions,self).resample(states_list,trans_counts=trans_counts,niter=1)

    def max_likelihood(self,stateseqs,expectations_list=None):
        raise NotImplementedError, "max_likelihood doesn't make sense on this class"

    @classmethod
    def test_sampling(cls,N=50,K=10,alpha_0=4.,lmbda_a_0=20.,lmbda_b_0=1.,true_lmbda=0.95):
        from ..basic.distributions import Categorical
        from matplotlib import pyplot as plt

        true_pi = np.random.dirichlet(np.repeat(alpha_0/K,K))
        counts = np.array([np.random.multinomial(N,true_pi) for i in range(K)])
        counts.flat[::K+1] = 0
        for i,tot in enumerate(counts.sum(1)):
            counts[i,i] = np.random.geometric(1-true_lmbda,size=tot).sum()

        pi = Categorical(alpha_0=alpha_0,K=K)
        trans = cls(lmbda_a_0=lmbda_a_0,lmbda_b_0=lmbda_b_0,pi=pi)

        plt.figure()
        plt.plot(true_pi,'r')
        cmap = plt.cm.get_cmap('Blues')
        print 'True lmbda: %0.5f' % true_lmbda
        for i in range(5):
            trans.resample(trans_counts=counts)
            print 'Sampled lmbda: %0.5f' % trans.lmbda
            plt.plot(trans.pi.weights,'--',label=str(i),color=cmap((i+2)/(5+2)))
        plt.legend()
        plt.show()

