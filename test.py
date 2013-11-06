from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

from util.general import top_eigenvector

import internals.subhmm_messages_interface as subhmms

def make_example_model(T=10000):
    import pyhsmm
    from pyhsmm.util.stats import cov
    from pyhsmm.util.text import progprint_xrange

    Nsuper = 3
    Nsub = 5
    obs_dim = 2

    obs_hypparams = dict(
            mu_0=np.zeros(obs_dim),
            sigma_0=np.eye(obs_dim),
            kappa_0=0.01,
            nu_0=obs_dim+10,
            )

    dur_hypparams = dict(
            r_discrete_distn=np.r_[0,0,0,0,0,1.,1.,1.],
            alpha_0=20,
            beta_0=2,
            )

    true_obs_distnss = [[pyhsmm.distributions.Gaussian(**obs_hypparams) for substate in xrange(Nsub)]
            for superstate in xrange(Nsuper)]

    true_dur_distns = [pyhsmm.distributions.NegativeBinomialIntegerRVariantDuration(
        **dur_hypparams) for superstate in range(Nsuper)]

    truemodel = pyhsmm.models.HSMMIntNegBinVariantSubHMMs(
            init_state_concentration=6,
            alpha=10.,gamma=10.,
            sub_alpha=10.,sub_gamma=10.,
            obs_distnss=true_obs_distnss,
            dur_distns=true_dur_distns)

    data, _ = truemodel.generate(T)

    return truemodel

def test_steadystate():
    model = make_example_model()
    s = model.states_list[0]
    A = s.trans_matrix

    bigN = A.shape[0]

    v = np.random.random(bigN).astype('float32')

    result1 = v.dot(A)
    result2 = subhmms.test_vector_matrix_mult(v,s.hsmm_trans_matrix,s.rs,s.ps,s.subhmm_trans_matrices,s.subhmm_pi_0s)
    print np.allclose(result1,result2)

    v1 = top_eigenvector(A,niter=1000)

    v2 = np.repeat(1./bigN,bigN).astype('float32')
    subhmms.steady_state(v2,s.hsmm_trans_matrix,s.rs,s.ps,
            s.subhmm_trans_matrices,s.subhmm_pi_0s,1000)
    v2 /= v2.sum()

    print np.allclose(v1,v2)

def test_generate():
    plt.figure()
    model = make_example_model(T=50000)

    # test sparse method
    for itr in xrange(5):
        model.states_list[0].generate_states()
        s1 = model.states_list[0].stateseq
        plt.plot(np.bincount(s1),'b-')

    # test hmm method
    for itr in xrange(5):
        model.states_list[0].hmm_generate_states()
        s2 = model.states_list[0].stateseq
        plt.plot(np.bincount(s2),'gx--')

if __name__ == '__main__':
    # test_steadystate()

    test_generate()

    plt.show()

