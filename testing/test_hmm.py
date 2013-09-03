from __future__ import division
import numpy as np

from pyhsmm import models as m, distributions as d

# likelihood hand-test
def likelihood_hand_tests():
    return _hand_test_helper, [d.Categorical(weights=row) for row in np.eye(2)], \
            np.array([[0.9,0.1],[0.1,0.9]]), np.array([1.,0.]), np.zeros(10,dtype=int), \
            9*np.log(0.9)

def _hand_test_helper(obs_distns,trans_matrix,init_distn,data,target_val):
    hmm = m.HMM(
            alpha=6,gamma=6,init_state_concentration=1, # placeholders
            obs_distns=obs_distns)
    hmm.trans_distn.A = trans_matrix
    hmm.init_state_distn.weights = init_distn
    hmm.add_data(data)
    assert np.isclose(hmm.log_likelihood(), target_val)


# likelihood direct sum test (would need complete data likelihood)

