from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

import pyhsmm
from pyhsmm.util.text import progprint_xrange

N = 4
obs_hypparams = dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.05,nu_0=4)
T = 1000

### generate data

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(N)]
truemodel = pyhsmm.models.HMM(alpha=N,init_state_concentration=1.,obs_distns=obs_distns)


def normalize(mat):
    return np.nan_to_num(mat / mat.sum(1)[:,None])

trans_matrices = [normalize(mat) for mat in
        [
            np.ones((N,N)),
            np.eye(N) + np.diag(0.3*np.ones(N-1),1) + np.diag(0.3*np.ones(N-1),-1),
        ]
    ]

datas, labels = defaultdict(list), defaultdict(list)
for idx, mat in enumerate(trans_matrices):
    truemodel.trans_distn.trans_matrix = mat
    for itr in range(3):
        mydata, mylabels = truemodel.generate(T)
        datas['group%d' % idx].append(mydata)
        labels['group%d' % idx].append(mylabels)

### inference (using true obs_distns)

model = pyhsmm.models.HMMSeparateTrans(alpha=N,init_state_concentration=1.,
        obs_distns=obs_distns)

for group_id, datalist in datas.iteritems():
    for data in datalist:
        model.add_data(group_id=group_id,data=data)


for itr in progprint_xrange(100):
    model.resample_states()
    model.resample_trans_distn()

plt.matshow(model.trans_distns['group1'].trans_matrix)
plt.show()

