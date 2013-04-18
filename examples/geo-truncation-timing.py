from __future__ import division
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import time

import pyhsmm

###################
#  generate data  #
###################

N = 4
T = 5000
obs_dim = 2

obs_distns = \
        [pyhsmm.distributions.Gaussian(
            mu_0=np.zeros(obs_dim), sigma_0=np.eye(obs_dim),kappa_0=0.2, nu_0=obs_dim+2)
                for state in range(N)]

dur_distns = \
        [pyhsmm.distributions.NegativeBinomialDuration(3*100,1./100,50*50,50*1)
                for state in range(N)]

truemodel = pyhsmm.models.HSMM(
        alpha=6.,gamma=6.,
        init_state_concentration=10.,
        obs_distns=obs_distns,
        dur_distns=dur_distns)

data, labels = truemodel.generate(T)

#########################
#  test message pasing  #
#########################

untrunc = pyhsmm.models.HSMM(
        init_state_concentration=10.,
        obs_distns=obs_distns,
        dur_distns=dur_distns,
        trans_distn=truemodel.trans_distn)
untrunc.add_data(data)

hardtrunc = pyhsmm.models.HSMM(
        init_state_concentration=10.,
        obs_distns=obs_distns,
        dur_distns=dur_distns,
        trans_distn=truemodel.trans_distn,
        trunc=250)
hardtrunc.add_data(data)

geotrunc = pyhsmm.models.HSMMGeoApproximation(
        init_state_concentration=10.,
        obs_distns=obs_distns,
        dur_distns=dur_distns,
        trans_distn=truemodel.trans_distn,
        trunc=250)
geotrunc.add_data(data)

tic = time.time()
betal_hardtrunc, _ = hardtrunc.states_list[0].messages_backwards()
elapsed = time.time() - tic
print '%f seconds for hard truncation' % elapsed

tic = time.time()
betal_geotrunc, _ = geotrunc.states_list[0].messages_backwards()
elapsed = time.time() - tic
print '%f seconds for geo truncation' % elapsed

tic = time.time()
betal_untrunc, _ = untrunc.states_list[0].messages_backwards()
elapsed = time.time() - tic
print '%f seconds for no truncation' % elapsed

##########
#  plot  #
##########

plt.figure()
plt.plot(betal_untrunc[:,0],label='untruncated')
plt.plot(betal_hardtrunc[:,0],'x--',label='hard truncation')
plt.plot(betal_geotrunc[:,0],'+--',label='geo truncation')
plt.legend(loc='best')

plt.show()

