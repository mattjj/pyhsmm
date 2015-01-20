from __future__ import division
import os
import numpy as np
from matplotlib import pyplot as plt
plt.ion()

import pyhsmm
from pyhsmm.util.text import progprint_xrange

###############
#  load data  #
###############

data = np.loadtxt(os.path.join(os.path.dirname(__file__),'example-data.txt'))[:2500]
data += 0.5*np.random.normal(size=data.shape) # some extra noise

##################
#  set up model  #
##################

# Set the weak limit truncation level
Nmax = 25

# and some hyperparameters
obs_dim = data.shape[1]
obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.25,
                'nu_0':obs_dim+2}

# instantiate a Sticky-HDP-HMM
obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(Nmax)]
model = pyhsmm.models.WeakLimitStickyHDPHMM(
        kappa=50.,alpha=6.,gamma=6.,init_state_concentration=1.,
        obs_distns=obs_distns)
model.add_data(data)

##############
#  animate!  #
##############

from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoClip

fig = model.make_figure()
model.plot(fig=fig,draw=False,plot_slice=slice(0,100))

def make_frame_mpl(t):
    model.resample_model()
    model.plot(fig=fig,update=True,draw=False,plot_slice=slice(0,100))
    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame_mpl, duration=5)
animation.write_videofile('gibbs.mp4',fps=50)

