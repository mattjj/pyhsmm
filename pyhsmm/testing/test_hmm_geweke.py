from __future__ import division
import numpy as np
from functools import wraps
from nose.plugins.attrib import attr
import os
import matplotlib.pyplot as plt

from .. import models, distributions
from ..util import testing

##########
#  util  #
##########

def runmultiple(n):
    def dec(fn):
        @wraps(fn)
        def wrapper():
            fig = plt.figure()
            for i in range(n):
                yield fn, fig
            plt.close('all')
        return wrapper
    return dec


def mkdir(path):
    # from
    # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

figure_dir_path = os.path.join(os.path.dirname(__file__),'figures')
mkdir(figure_dir_path)

###########
#  tests  #
###########

@attr('slow')
@runmultiple(2)
def discrete_geweke_test(fig):
    Nstates = 2
    Nemissions = 2
    alpha = 3.
    init_state_concentration=3.
    T = 10
    num_iter = 10000
    num_checks = 10

    obs_distns = [distributions.Categorical(K=Nemissions,alpha_0=1.)
            for _ in range(Nstates)]

    hmm = models.HMM(
            alpha=alpha,init_state_concentration=init_state_concentration,
            obs_distns=obs_distns)

    # generate state sequences and parameters from the prior
    prior_stateseqs = []
    prior_weights = []
    for itr in xrange(num_iter):
        hmm.resample_model() # sample parameters from the prior
        _, stateseq = hmm.generate(T,keep=False)
        prior_stateseqs.append(stateseq)
        prior_weights.append(hmm.obs_distns[0].weights)
    prior_stateseqs = np.array(prior_stateseqs)
    prior_weights = np.array(prior_weights)

    # generate state sequences and parameters using Gibbs
    hmm.generate(T,keep=True)
    s = hmm.states_list[0]

    gibbs_stateseqs = []
    gibbs_weights = []
    for itr in xrange(num_iter):
        s.generate_obs() # resamples data given state sequence, obs params
        hmm.resample_model() # resamples everything else as usual
        gibbs_stateseqs.append(s.stateseq)
        gibbs_weights.append(hmm.obs_distns[0].weights)
    gibbs_stateseqs = np.array(gibbs_stateseqs)
    gibbs_weights = np.array(gibbs_weights)

    # test that they look similar by checking probability of co-assignment
    time_indices = np.arange(T)
    for itr in xrange(num_checks):
        i,j = np.random.choice(time_indices,replace=False,size=2)
        prior_prob_of_coassignment = (prior_stateseqs[:,i] == prior_stateseqs[:,j]).std()
        gibbs_prob_of_coassignment = (gibbs_stateseqs[:,i] == gibbs_stateseqs[:,j]).std()

        assert np.isclose(
                prior_prob_of_coassignment,gibbs_prob_of_coassignment,
                rtol=0.025,atol=0.025,
                )

    # test that they look similar by checking parameters
    testing.populations_eq_quantile_plot(prior_weights,gibbs_weights,fig=fig)
    figpath = os.path.join(figure_dir_path,'discrete_geweke_test_weights.pdf')
    plt.savefig(figpath)

@attr('slow')
@runmultiple(2)
def discrete_geweke_multiple_seqs_test(fig):
    Nstates = 2
    Nemissions = 2
    alpha = 3.
    init_state_concentration=3.
    T = 10
    num_seqs = 3
    num_iter = 10000
    num_checks = 10

    obs_distns = [distributions.Categorical(K=Nemissions,alpha_0=1.)
            for _ in range(Nstates)]

    hmm = models.HMM(
            alpha=alpha,init_state_concentration=init_state_concentration,
            obs_distns=obs_distns)

    # generate state sequences and parameters from the prior
    prior_stateseqss = [[] for _ in xrange(num_seqs)]
    prior_weights = []
    for itr in xrange(num_iter):
        hmm.resample_model() # sample parameters from the prior

        for itr2 in xrange(num_seqs):
            _, stateseq = hmm.generate(T,keep=False)
            prior_stateseqss[itr2].append(stateseq)

        prior_weights.append(hmm.obs_distns[0].weights)

    prior_stateseqss = np.array(prior_stateseqss)
    assert prior_stateseqss.shape == (num_seqs,num_iter,T)
    prior_weights = np.array(prior_weights)

    # generate state sequences and parameters using Gibbs
    for itr in xrange(num_seqs):
        hmm.generate(T,keep=True)
    assert len(hmm.states_list) == num_seqs

    gibbs_stateseqss = [[] for _ in xrange(num_seqs)]
    gibbs_weights = []
    for itr in xrange(num_iter):
        for s in hmm.states_list:
            s.generate_obs() # resamples data given state sequence, obs params
        hmm.resample_model() # resamples everything else as usual

        for itr2, s in enumerate(hmm.states_list):
            gibbs_stateseqss[itr2].append(s.stateseq)

        gibbs_weights.append(hmm.obs_distns[0].weights)

    gibbs_stateseqss = np.array(gibbs_stateseqss)
    assert gibbs_stateseqss.shape == (num_seqs,num_iter,T)
    gibbs_weights = np.array(gibbs_weights)

    # test that they look similar by checking probability of co-assignment
    time_indices = np.arange(T)
    seq_indices = np.arange(num_seqs)
    for itr in xrange(num_checks):
        i,j = np.random.choice(time_indices,replace=False,size=2)
        si,sj = np.random.choice(seq_indices,replace=True,size=2)
        prior_prob_of_coassignment = \
                (prior_stateseqss[si,:,i] == prior_stateseqss[sj,:,j]).std()
        gibbs_prob_of_coassignment = \
                (gibbs_stateseqss[si,:,i] == gibbs_stateseqss[sj,:,j]).std()

        assert np.isclose(
                prior_prob_of_coassignment,gibbs_prob_of_coassignment,
                rtol=0.025,atol=0.025,
                )

    # test that they look similar by checking parameters
    testing.populations_eq_quantile_plot(prior_weights,gibbs_weights,fig=fig)
    figpath = os.path.join(figure_dir_path,
            'discrete_geweke_multiple_seqs_test_weights.pdf')
    plt.savefig(figpath)

