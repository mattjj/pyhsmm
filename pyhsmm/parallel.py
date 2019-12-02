from __future__ import division
import numpy as np
from scipy.special import logsumexp

# NOTE: pass arguments through global variables instead of arguments to exploit
# the fact that they're read-only and multiprocessing/joblib uses fork

model = None
args = None

def _get_stats(idx):
    grp = args[idx]

    if len(grp) == 0:
        return []

    datas, kwargss = zip(*grp)

    states_list = []
    for data, kwargs in zip(datas,kwargss):
        model.add_data(data,stateseq=np.empty(data.shape[0]),**kwargs)
        states_list.append(model.states_list.pop())

    for s in states_list:
        s.meanfieldupdate()

    return [s.all_expected_stats for s in states_list]

def _get_sampled_stateseq(idx):
    grp = args[idx]

    if len(grp) == 0:
        return []

    datas, kwargss = zip(*grp)

    states_list = []
    for data, kwargs in zip(datas,kwargss):
        model.add_data(data,initialize_from_prior=False,**kwargs)
        states_list.append(model.states_list.pop())

    return [(s.stateseq, s.log_likelihood()) for s in states_list]

def _get_sampled_stateseq_and_labels(idx):
    grp = args[idx]
    if len(grp) == 0:
        return []

    data, kwargss = zip(*grp)

    states_list = []
    for data, kwargs in zip(datas,kwargss):
        model.add_data(data,initialize_from_prior=False,**kwargs)
        states_list.apppend(model.states_list.pop())

    return [(s.stateseq,s.component_labels,s.log_likelihood())
            for s in states_list]


cmaxes = None
alphal = None
scaled_alphal = None
trans_matrix = None
aBl = None
def _get_predictive_likelihoods(k):
    future_likelihoods = logsumexp(
            np.log(scaled_alphal[:-k].dot(np.linalg.matrix_power(trans_matrix,k))) \
                    + cmaxes[:-k,None] + aBl[k:], axis=1)
    past_likelihoods = logsumexp(alphal[:-k], axis=1)

    return future_likelihoods - past_likelihoods

