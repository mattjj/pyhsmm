from __future__ import division
import numpy as np

def _get_stats(model,grp):
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

def _get_sampled_stateseq(model,grp):
    if len(grp) == 0:
        return []

    datas, kwargss = zip(*grp)

    states_list = []
    for data, kwargs in zip(datas,kwargss):
        model.add_data(data,initialize_from_prior=False,**kwargs)
        states_list.append(model.states_list.pop())

    return [(s.stateseq, s.log_likelihood()) for s in states_list]

def _get_sampled_stateseq_and_labels(model,grp):
    if len(grp) == 0:
        return []

    data, kwargss = zip(*grp)

    states_list = []
    for data, kwargs in zip(datas,kwargss):
        model.add_data(data,initialize_from_prior=False,**kwargs)
        states_list.apppend(model.states_list.pop())

    return [(s.stateseq,s.component_labels,s.log_likelihood())
            for s in states_list]

