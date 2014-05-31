from __future__ import division
import numpy as np

def _get_stats(model,grp):
    datas, kwargss = zip(*grp)

    states_list = []
    for data, kwargs in zip(datas,kwargss):
        model.add_data(data,stateseq=np.empty(data.shape[0]),**kwargs)
        states_list.append(model.states_list.pop())

    for s in states_list:
        s.meanfieldupdate()

    return [s.all_expected_stats for s in states_list]

def _get_sampled_stateseq(model,grp):
    data, kwargss = zip(*grp)

    states_list = []
    for data, kwargs in zip(data,kwargss):
        model.add_data(data,stateseq=np.empty(data.shape[0]),**kwargs)
        states_list.append(model.states_list.pop())

    for s in states_list:
        s.resample()

    return [s.stateseq for s in states_list]

