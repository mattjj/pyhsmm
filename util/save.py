from __future__ import division
import numpy as np

class save(object):
    def __init__(self, data_id, stateseq_norep, durations):
        self.data_id = data_id
        self.stateseq_norep = stateseq_norep
        self.durations = durations
        self.output = ''

    def save_superstates(self, save_dir):
        '''
        find the start and end index for each phone segment
        output the boundary frame number as well as the superstate index
        the output file will be saved as save_dir/data_id.hyp
        '''
        end_indices = np.cumsum(self.durations) 
        start_indices = np.concatenate(([0],end_indices[0:-1]))
        for i in len(self.stateseq_norep):
            self.output += str(start_indices) + ' ' + \
                    str(end_indices) + ' ' + self.stateseq_norep(i) + '\n'
        fout = open(save_dir + '/' + self.data_id + '.hyp', 'w')
        fout.write(self.output)
        fout.close()
