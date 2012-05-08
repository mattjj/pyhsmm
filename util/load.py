import os
import string
import struct
import numpy

class load(object):
   def __init__(self, dim):
      self.utts = []
      self.tags = []
      self.segs = []
      self.dim = dim 

   def load_data(self, fname_list):
      flist = open(fname_list)
      line = flist.readline().strip()
      while line:
         tokens = line.strip().split()
         tag = tokens[0]
         fname_in = tokens[1]
         utt = list() 
         print 'Loading ', fname_in 
         with open(fname_in, 'rb') as f:
            frame = list() 
            c = f.read(4)
            while c != "":
               frame.append(struct.unpack("f", c)[0])
               if len(frame) % self.dim == 0:
                  utt.append(numpy.array(frame))
                  frame = list() 
               c = f.read(4)
            if len(frame) != 0:
               print 'File loading Error: ', fname_in
         line = flist.readline().strip()
         self.utts.append(numpy.vstack(utt).copy())
         self.tags.append(tag)
      return self.tags, self.utts 
  
   def load_pre_segs(self, file_dir, ext):
      for t in self.tags:
         fname = file_dir + '/' + t + '.' + ext
         fseg = open(fname)
         line = fseg.readline()
         segs = list()
         while line:
            tokens = line.strip().split()
            segs.append([int(tokens[0]), int(tokens[1]) + 1])
            line = fseg.readline()
         fseg.close()
         self.segs.append(segs)
      return self.segs

   def print_input(self):
      print self.tags, self.utts
