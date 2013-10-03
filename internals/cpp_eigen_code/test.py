from __future__ import division
import numpy as np
from numpy.core.umath_tests import inner1d

import test2

def clockmatrix(r,p):
    return np.diag(np.repeat(p,r-1),k=1) + (1-p)*np.eye(r)

def construct_trans(super_trans,negbin_params,sub_transs,sub_initstates):
    rs, ps = zip(*negbin_params)
    Nsubs = [pi_sub.shape[0] for pi_sub in sub_initstates]
    N = super_trans.shape[0]

    blocksizes = [r*Nsub for r, Nsub in zip(rs,Nsubs)]
    blockstarts = np.concatenate(((0,),np.cumsum(blocksizes)[:-1]))

    out = np.zeros((sum(blocksizes),sum(blocksizes)))

    # blocks are kron(clockmatrix,subtrans)
    for (r,p), subtrans, blockstart, blocksize in zip(negbin_params, sub_transs, blockstarts, blocksizes):
        out[blockstart:blockstart+blocksize,blockstart:blockstart+blocksize] = \
                np.kron(clockmatrix(r,p),subtrans)

    # off-blocks are init states scaled by super_trans
    for i, (iblockstart, iblocksize, p, Nsub) in enumerate(zip(blockstarts, blocksizes, ps, Nsubs)):
        for j, (init_distn,jNsub,jblockstart) in enumerate(zip(sub_initstates,Nsubs,blockstarts)):
            if i != j:
                out[iblockstart+iblocksize-Nsub:iblockstart+iblocksize,jblockstart:jblockstart+jNsub] = \
                        np.outer(np.repeat(p,Nsub),init_distn) * super_trans[i,j]

    return out

def mult(v,super_trans,negbin_params,sub_transs,sub_initstates):
    rs, ps = zip(*negbin_params)
    Nsubs = [pi_sub.shape[0] for pi_sub in sub_initstates]
    N = super_trans.shape[0]

    blocksizes = [r*Nsub for r, Nsub in zip(rs,Nsubs)]
    blockstarts = np.concatenate(((0,),np.cumsum(blocksizes)[:-1]))

    out = np.zeros(sum(blocksizes))

    # collect across parts (do inside other loop in message-passing code)
    # but still collect as a vector of scalars!
    incomings = [pi.dot(v[blockstart:blockstart+Nsub]) for pi,blockstart,Nsubs
            in zip(sub_initstates,blockstarts,Nsubs)]

    for i, ((r,p), subtrans, blockstart, blocksize) \
            in enumerate(zip(negbin_params, sub_transs, blockstarts, blocksizes)):

        # within-block
        block = slice(blockstart,blockstart+blocksize)
        out[block] = \
                clockmatrix(r,p).dot(v[block].reshape((r,-1))).dot(subtrans.T).ravel()

        # across block, loop over j
        Nsub = subtrans.shape[0]
        end = slice(blockstart+blocksize-Nsub,blockstart+blocksize)
        out[end] += p*super_trans[i].dot(incomings)

    return out


def rand_trans(n):
    A = np.random.rand(n,n).astype('float32')
    A /= A.sum(1)[:,None]
    return A

def rand_init(n):
    pi = np.random.rand(n,).astype('float32')
    pi /= pi.sum()
    return pi


def log_messages_backwards(A,aBl):
    errs = np.seterr(divide='ignore')
    Al = np.log(A)
    np.seterr(**errs)

    betal = np.zeros_like(aBl)

    for t in xrange(betal.shape[0]-2,-1,-1):
        np.logaddexp.reduce(Al + betal[t+1] + aBl[t+1],axis=1,out=betal[t])

    return betal

def normalized_messages_backwards(A,aBl):
    betan = np.empty_like(aBl)
    logtot = 0.

    betan[-1] = 1.
    for t in xrange(betal.shape[0]-2,-1,-1):
        cmax = aBl[t+1].max()
        betan[t] = A.dot(betan[t+1] * np.exp(aBl[t+1] - cmax))
        logtot += cmax + np.log(betan[t].sum())
        betan[t] /= betan[t].sum()

    return betan, logtot


if __name__ == '__main__':
    N = 3
    Nsub = 2

    super_trans = rand_trans(N)
    super_trans.flat[::super_trans.shape[0]+1] = 0.
    super_trans /= super_trans.sum(1)[:,None]
    sub_transs = [rand_trans(Nsub) for i in range(N)]
    sub_initstates = [rand_init(Nsub) for i in range(N)]
    negbin_params = [(2,0.7) for i in range(N)]

    A = construct_trans(super_trans,negbin_params,sub_transs,sub_initstates)

    v = np.random.rand(A.shape[0],).astype('float32')

    print np.allclose(A.dot(v),mult(v,super_trans,negbin_params,sub_transs,sub_initstates))
    print np.allclose(A.sum(1),1.)

    rs, ps = zip(*negbin_params)
    out = test2.fast_mult(v,super_trans,np.array(rs,dtype='int32'),np.array(ps,dtype='float32'),sub_transs,sub_initstates)

    print np.allclose(out,A.dot(v))


    aBls = [np.log(np.random.rand(100,sub_trans.shape[0]).astype('float32')) for sub_trans in sub_transs]
    aBl = np.concatenate([np.tile(aBl,(1,r)) for aBl,r in zip(aBls,rs)],axis=1)
    betal = log_messages_backwards(A,aBl)

    betan, logtot = normalized_messages_backwards(A,aBl)

    print np.isclose(np.logaddexp.reduce(betal[0]),logtot)

    betan2, logtot2 = test2.messages_backwards_normalized(
            super_trans,np.array(rs,dtype='int32'),np.array(ps,dtype='float32'),sub_transs,sub_initstates,aBls)

    print np.isclose(logtot2, logtot)
    print np.allclose(betan2,betan)

