from __future__ import division
import numpy as np

# These utilities are for aligned allocation. AFAICT numpy only guarantees
# 8-byte alignment, while SIMD instructions require 16-byte alignment (or even
# 32-byte alignment for AVX instructions).
# See e.g. http://stackoverflow.com/q/9895787

# TODO currently assumes C order

def _make_allocator(fn):
    def allocator(shape,dtype=float,alignment=32):
        nbytes = np.prod(shape)*np.dtype(dtype).itemsize
        buf = fn(nbytes+alignment,dtype=np.uint8)
        start = -buf.ctypes.data % alignment
        return buf[start:start+nbytes].view(dtype).reshape(shape)
    return allocator

zeros = _make_allocator(np.zeros)
empty = _make_allocator(np.empty)


def empty_like(a,**kwargs):
    return empty(a.shape,a.dtype,**kwargs)

def zeros_like(a,**kwargs):
    return zeros(a.shape,a.dtype,**kwargs)


def aligned(a,alignment=32):
    if a.ctypes.data % alignment == 0:
        return a
    else:
        extra = alignment // a.itemsize
        buf = empty(a.shape,a.dtype,alignment)
        buf[...] = a[...]
        return buf

