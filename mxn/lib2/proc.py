""" Data Processing Routines and Classes

Includes:
- rebin / smooth
- image processing routines
- Tiling class
- block processing routines (block_xxx)

"""

import numpy as np
import scipy as sp
import scipy.ndimage

def rebin(a, shape=None, div=None):
    """Rebin/resize image to smaller image with averaging.

    Provide either new shape or divisors.

    If original shape is not a multiple of the new shape,
    then cut corners.
    """
    a = np.asarray(a) # if already an array, then no overhead.

    n_dim = len(a.shape)
    if shape is None and div is not None:
        shape = np.array(a.shape) // np.array(list(div)+[1]*(n_dim-len(div)))
    sh = np.array([[shape[i],a.shape[i]//shape[i]]
                   for i,x in enumerate(shape)]).flatten()

    if n_dim == 2:
        if not any(np.mod(a.shape, shape)):
            return a.reshape(sh).mean(-1).mean(1)
        else:
            return a[0:sh[0]*sh[1],0:sh[2]*sh[3]].reshape(sh).mean(-1).mean(1)
    elif n_dim == 3:
        if not any(np.mod(a.shape, shape)):
            return a.reshape(sh).mean(1).mean(2).mean(3)
        else:
            return a[0:sh[0]*sh[1],0:sh[2]*sh[3],0:sh[4]*sh[5]]\
                .reshape(sh).mean(1).mean(2).mean(3)
    elif n_dim == 4:
        if not any(np.mod(a.shape, shape)):
            return a.reshape(sh).mean(1).mean(2).mean(3).mean(4)
        else:
            return a[0:sh[0]*sh[1],0:sh[2]*sh[3],0:sh[4]*sh[5],0:sh[6]*sh[7]]\
                .reshape(sh).mean(1).mean(2).mean(3).mean(4)
    else:
        print("Error in rebin: case not considered.")


def smooth(data,smm,**kwargs):
    """Boxcar/idl type smooth, based on scipy.ndimage.uniform_filter.

    data : 2d array (can be modified to more).
    """
    if type(data[0,0]) in [complex,np.complex64,np.complex128]:
        res = np.empty(data.shape, dtype=complex)
        res.real = sp.ndimage.uniform_filter(np.real(data),smm,**kwargs)
        res.imag = sp.ndimage.uniform_filter(np.imag(data),smm,**kwargs)
        return res
    else:
        return sp.ndimage.uniform_filter(data,smm,**kwargs)

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None):
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    from scipy.signal import convolve
    g = gauss_kern(n, sizey=ny)
    improc = convolve(im,g, mode='valid')
    return(improc)


class Tiling:
    """1-d block-processing.

    Given size (n) and blocksize (d), will compute the blocksizes for
    individual elements.

    Parameters:
    - dim : dimension (or shape, then left-most is used for dim
    - bs : blocksize (default: 256)
    - clip : either to only use complete bs blocksizes (and
      skiping last if not large enough)

    To be extended to include:
    - overlapping
    - handling of left-over
    - offset
    - etc.

    Right now, this is just a basic implementation to get started.

    Usage example:
    tile = Tiling(ydim, bs)
    for ind in tile:
        out[ind,:,:] = in[ind,:,:]**2
    """
    def __init__(self, dim_or_shape, bs=3000, clip=True):
        if np.iterable(dim_or_shape):
            self.dim = dim_or_shape[0]
        else:
            self.dim = dim_or_shape
        self.bs = bs
        self.clip = clip
        if self.clip:
            self.n_blocks = self.dim // bs
            self.blocksizes = np.zeros(self.n_blocks,dtype=int) + self.bs
        else:
            self.n_blocks = np.ceil(self.dim / bs)
            self.blocksizes = np.zeros(self.n_blocks,dtype=int) + self.bs
            self.blocksizes[-1] = self.dim - bs*(self.n_blocks-1)
        self.s = np.arange(self.n_blocks)*self.bs    # start index
        self.e = np.cumsum(self.blocksizes)          # end+1 index
        # allows to use array[t.s[i]:t.e[i], :, :]
    def __getitem__(self, i):
        return np.arange(self.s[i], self.e[i], dtype='int')



def block_filter(func, data, bs=3000, out=None, dtype=None, clip=False, **kwargs):
    """Apply filter/function blockwise.
    Return array (out) has same dimension as input array (data).
    For Tiling, clip==False, to process the same number of lines.

    Example usage:
    >> out = block_filter(boxcar, amp**2, 1000, div=(5,15))
    - will apply boxcar filter with [5,15] window to data, in blocks
      of 1000 lines. out will be created inside.
      - Actually, boxcar is a bad example, as it needs overlapping
        windows!
    >> block_filter(boxcar, amp**2, out, 1000, div=(5,15))
    - apply same as before, but both, data and out, are already defined,
      possibly as np.memmap(), so that they don't use up the memory.
    """
    tiles = Tiling(data.shape, bs, clip=clip)
    if out is None:
        if dtype is None:
            dtype = data.dtype
        dim = list(data.shape)
        dim[0] = sum(tiles.blocksizes)
        out = np.empty(dim, dtype=dtype)
    for block in tiles:
        out[block, ...] = func(data[block, ...], **kwargs)
    return out

def block_filter2(func, data, data2, bs=3000, out=None,
                  dtype=None, clip=False, **kwargs):
    """Apply filter/function blockwise, combining 2 input data sets.
    Function should take two parameter data sets, and combine them to 1.
    """
    tiles = Tiling(data.shape, bs, clip=clip)
    if out is None:
        if dtype is None:
            dtype = data.dtype
        dim = list(data.shape)
        dim[0] = sum(tiles.blocksizes)
        out = np.empty(dim, dtype=dtype)
    for block in tiles:
        out[block, ...] = func(data[block, ...], data2[block, ...], **kwargs)
    return out

def block_filter3(func, data, data2, data3, bs=3000, out=None,
                  dtype=None, clip=False, **kwargs):
    """Apply filter/function blockwise, combining 2 input data sets.
    Function should take two parameter data sets, and combine them to 1.
    """
    tiles = Tiling(data.shape, bs, clip=clip)
    if out is None:
        if dtype is None:
            dtype = data.dtype
        dim = list(data.shape)
        dim[0] = sum(tiles.blocksizes)
        out = np.empty(dim, dtype=dtype)
    for block in tiles:
        out[block, ...] = func(data[block, ...], data2[block, ...],
                               data3[block, ...], **kwargs)
    return out

def block_filter4(func, data, data2, data3, data4, bs=3000, out=None,
                  dtype=None, clip=False, **kwargs):
    """Apply filter/function blockwise, combining 2 input data sets.
    Function should take two parameter data sets, and combine them to 1.
    """
    tiles = Tiling(data.shape, bs, clip=clip)
    if out is None:
        if dtype is None:
            dtype = data.dtype
        dim = list(data.shape)
        dim[0] = sum(tiles.blocksizes)
        out = np.empty(dim, dtype=dtype)
    for block in tiles:
        out[block, ...] = func(data[block, ...], data2[block, ...],
                               data3[block, ...], data4[block, ...], **kwargs)
    return out
def block_filter5(func, data, data2, data3, data4, data5, bs=3000, out=None,
                  dtype=None, clip=False, **kwargs):
    """Apply filter/function blockwise, combining 2 input data sets.
    Function should take two parameter data sets, and combine them to 1.
    """
    tiles = Tiling(data.shape, bs, clip=clip)
    if out is None:
        if dtype is None:
            dtype = data.dtype
        dim = list(data.shape)
        dim[0] = sum(tiles.blocksizes)
        out = np.empty(dim, dtype=dtype)
    for block in tiles:
        out[block, ...] = func(data[block, ...], data2[block, ...],
                               data3[block, ...], data4[block, ...],
                               data5[block, ...], **kwargs)
    return out

def block_gather(func, data, bs=3000, out=None, dtype=None, clip=True, **kwargs):
    """Apply filter/function blockwise, assuming the entire block will
    result in a single line.
    - Can be extended to have bs_out be bigger than 1.
    - By default, for Tiling, clip==True.

    Example usage:
    >> out = block_gather(lambda x: x.mean(axis=1), amp**2, 10)
      - presumming only in y-direction (taking the mean along 10 y values)
    """
    tiles = Tiling(data.shape, bs, clip=clip)
    if out is None:
        if dtype is None:
            dtype = data.dtype
        dim = [data.shape[0]//bs] + list(data.shape[1:])
        out = np.empty(dim, dtype=dtype)
    for i,block in enumerate(tiles):
        out[i, ...] = func(data[block, ...], **kwargs)
    return out

def block_rebin(func, data, bs=3000, out=None,
                dtype=None, div=None, clip=True, **kwargs):
    """Performs operation (filter/func) on data, and then rebins and saves in out.
    Doesn't check, but expects that bs is multiple of div[0]"""
    if div is None: raise Exception("block_rebin expects a div!")
    tiles = Tiling(data.shape, bs, clip=clip)
    if not clip:
        raise Exception("clip==False not supported yet")
    if out is None:
        if dtype is None:
            dtype = data.dtype
        nd = len(div)
        dim = list(data.shape[:nd]//np.array(div)) + list(data.shape[nd:])
        dim[0] = sum(tiles.blocksizes) // div[0]
        out = np.empty(dim, dtype=dtype)
    for i,block in enumerate(tiles):
        out[tiles.s[i]//div[0]:tiles.e[i]//div[0], ...] = rebin(func(data[block, ...], **kwargs),div=div)
    return out

def block_rebin2(func, data, data2, bs=3000, out=None,
                 dtype=None, div=None, clip=True, **kwargs):
    """Performs operation (filter/func) on 2 data sets, and then rebins the result.
    Doesn't check, but expects that bs is multiple of div[0]"""
    if div is None: raise Exception("block_rebin expects a div!")
    tiles = Tiling(data.shape, bs, clip=clip)
    if out is None:
        if dtype is None:
            dtype = data.dtype
        nd = len(div)
        dim = list(data.shape[:nd]//np.array(div)) + list(data.shape[nd:])
        dim[0] = sum(tiles.blocksizes) // div[0]
        out = np.empty(dim, dtype=dtype)
    for i,block in enumerate(tiles):
        out[tiles.s[i]//div[0]:tiles.e[i]//div[0], ...] = rebin(
            func(data[block, ...], data2[block, ...], **kwargs),div=div)
    return out

def block_rebin3(func, data, data2, data3, bs=3000, out=None,
                 dtype=None, div=None, clip=True, **kwargs):
    """Performs operation (filter/func) on 3 data sets, and then rebins the result.
    Doesn't check, but expects that bs is multiple of div[0]"""
    if div is None: raise Exception("block_rebin expects a div!")
    tiles = Tiling(data.shape, bs, clip=clip)
    if out is None:
        if dtype is None:
            dtype = data.dtype
        nd = len(div)
        dim = list(data.shape[:nd]//np.array(div)) + list(data.shape[nd:])
        dim[0] = sum(tiles.blocksizes) // div[0]
        out = np.empty(dim, dtype=dtype)
    for i,block in enumerate(tiles):
        out[tiles.s[i]//div[0]:tiles.e[i]//div[0], ...] = rebin(
            func(data[block, ...], data2[block, ...], data3[block, ...],
                 **kwargs),div=div)
    return out
