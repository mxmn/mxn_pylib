"""File/Data Input / Output Routines and Classes.


Includes:
- binary name-labeled file in/out routines: rdat/sdat
- rat format files
- h5py handling

"""

import numpy as np
import scipy as sp
import pylab as plt
import os.path

import time
import h5py
import glob



def find_dim_label(s):
    """Types (following numpy's dtype char):
    fdgbhilFD:
      - f=float32, d:double=float64, g:float128,
      - b=byte=int8, h=half=int16, i=int=int32, l=long=int64,
      - F=complex64(floats), D=complex128(doubles)
    Example dim_labels:
      1234x124_d
    """
    import re
    # starts with "_" and is followed by another "_" or "." for file ending
    pattern = re.compile("_\d+[x\d+]+_[fdgbhilFD][._]")
    reg = pattern.search(s)
    shape = tuple([int(x) for x in reg.group()[1:-3].split("x")])
    dtype = reg.group()[-2]
    return shape, dtype
def slfb(filename, data, new_dtype=None, no_label=False, envi=False, **kwargs):
    """Save Labeled-Flat-Binary data.

    filename is the prefix (including path). Dimension and type label are added
    automatically.

    Using numpy's tofile and to be read either by memmap or fromfile.
    An alternative is to use numpy's .npy and .npz formats. But then one would
    need to write readers/writers for other languages.

    envi : bool, either to save an ENVI header file.
    """
    if not isinstance(data, (np.ndarray, h5py._hl.dataset.Dataset) ):
        raise Exception("Cast data as np.array or h5py dataset")
    if new_dtype is not None and np.dtype(new_dtype) != data.dtype:
        if isinstance(data, np.ndarray):
            data = data.astype(new_dtype)
        elif isinstance(data, h5py._hl.dataset.Dataset):
            # workaround: cast as numpy array
            data = np.array(data).astype(new_dtype)
            # raise Exception("Doesn't work yet with casting h5py dataset"
            #                 " to another type when saving data.")
            # with data.astype(new_dtype):
            #     slfb(filename, dat, no_label=no_label, envi=envi, **kwargs)
            # return
    dtype = data.dtype.char
    if no_label:
        if not (filename.endswith(".dat") or filename.endswith(".bin")):
            full_filename = filename + ".dat"
        else:
            full_filename = filename
    else:
        full_filename = filename + "_" + "x".join([str(x) for x in data.shape]) + "_" + dtype+".dat"
    with open(full_filename,'w+') as fid:
        if isinstance(data, np.ndarray):
            data.tofile(fid)
        if isinstance(data, h5py._hl.dataset.Dataset):
            np.array(data).tofile(fid)
    if envi:
        save_envi_hdr(full_filename, data=data)
def sdat(*args, **kwargs):
    """Usage: sdat(filename, data)"""
    slfb(*args, **kwargs)
def rlfb(filename, mode='c', shape=None, dtype=None,
         _filename=None, # for output only
         **kwargs):
    """Read Labeled-Flat-Binary data, in np.memmap.

    possible modes:
    'r'   Open existing file for reading only.
    'r+'  Open existing file for reading and writing.
    'w+'  Create or overwrite existing file for reading and writing.
    'c'   Copy-on-write: assignments affect data in memory, but changes are not
          saved to disk. The file on disk is read-only.
    """
    if not os.path.exists(filename):
        candidates = glob.glob(filename + "*x*_*.dat")
        if len(candidates) > 1:
            print("\nMultiple candidates for reading: ",candidates)
            print("\nSelect one of the files and read again!\n")
            return None
        elif len(candidates) == 0:
            print("No file found that would match the file root: ",filename)
        else:
            _filename = candidates[0]
            print("Reading file: ",os.path.basename(_filename))
    else:
        _filename = filename
    if shape is not None:
        if dtype is None:
            dtype = 'f'
    else:
        shape, dtype = find_dim_label(_filename)
    return np.memmap(_filename, dtype=dtype, mode=mode, shape=tuple(shape))
def rdat(*args, **kwargs):
    # one could build in automatic search for filenames with arbitrary dim labels
    return rlfb(*args, **kwargs)
def ldat(*args, **kwargs):
    return rlfb(*args, **kwargs)


def save_envi_hdr(filename, d=None, data=None,
                  samples=None, lines=None, bands=1, interleave="bip",
                  dtype=None, offset=0,
                  map_info=""):
    """Saves envi header, given dict d, and filename.

    IDL dtype:
      1: byte (8b)
      2: short/int (int16)
      3: long/int (int32)
      4: float32
      5: float64/double
      6: complex64
      9: complex128
      12-13: unsigned int/long
      14-15: long64
    Interleave schemes:
      bip: [az, rg, ch]
      bil: [az, ch, rg]
      bsq: [ch, az, rg]
    """
    if d is None: d = {}
    if "desription" not in d: d["description"] = filename
    if "samples" not in d: d["samples"] = samples
    if "lines" not in d: d["lines"] = lines
    if "bands" not in d: d["bands"] = bands
    if "interleave" not in d: d["interleave"] = interleave

    if "dtype" not in d: d["dtype"] = dtype
    if "offset" not in d: d["offset"] = offset
    if "map_info" not in d: d["map_info"] = map_info

    if data is not None:
        n = np.shape(data)
        nd = len(n)
        if nd > 2: raise Exception("Not implemented yet")
        d["lines"] = n[0]
        d["samples"] = n[1]
        if dtype is None:
            dt = data.dtype
            if   dt in [np.complex64]: dtype = 6
            elif dt in [np.complex128]: dtype = 9
            elif dt in [np.int8]: dtype=1
            elif dt in [np.int16]: dtype=2
            elif dt in [np.int32]: dtype=3
            elif dt in [np.int64]: dtype=14
            elif dt in [np.float32]: dtype=4
            elif dt in [np.float64]: dtype=5
            d["dtype"] = dtype


    header = '''ENVI
description = {{{description}}}
samples = {samples}
lines = {lines}
bands = {bands}
header offset = {offset}
data type = {dtype}
interleave = {interleave}
sensor type = Unknown
byte order = 0
map info = {map_info}
wavelength units = Unknown'''.format(**d)

    with open(filename+".hdr",'w') as fid:
        fid.write(header)

def envi_hdr_info(filename):
    """Returns a dict of envi header info. All values are strings."""
    with open(filename,'r') as fid:
        x = [line.split("=") for line in fid]
    envi_info = {a[0].strip(): a[1].strip() for a in x if len(a) ==2}
    return envi_info

class H5PY:
    """To set up h5py file, and convenince methods.

    Most simple:

    NO, DON'T DO THIS: h5f = H5PY(tdm.rpath + "test.h5").h5f
    --> this will crash <--
    because the reference to object is lost, and h5f closes the file!
    - therefore, always generate and keep the full object!
    - and just use the [] notations.
    """

    def __init__(self, filename):
        try:
            self.h5f = h5py.File(filename,'a')
        except:
            print("H5PY: Not able to open with 'a'")
            try:
                self.h5f = h5py.File(filename,'r+')
            except:
                print("H5PY: Not able to open with 'r+'")
                try:
                    self.h5f = h5py.File(filename,'w')
                except:
                    print("H5PY: Could not open {} file (even with 'w')!"
                          .format(filename))
    def __getitem__(self, name):
        return self.h5f[name]
    def __setitem__(self, name, val):
        if name in self.h5f:
            del self.h5f[name]
        self.h5f[name] = val
    def h5_file(self):
        return self.h5f
    def __del__(self):
        try:
            self.h5f.close()
        except:
            print("H5PY: could not close h5f file correctly")


def load_file(filename, dtype='float32', mode='r', shape=None,
              memory=False, h5f=None):
    """Failure tollerant load of data.

    Should support: memmap, memory (not supported yet), h5py dataset.

    dtype can accept array of types. If file size doesn't match the first, it
    is tried with the following data types.

    dtype : scalar, or array-like
       list of dtypes to check.

    """
    if h5f is not None:
        if filename in h5f:
            return h5f[filename]
        else:
            print("Dataset does not exist in h5 file: ",filename)
            return None
    if not os.path.exists(filename):
        print("File does not exist: ",filename)
        return None
    if shape is None:
        print("Need to provide at least the shape (image dimensions)")
        return None
    this_dtype = None
    if np.isscalar(dtype):
        if (os.path.getsize(filename) == np.prod(shape) * np.dtype(dtype).itemsize):
            this_dtype = dtype
    else:
        for dt in dtype:
            if os.path.getsize(filename) == np.prod(shape) * np.dtype(dt).itemsize:
                this_dtype = dt
    if this_dtype is None:
        print("Image dimensions and data type don't match with file type!",
              "\nFile not loaded -- recheck: ",filename)
        return None
    # if os.path.getsize(filename) != np.prod(shape) * np.dtype(dtype).itemsize:
    #     print("Image dimensions and data type don't match with file type!",
    #           "\nFile not loaded -- recheck: ",filename)
    #     return None
    return np.memmap(filename, dtype=this_dtype,mode=mode,shape=shape)


def get_file(pattern):
    """Get filename from a shell expression.

    pattern : str, unix-type pattern for the file to search.

    Returns a single file or None if not found.
    If more than one file found, prints a warning, and returns first file.
    """
    files = glob.glob(pattern)
    if len(files) == 0:
        return None
    elif len(files) == 1:
        return files[0]
    else:
        print("WARNING: more than 1 file names matched the pattern. Return first")
        print("All matches: ",files)
        return files[0]
