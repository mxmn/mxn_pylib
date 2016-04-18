"""Basic utility lib"""

from __future__ import absolute_import, division, print_function

import os

import numpy as np


def time_now():
    from datetime import datetime
    return datetime.now().strftime('%y/%m/%d %H:%M:%S')


def minmax(data):
    if type(data) is np.ndarray:
        return np.array([np.min(data), np.max(data)])
    else: return np.array([min(data), max(data)])

def in_range(val, vmin, vmax=None):
    """Checks either value is in range of [vmin..vmax],
    or within the min/max values of provided array
    """
    if np.isscalar(vmin):
        return vmin <= val <= vmax
    elif isinstance(vmin, slice) and vmax is None:
        if vmin.step is None or vmin.step > 0:
            return vmin.start <= val <= vmin.stop
        else:
            return vmin.start >= val >= vmin.stop
    else:
        if vmax is None: vmax = vmin
        return np.min(vmin) <= val <= np.max(vmax)
def arr_in_range(a, vmin, vmax=None):
    """Returns bool array, noting either elements inside vmin/vmax.
    vmin can be 2-elem range."""
    if vmax is None:
        vmin, vmax = vmin

    return np.logical_and(a >= vmin, a <= vmax)

def testing(x):
    np.max(xa)


def slice_extend(sl, max_stop, min_start=0):
    """Failure-tollerant slice extension in a specified range.

    Given a slice, checks that sl.start is not less than min_start,
    and that s.stop is not more or equal to max_stop.
    If one condition appears, adjust slice (transforming to list)
    by filling with min_start or (max_stop-1).
    """
    sl_pre, first = (min_start - sl.start if sl.start < min_start else 0,
                     max(min_start, sl.start))
    sl_post, last = (sl.stop - max_stop if sl.stop > max_stop else 0,
                     min(max_stop, sl.stop))
    if sl_pre != 0 or sl_post != 0:
        sl = ([min_start,]*sl_pre + list(range(first, last)) +
              [max_stop-1,]*sl_post)
    return sl


def info(a):
    """Provide maximum of information on arbitrar input data structure.
    mn, 3/11/15"""
    if type(a) is np.ndarray:
        mm(a)
    elif not np.iterable(a):
        print("scalar value: ",a," type: ",type(a))
    elif type(a) is dict:
        print("dict keys: ",list(a.keys()))
        for i,k in enumerate(a.keys()):
            if i <= 3:
                print("...info on dict element {}: {}".format(i,k))
                info(a[k])
    elif type(a) is list:
        t = {type(x) for x in a}
        print("list: number of elements: ",len(a)," types: ",t)
        for i in range(min(3,len(a))):
            print("...info on list element {}:".format(i))
            info(a[i])
    else:
        print("type: ",type(a))
        try:
            a.info()
        except:
            print(a)

def isfloat(d):
    assert type(d) is np.ndarray
    return np.issubdtype(d.dtype, float)
def isint(d):
    assert type(d) is np.ndarray
    return np.issubdtype(d.dtype, int)
def isbool(d):
    assert type(d) is np.ndarray
    return np.issubdtype(d.dtype, bool)


# Basic stats
def mm(data, f=None):
    """Some basic stats about the data.

    :param f: string; number format specification.
    """
    if len(data) == 0:
        print "[empty]"
        return
    if isinstance(data, np.ndarray):
        if f is None:
            if np.issubdtype(data.dtype, float):
                f = u'{:.7}'
            elif np.issubdtype(data.dtype, int):
                f = u'{:7}'
            elif np.issubdtype(data.dtype, bool):
                f = u'{:1}'
            else:
                f = u'{}'
        s1 = "np.{} {}: ".format(data.dtype, data.shape)
        s2 = (f+" {} "+f+" ["+f+" .. "+f+"]").format(
            data.mean(), u'\u00B1', data.std(), data.min(), data.max())
        if len(s1)+len(s2) <= 80:
            print s1+s2
        else:
            print s1+"\n  "+s2
    elif isinstance(data, list):
        print 'Length:', len(data), 'Type:', type(data)
        print 'Stats:', np.mean(data), u'\u00B1', np.std(data), 'in [', min(data), '..', max(data), ']'
    else: print 'Value:', data

def mmf(a, round=None, fmt=None):
    """Returns a string with main stats info. (assumes numpy array)"""
    if len(a) == 0:
        return "[empty]"
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    if fmt is None:
        fmt = '{:.3f}'
    if a.dtype.kind in ['i', 'u']:
        return (fmt+u' \u00B1'+fmt+' [{}..{}]').format(
            a.mean(), a.std(), a.min(), a.max())
    return (fmt+u' \u00B1'+fmt+' ['+fmt+'..'+fmt+']').format(
        a.mean(), a.std(), a.min(), a.max())

def mmfm(a, round=None):
    """Returns a string with main stats info. (assumes numpy array)"""
    if type(a) is not np.ndarray: a = np.array(a)
    return u'{:.2f} \\u00B1{:.2f} [{}..{}] med: {} #: {}'\
        .format(a.mean(),a.std(),a.min(),a.max(),np.median(a),len(a))

# to be updated, e.g. including round after point; +median +number of elemetns

def form(data):
    """Size/form/shape of data."""
    print((np.shape(data)))


def print_parameters(obj, keys=None, title=None):
    """Nice print of attributes of an object."""
    import types
    if keys is None:
        keys = list(obj.__dict__.keys())
        keys.sort()
    numeric_types = set([float, int, complex, bool,
                         np.float32, np.float64, np.int64, np.int])
    string_types = set([bytes, str])
    if title is None:
        title = obj.__class__
    print(title)
    ksize = 0
    for k in keys:
        ksize = max(ksize, len(k))
    ksize = min(30,ksize)
    ks = "{:>"+str(ksize)+"s}"+"  "  # +"\t"
    for key in keys:
        value = obj.__dict__[key]
        if type(value) in numeric_types:
            print((ks+"{}").format(key,value))
        elif type(value) in string_types:
            print((ks+"{:20s}").format(
                key,value[max(0,len(value)-(80-ksize)):]))
        elif isinstance(value, np.ndarray):
            if value.size <= 3:
                print((ks+"{}").format(key,value))
            else:
                print((ks+"np{} <{}>").format(key,value.shape,value.dtype.name))
        elif isinstance(value, (list, tuple)):
            if len(value) <=3 and (type(value[0]) in numeric_types or
                                   type(value[0]) in string_types):
                print((ks+"{}").format(key,value))
            else:
                print((ks+"{} len: {}>").format(key,type(value),len(value)))
        else: print((ks+"{}").format(key,type(value)))

def chunks(arr, dx):
    """Generator that yields chunks of size dx from list.
    Last one might be shorter."""
    for i in range(0,len(arr),int(dx)):
        yield arr[i:i+int(dx)]

def pars(*args, **kwargs):
    """Obsolote, use print_parameters() instead."""
    print_parameters(obj, *args, **kwargs)

def timestamp():
    """Returns string with a current time timestamp"""
    import datetime
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

def timedcall(func, *args):
    import time
    t0 = time.time()
    result = func(*args)
    t1 = time.time()
    return t1-t0, result

def timedcalls(n, func, *args):
    """If n integer, run n times
    If n is float, run up to n seconds."""
    import time
    if type(n) is int:
        times = [timedcall(func, *args)[0] for _ in range(n)]
    elif type(n) is float:
        times = []
        while sum(times) < n:
            times.append(timedcall(func, *args)[0])
    return min(times), sum(times)/len(times), max(times)


def normscl(a, vmin=0, vmax=1, byte=False):
    """Scales/normalizes array to given range.

    Default range: 0.0 to 1.0
    If byte==True: 0 to 255

    Change Log:
    - 10/9/15, mn: changed order of arguments (moved byte to last).
    """
    if byte:
        vmin = 0
        vmax = 255
    mn = np.min(a)
    mx = np.max(a)
    s = (a-mn) / (mx-mn) * (vmax-vmin) + vmin
    if byte:
        s = s.astype(int)
    return s

def bytscl(d, vmin=None, vmax=None):
    """Similar to same-name IDL function.

    Almost same functionality as normscl(d,byte=True), but really converting
    to 8-bit data (unsigned int).
    """
    if vmin is None: vmin=np.min(d)
    if vmax is None: vmax=np.max(d)
    res = d-vmin
    res *= 255.0/(vmax-vmin) # '*' is marginally less expensive than '/'
    return res.astype('uint8')


def computer_info(as_dict=False):
    """Returns all potential info as dict or Dummy class for easy access."""
    import getpass, socket
    d = {
        'user' : getpass.getuser(),
        'home' : os.getenv("HOME"),
        'hostname' : socket.gethostname(),
    }
    if not as_dict:
        d = Dummy(**d)
    return d

#
# new classes go at the end of this file
#

class Dummy:
    """Generic class for easy import and representation of dict keys/values."""
    def __init__(self, **kwargs):
        """Can be initialized with arbitrary named parameters"""
        for k,v in kwargs.items():
            if k[0] != "_":
                setattr(self, k.replace(' ','_'), v)
    def info(self):
        """Print available parameters"""
        print_parameters(self)
    def update(self, other):
        """Update given another dict or Dummy object"""
        if isinstance(other, dict):
            for k,v in other.items():
                setattr(self, k, v)
        else: # import all fields
            for k,v in other.__dict__.items():
                if k[0] != "_":
                    setattr(self, k, v)
    def pickle(self, filename):
        import pickle
        with open(filename, 'wb') as fid:
            pickle.dump(self, fid)
    def load(self, filename):
        import pickle
        d = pickle.load(filename)
        for k,v in d.__dict__.items():
            setattr(self, k, v)
            self.k = v



def bbox(*arrs):
    """Returns bounding-box type array, similar to extent.
    e.g.:
    bbox(X, Y) => [minX, maxX, minY, maxY]
    """
    return np.array([minmax(a) for a in arrs]).ravel()

##
## to go to ml.py or stats.py module
##
def print_confusion_matrix(true, est, percentage=False):
    t, e = np.array(true), np.array(est)
    TP, FP, FN, TN = (np.sum(np.logical_and(t == 1, e == 1)),
                      np.sum(np.logical_and(t == 0, e == 1)),
                      np.sum(np.logical_and(t==1, e==0)),
                      np.sum(np.logical_and(t==0, e==0)))
    n, nt1, nt0, ne1, ne0 = (len(true),
                             np.sum(true),np.sum(1-true),
                             np.sum(est), np.sum(1-est))
    if percentage:
        TP, FP, FN, TN = np.array([TP,FP,FN,TN]) / n
        nt1,nt0,ne1,ne0 = np.array([nt1,nt0,ne1,ne0]) / n
    num_str="""        Confusion Matrix
             T R U E
PR   TP: {:<10} FP: {:<10}    Predicted=1: {}
ED   FN: {:<10} TN: {:<10}    Predicted=0: {}

True =1: {:<10} =0: {:<10}    Total      : {}"""
    perc_str="""        Confusion Matrix
             T R U E
PR   TP: {:<8.3f} FP: {:<8.3f}    Predicted=1: {:<8.3}
ED   FN: {:<8.3f} TN: {:<8.3f}    Predicted=0: {:<8.3}

True =1: {:<8.3f} =0: {:<8.3f}    Total      : {}"""
    print((perc_str if percentage else num_str).format(
        TP, FP, ne1, FN, TN, ne0,
        nt1, nt0, n))
