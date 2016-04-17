"""Timing related decorator classes and functions.

Example usage:

from mxn.lib2.timer import Timer
with Timer() as tm:
   ## some code block (does not need to do anything with tm!)

or as decorator:

from mxn.timer import timer
@timer
def some_function()
"""

from __future__ import absolute_import, division, print_function
import time


class Timer(object):
    def __init__(self, title="", verbose=True, log=None):
        self.title = title if title!="" else "++TIMER++"
        self.verbose = verbose
        self.log = log

    def __enter__(self):
        s = "{} start: {}".format(self.title,time.ctime())
        if self.log is None: print(s)
        else: self.log.info(s)
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            if self.secs < 1:
                s = (self.title+' elapsed: {:.5} ms'.format(self.msecs))
            elif self.secs < 100:
                s = (self.title+' elapsed: {:.3f} s'.format(self.secs))
            else:
                s = (self.title+' elapsed: {:.3f} min'.
                     format(self.secs/60.0))
        if self.log is None: print(s)
        else: self.log.info(s)



def timer(f):
    def helper(*args, **kwargs):
        title = f.__name__
        start = time.time()
        print("++DECO-TIMER++ {} start: {}".format(title,time.ctime()))
        res = f(*args,**kwargs)
        end = time.time()
        secs = end - start
        msecs = secs * 1000  # millisecs
        print("++DECO-TIMER++ {} elapsed: ".format(title), end="")
        if secs < 1:
            print('%f ms' % msecs)
        elif secs < 100:
            print('%f s' % secs)
        else:
            print('%f min' % (secs/60.0))
        return res
    return helper


class DTimer(object):
    """Timer decorrator class."""

    def __init__(self, title="", verbose=True, log=None):
        self.title = title
        self.verbose = verbose
        self.log = log

    def __call__(self, f):
        def helper(*args, **kwargs):
            if self.title == "": self.title = f.__name__
            #self.title = "++DTIMER++ "+self.title
            if self.title=="": self.title = "++DTIMER++ "
            #s = "++DTIMER++ {} start: {}".format(self.title,time.ctime())
            s = "{} start: {}".format(self.title,time.ctime())
            if self.log is None: print(s)
            else: self.log.info(s)
            start = time.time()
            res = f(*args,**kwargs)
            end = time.time()
            secs = end - start
            msecs = secs * 1000  # millisecs
            #s = "++DTIMER++ "+self.title+" elapsed: "
            s = self.title+" elapsed: "
            if secs < 1:
                s += '{:.6f} ms'.format(msecs)
            elif secs < 100:
                s += '{:.3f} s'.format(secs)
            else:
                s += '{:.3f} min'.format(secs/60.0)
            if self.log is None: print(s)
            else: self.log.info(s)
            return res
        return helper

@timer
def timer_test():
    print("this is just a decorator timer test")
