from numpy import *
from ctypes import *
from numpy.ctypeslib import ndpointer

lib = cdll.LoadLibrary("./accel_c.so")
lib.argmindist_double.argtypes = [
    c_int,c_int,
    ndpointer(dtype='float',flags='C_CONTIGUOUS'),
    ndpointer(dtype='float',flags='C_CONTIGUOUS')]

def argmindist(v,data):
    print v,data
    assert prod(data.shape[1:])==prod(v.shape)
    return lib.argmindist_double(
        prod(data.shape[1:]),
        data.shape[0],
        v,
        data)
    

