# Helper to scale an array (with complex value support) into a new range
import numpy as np
import math

def _scale_array(S,lower,upper):
    absS = abs(S)
    return (absS - absS.min())/(absS.max()-absS.min()) * (upper - lower) + lower

def to_uint8(S, low = 0, high = 255):
    return _scale_array(S, 0, 255).astype('uint8')

def to_uint16(S, low = 0, high = 65535):
    return _scale_array(S, 0, 255).astype('uint16')

def to_float32(S, low = 0.0, high = 1.0):
    return _scale_array(S,low, high).astype('float32')

def normalise(S):
    absS = [abs(s0) for s0 in S]
    return (absS - np.mean(absS)) / np.std(absS)

def to_log_scale(S, offset = 1e-9):
    return np.log10(S + offset)