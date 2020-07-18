'''
Misc stats functions
'''

# pylint: disable=C0103, R0912, R0914

import numpy as np

def logistic(x, a, b):
    '''
    Calculate logistic function of ndarray x, with offset a and slope 1/b
    '''
    return  1/(1+np.exp((-x+a)/b))
