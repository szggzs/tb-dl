import numpy as np


'''
    Calculate the mean error of every band.
'''
def mean_error(pred, gt):
    err = np.fabs(pred - gt).mean(axis=1)
    return err
