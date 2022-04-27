"""
Created on Thu Jan  6 15:21:18 2022

@author: linh.trinh

this is library to perform time series operation such as moving Average
"""

import numpy as np
import pandas as pd
import numpy.matlib as matlib

def count_data(input_m, windowSize):
    '''
    count number of valid data along a moving windowSize
    '''
    
    res_ = np.full_like(input_m, np.nan)
    
    for i in range(windowSize - 1, input_m.shape[0]):
        res_[i] = np.sum(np.isfinite(input_m[i - windowSize+1: i+1]), axis=0)
    
    return res_

def movingSum(input_m, windowSize):
    '''
    compute moving sum of input_m along column axis, with given windowSize
    NaN is treated as 0, but if all data is NaN then return sum as NaN
    '''
    
    assert windowSize > 0
    
    dataNum = count_data(input_m, windowSize)
    dataNum = dataNum.astype(float)
    
    nanFiltered_m = np.copy(input_m)
    nanFiltered_m[~np.isfinite(nanFiltered_m)] = 0.0
    cumsum = np.cumsum(nanFiltered_m[::-1], 0)[::-1]
    result = np.ndarray(input_m.shape, dtype=input_m.dtype)
    result[windowSize: -1, :] = cumsum[1: - windowSize, :] - cumsum[windowSize + 1:, :]
    result[-1] = cumsum[-windowSize]
    result[0: windowSize] = np.cumsum(nanFiltered_m[0: windowSize], 0)
    
    result[dataNum == 0] = np.nan
    
    return result

def movingAverage(input_m, windowSize):
    '''
    compute moving average of input_m along column axis, with given windowSize
    skip NaN data, if all NaN data then return NaN value
    '''
    
    dataNum = count_data(input_m, windowSize)
    dataNum = dataNum.astype(float)
    
    result = movingSum(input_m, windowSize)/ dataNum
    result[dataNum == 0] = np.nan
 
    return result

def movingVariance(input_m, windowSize):
    '''
    compute moving variance of input_m along column axis, with given windowSize
    skip NaN data, if all NaN data then return NaN value
    '''
    
    dataNum = count_data(input_m, windowSize)
    dataNum = dataNum.astype(float)
    
    firstMoment = movingSum(input_m, windowSize)/ dataNum
    secondMoment = movingSum(input_m ** 2, windowSize)/ dataNum
    result = secondMoment - firstMoment ** 2
    
    result[dataNum == 0] = np.nan
    result[result<0] = 0.   
    
    return result

def movingProduct(input_m, windowSize):
    '''
    compute moving product of input_m along column axis, with given windowSize
    NaN is treated as 1, but if all data is NaN then return product as NaN
    '''
    
    assert windowSize > 0
    
    dataNum = count_data(input_m, windowSize)
    dataNum = dataNum.astype(float)
    
    nanFiltered_m = np.copy(input_m)
    nanFiltered_m[~np.isfinite(nanFiltered_m)] = 1.0
    result = np.zeros(input_m.shape, dtype=input_m.dtype) * np.nan
    result[:windowSize, :] = np.cumprod(nanFiltered_m[:windowSize, :], 0)

    for di in range(windowSize, result.shape[0]):
        result[di, :] = np.nanprod(nanFiltered_m[di - windowSize + 1: di + 1, :], axis=0)
    result[dataNum == 0] = np.nan

    return result
    
def movingRoot(input_m, windowSize):
    '''
    compute moving root of input_m along column axis, with given windowSize
    skip NaN data, if all NaN data then return NaN value
    '''
    
    dataNum = count_data(input_m, windowSize)
    dataNum = dataNum.astype(float)
    
    result = np.power(movingProduct(input_m, windowSize), 1.0/dataNum )
    result[dataNum == 0] = np.nan

    return result

def movingWeightedSum(input_m, windowSize, weight_v):
    '''
    compute weighted moving sum of input_m along column axis
    with given windowSize and weight for each point in windowSize given by weight_v
    NaN is treated as 0, but if all data is NaN then return sum as NaN
    '''
    
    result = np.full_like(input_m, np.nan)
    nanFiltered_m = np.copy(input_m)
    nanFiltered_m[~np.isfinite(nanFiltered_m)] = 0.0
    
    dataNum = count_data(input_m, windowSize)
    dataNum = dataNum.astype(float)
    
    weight_m = matlib.repmat(weight_v, input_m.shape[1], 1).T 
    
    for i in range(input_m.shape[0]):
        if i < windowSize:
            result[i] = np.sum(nanFiltered_m[:i+1] * weight_m[::-1][:i+1][::-1], axis = 0)
        else:
            result[i] = np.sum(nanFiltered_m[i - windowSize+1 : i+1] * weight_m, axis = 0)
    
    result[dataNum == 0] = np.nan
    
    return result

def movingWeightedAverage(input_m, windowSize, weight_v):
    '''
    compute weighted moving average of input_m along column axis
    with given windowSize and weight for each point in windowSize given by weight_v
    skip NaN data, if all NaN data then return NaN value
    '''
    
    dataNum = count_data(input_m, windowSize)
    dataNum = dataNum.astype(float)
    
    weight_m = np.matlib.repmat(weight_v, input_m.shape[1], 1).T 
    weight_sum_m = np.full_like(input_m, np.nan)
    
    for i in range(input_m.shape[0]):
        if i < windowSize:
            target_weight_m = weight_m[::-1][:i+1][::-1]
            target_value_m = input_m[:i+1]
            target_weight_m[~np.isfinite(target_value_m)] = 0.
            weight_sum_m[i] = np.sum(target_weight_m, axis = 0)
        else:
            target_weight_m = weight_m.copy()
            target_value_m = input_m[i - windowSize+1 : i+1]
            target_weight_m[~np.isfinite(target_value_m)] = 0.
            weight_sum_m[i] = np.sum(target_weight_m, axis = 0)
    weight_sum_m[dataNum == 0] = np.nan
    
    result = movingWeightedSum(input_m, windowSize, weight_v)/ weight_sum_m
    result[dataNum == 0] = np.nan
    
    return result

def delay(input_m, period=1):
    '''
    delay the data by a given period
    '''
    
    result = np.copy( input_m )
    result = np.pad(result, [(period, 0), (0, 0)], mode='edge')[:-period]
    
    return result

def delta(X_m, d, fillna=False):
    '''
    Returns difference between value at day T and T-d of di*ii matrix X
    '''

    if fillna:
        x = np.nan_to_num(X_m)
    else:
        x = X_m.copy()

    result = x[d:] - x[:-d]

    result = np.pad(result, ((d,0), (0,0)), mode='edge')

    return result

def ts_min(X_m, d, fillna=False):
    '''
    Time series min, i.e min value in the past d days of di*ii matrix X
    '''
    if fillna:
        x_m = np.nan_to_num(X_m)
    else:
        x_m = X_m.copy()

    result = np.zeros(x_m.shape, dtype=np.float)

    for di in range(result.shape[0]):
        if di >= d:
            result[di,:] = np.nanmin(x_m[di-d+1:di+1,:], axis=0)
        else:
            result[di,:] = np.nanmin(x_m[:di+1,:], axis=0)

    return result

def ts_max(X_m, d, fillna=False):
    '''
    # Time series max, i.e min value in the past d days of di*ii matrix X
    '''
    if fillna:
        x_m = np.nan_to_num(X_m)
    else:
        x_m = X_m.copy()

    result = np.zeros(x_m.shape, dtype=np.float)

    for di in range(result.shape[0]):
        if di >= d:
            result[di,:] = np.nanmax(x_m[di-d+1:di+1,:], axis=0)
        else:
            result[di,:] = np.nanmax(x_m[:di+1,:], axis=0)

    return result


def movingCorr(X_m, Y_m, d=60, fillna=False):
    '''
    Compute moving correlation between corresponding columns in X_m and Y_m, with a window size of d
    '''
    if fillna:
        x_m = np.nan_to_num(X_m)
        y_m = np.nan_to_num(Y_m)
    else:
        x_m = X_m.copy()
        y_m = Y_m.copy()
        
    mask = np.isfinite(x_m) & np.isfinite(y_m)
    x_m[~mask] = np.nan
    y_m[~mask] = np.nan
    
    x_mov_sum = movingSum(x_m, d)
    y_mov_sum = movingSum(y_m, d)

    x_mov_sum_sq = movingSum(x_m**2, d)
    y_mov_sum_sq = movingSum(y_m**2, d)

    mov_sum_xy = movingSum(x_m*y_m, d)

    dataNum = count_data(x_m, d)
    dataNum = dataNum.astype(float)
    
    result = (dataNum*mov_sum_xy - (x_mov_sum * y_mov_sum))/np.sqrt((dataNum*x_mov_sum_sq-x_mov_sum**2)*(dataNum*y_mov_sum_sq-y_mov_sum**2))
    result[dataNum == 0] = np.nan
    
    return result

def ts_rank(input_m, d = 5, fillna=True):
    if fillna:
        input_m = np.nan_to_num(input_m)
    result = np.copy(input_m)
    for i in range(d, input_m.shape[0]+1):
        df = pd.DataFrame(input_m[i-d:i])
        temp = np.array(df.rank(axis=0, method='average'))
        temp = temp/ np.amax(temp, axis=0)
        result[i-1,:] = temp[d-1,:]
    result[:d-1] = np.nan 
    return result
    
def ts_zscore(input_m, d = 5, fillna=True):
    if fillna:
        input_m = np.nan_to_num(input_m)
    result = np.copy(input_m)
    for i in range(d, input_m.shape[0]+1):
        x = input_m[i-d:i]
        mean = np.mean(x, axis=0)
        sd = np.std(x, axis=0)
        x  = (x - mean)/ sd
        result[i-1,:] = x[d-1,:]
    result[:d-1] = np.nan 
    return result

