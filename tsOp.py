import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import nanOp
from numpy import *
import statsmodels.tsa.stattools as tss
import numba
import datetime
from datetime import date
from numpy import matlib as mb

@numba.jit(nopython=True, nogil=True)
def ffill_ti(input_t):
    # ffill along ti axis of minute bar price
    input_t = input_t.copy()

    for ti in range(1, input_t.shape[1]):
        for di in range(input_t.shape[0]):
            mask_v = ~np.isfinite(input_t[di, ti, :])

            for ii, mask in enumerate(mask_v):
                if mask:
                    input_t[di, ti, ii] = input_t[di, ti - 1, ii]

    return input_t

def delta(X, d, fillna=False):
    '''
    Returns difference between value at day T and T-d,
    :param X:
    :param d:
    :param fillna:
    :return:
    '''

    if fillna:
        x = np.nan_to_num(X)
    else:
        x = X

    result = x[d:] - x[:-d]

    result = np.pad(result, ((d,0), (0,0)), mode='edge')

    return result

#def delta_r(X, d, fillna=True):
#    '''
#    Returns difference between value at day T and T-d,
#    :param X:
#    :param d:
#    :param fillna:
#    :return:
#    '''
#    
#    x = X
#    if fillna:
#        x[~np.isfinite(x)] = 1
#
#    result = x[d:] / x[:-d]
#
#    result = np.pad(result, ((d,0), (0,0)), mode='edge')
#
#    return result

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

def corr(X_m, Y_m, d=5, fillna=True):
    '''
    Compute correlation between corresponding columns in X_m and Y_m, with a window size of d

    :param X_v: Vector X
    :param Y_v: Vector Y
    :param d: Past d days to look back for computation
    :return: Correlation coefficient r
    '''
    if fillna:
        x_m = np.nan_to_num(X_m)
        y_m = np.nan_to_num(Y_m)
    else:
        x_m = X_m
        y_m = Y_m

    x_cumsum = np.cumsum(x_m, axis=0)
    y_cumsum = np.cumsum(y_m, axis=0)
    x_mov_sum = x_cumsum[d-1:] - np.pad(x_cumsum[:-d], ((1,0),(0,0)), 'constant')
    y_mov_sum = y_cumsum[d-1:] - np.pad(y_cumsum[:-d], ((1,0),(0,0)), 'constant')

    x_sq_cumsum = np.cumsum(x_m**2, axis=0)
    y_sq_cumsum = np.cumsum(y_m**2, axis=0)

    x_mov_sum_sq = x_sq_cumsum[d-1:] - np.pad(x_sq_cumsum[:-d], ((1,0),(0,0)), 'constant')
    y_mov_sum_sq = y_sq_cumsum[d-1:] - np.pad(y_sq_cumsum[:-d], ((1,0),(0,0)), 'constant')

    prod_xy = x_m * y_m
    prod_xy_cumsum = np.cumsum(prod_xy, axis=0)

    mov_sum_xy = prod_xy_cumsum[d-1:] - np.pad(prod_xy_cumsum[:-d], ((1,0),(0,0)), 'constant')

    result = (d*mov_sum_xy - (x_mov_sum * y_mov_sum))/np.sqrt((d*x_mov_sum_sq-x_mov_sum**2)*(d*y_mov_sum_sq-y_mov_sum**2))

    return np.pad(result, ((d-1,0),(0,0)), 'edge')

def corr_sp(X_m, Y_m, d=5, fillna=True):
    '''
    Compute spearman correlation between corresponding columns in X_m and Y_m, with a window size of d

    :param X_v: Vector X
    :param Y_v: Vector Y
    :param d: Past d days to look back for computation
    :return: Correlation coefficient r
    '''
    if fillna:
        x_m = np.nan_to_num(X_m)
        y_m = np.nan_to_num(Y_m)
    else:
        x_m = X_m
        y_m = Y_m
    
    result = np.zeros(X_m.shape)
    for i in range(d, result.shape[0]+1):
        x = pd.DataFrame(x_m[i-d:i])
        x = np.array(x.rank(axis=0, method='average'))
        y = pd.DataFrame(y_m[i-d:i])
        y = np.array(y.rank(axis=0, method='average'))
        result[i-1,:] = corr(x, y, d)[-1]
    result[:d-1] = np.nan 
    
    return result 

def ts_min(X_m, d, fillna=False):
    '''

    Time series min, i.e min value in the past d days
    '''
    if fillna:
        x_m = np.nan_to_num(X_m)
    else:
        x_m = X_m

    result = np.zeros(x_m.shape, dtype=np.float)

    for di in range(d-1, result.shape[0]):
        result[di,:] = np.nanmin(x_m[di-d+1:di+1,:], axis=0)

    #result= np.pad(result, ((d-1, 0),(0,0)), 'edge')

    return result

def ts_max(X_m, d, fillna=False):
    '''

    # Time series max, i.e min value in the past d days
    '''
    if fillna:
        x_m = np.nan_to_num(X_m)
    else:
        x_m = X_m

    result = np.zeros(x_m.shape, dtype=np.float)

    for di in range(d-1, result.shape[0]):
        result[di,:] = np.nanmax(x_m[di-d+1:di+1,:], axis=0)

    #result = np.pad(result, ((d - 1, 0), (0, 0)), 'edge')

    return result

### support function for ts_n_min and max 

def v_n_min(input_v, n):
    v = np.copy(input_v)
    a = np.sort(v)
    result = a[n-1]
    if ~np.isfinite(result):
        result = np.nanmax(v)
    return result

def v_n_max(input_v, n):
    v = np.copy(input_v)
    v[~np.isfinite(v)] = np.nanmin(v)
    a = np.sort(v)[::-1]
    result = a[n-1]
    return result

def m_n_min(input_m, n):
    result_v = input_m[-1]*np.nan
    for i in range(len(result_v)):
        result_v[i] = v_n_min(input_m[:,i], n)
    return result_v

def m_n_max(input_m, n):
    result_v = input_m[-1]*np.nan
    for i in range(len(result_v)):
        result_v[i] = v_n_max(input_m[:,i], n)
    return result_v

def ts_n_min(X_m, n=5, d=20, fillna=False):
    '''
    
    # time series n-th max value in the past d days
    '''
    assert d >= n
    
    if fillna:
        x_m = np.nan_to_num(X_m)
    else:
        x_m = X_m
        
    result_m = np.full_like(x_m, np.nan)
    
    for di in range(d-1, result_m.shape[0]):
        temp = x_m[di-d+1:di+1,:]
        result_m[di,:] = m_n_min(temp, n)
    
    return result_m
    
def ts_n_max(X_m, n=5, d=20, fillna=False):
    '''
    
    # time series n-th max value in the past d days
    '''
    assert d >= n
    
    if fillna:
        x_m = np.nan_to_num(X_m)
    else:
        x_m = X_m
        
    result_m = np.full_like(x_m, np.nan)
    
    for di in range(d-1, result_m.shape[0]):
        temp = x_m[di-d+1:di+1,:]
        result_m[di,:] = m_n_max(temp, n)
    
    return result_m
    
def lott_min(X_m, n = 5, d = 20, fillna = False):
    
    assert d >= n
    
    if fillna:
        x_m = np.nan_to_num(X_m)
    else:
        x_m = X_m
    
    result_m = np.zeros(x_m.shape)
    for i in range(1,n+1):
        result_m = result_m + ts_n_min(x_m, i, d)
    result_m = result_m/ n
    
    return result_m

def lott_max(X_m, n = 5, d = 20, fillna = False):
    
    assert d >= n
    
    if fillna:
        x_m = np.nan_to_num(X_m)
    else:
        x_m = X_m
    
    result_m = np.zeros(x_m.shape)
    for i in range(1,n+1):
        result_m = result_m + ts_n_max(x_m, i, d)
    result_m = result_m/ n
    
    return result_m
    
def consecHigh(x):
    
    result = np.zeros(x.shape)
    nanFiltered_x = np.copy(x)
    nanFiltered_x[~np.isfinite(nanFiltered_x)] = 0.0
    for di in range(1,x.shape[0]):
        up = x[di] > x[di-1]
        result[di,up] = result[di-1,up] + 1
        result[di,~up] = 0
    
    return result

def consecLow(x):
    
    result = np.zeros(x.shape)
    nanFiltered_x = np.copy(x)
    nanFiltered_x[~np.isfinite(nanFiltered_x)] = 0.0
    for di in range(1,x.shape[0]):
        down = x[di] < x[di-1]
        result[di,down] = result[di-1,down] - 1
        result[di,~down] = 0
    
    return result

def consecTrend(x):
    
    result = consecHigh(x) + consecLow(x)
    
    return result
    
def fastMProd(input_m, windowSize):
    # ignore NaN value (put to zero)
    
    assert windowSize > 0
    nanFiltered_m = np.copy(input_m)
    nanFiltered_m[~np.isfinite(nanFiltered_m)] = 1.0
    cumprod = np.cumprod(nanFiltered_m[::-1], 0)[::-1]
    result = np.ndarray(input_m.shape, dtype=input_m.dtype)
    result[windowSize: -1, :] = cumprod[1: - windowSize, :] / cumprod[windowSize + 1:, :]
    result[-1] = cumprod[-windowSize]
    result[0: windowSize] = np.cumprod(nanFiltered_m[0: windowSize], 0)
    
    return result
   
def movingSum(input_m, windowSize):
    # ignore NaN value (put to zero)
    
    assert windowSize > 0
    nanFiltered_m = np.copy(input_m)
    nanFiltered_m[~np.isfinite(nanFiltered_m)] = 0.0
    result = np.zeros(input_m.shape, dtype=input_m.dtype)
    result[:windowSize, :] = np.cumsum(nanFiltered_m[:windowSize, :], 0)

    for di in range(windowSize, result.shape[0]):
        result[di, :] = np.nansum(nanFiltered_m[di - windowSize + 1: di + 1, :], axis=0)

    return result

def movingMean(input_m, N=5):
    cumsum, moving_aves = [0], []
    
    for i, x in enumerate(input_m):
        cumsum.append(cumsum[i-1]+x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            moving_aves.append(moving_ave)  
    
    moving_aves = np.array(moving_aves)
                      
    return moving_aves

def fastMSum(input_m, windowSize):
    # ignore NaN value (put to zero)
    
    assert windowSize > 0
    nanFiltered_m = np.copy(input_m)
    nanFiltered_m[~np.isfinite(nanFiltered_m)] = 0.0
    cumsum = np.cumsum(nanFiltered_m[::-1], 0)[::-1]
    result = np.ndarray(input_m.shape, dtype=input_m.dtype)
    result[windowSize: -1, :] = cumsum[1: - windowSize, :] - cumsum[windowSize + 1:, :]
    result[-1] = cumsum[-windowSize]
    result[0: windowSize] = np.cumsum(nanFiltered_m[0: windowSize], 0)
    
    return result

def movingAverage_1(input_m, windowSize=5):
    # ignore Nan value (put to zero)
    result = np.copy(input_m)
    result[:,:] = np.nan
    
    for i in range(windowSize-1, input_m.shape[0]):
        result[i] = np.nanmean(input_m[i - windowSize+1: i+1], axis=0)
 
    return result

def movingAverage(input_m, windowSize=5, nanIsZero= False, useFastMSum=True):
    dataNumShape = [input_m.shape[0], 1] if len(input_m.shape) == 2 else [input_m.shape[0]]
    dataNum = np.array([min(di + 1, windowSize) for di in range(input_m.shape[0])], dtype=input_m.dtype).reshape(
        dataNumShape)
    if (not nanIsZero):
        dataNum = mb.repmat(dataNum,1,input_m.shape[1]) - nanOp.countNanInWindow(input_m, windowSize)
        dataNum1 = np.copy(dataNum)
        dataNum[dataNum == 0] = 1.
        
    dataNum = dataNum.astype(float)
    
    if useFastMSum:
        result = fastMSum(input_m, windowSize) / dataNum
    else:
        result = movingSum(input_m, windowSize) / dataNum
    
    if (not nanIsZero):
        result[dataNum1 == 0] = np.nan 

    return result

def ts_coverage(input_m, d = 250):
    result = np.copy(input_m)
    for i in range(d-1, input_m.shape[0]):
        result[i] = 1. * np.sum(np.isfinite(input_m[i-d+1:i+1]),axis=0)/ d
    result[:d] = np.nan
    return result

def ts_count(input_m, d = 250, x=0.):
    result = np.copy(input_m)
    for i in range(d-1, input_m.shape[0]):
        result[i] = np.sum(input_m[i-d+1:i+1]==x, axis=0)
    result[:d] = np.nan
    return result

def movingRoot(input_m, windowSize=5, nanIsOne = False):
    dataNumShape = [input_m.shape[0], 1] if len(input_m.shape) == 2 else [input_m.shape[0]]
    dataNum = np.array([min(di + 1, windowSize) for di in range(input_m.shape[0])], dtype=input_m.dtype).reshape(
        dataNumShape)
    if (not nanIsOne):
        dataNum = mb.repmat(dataNum,1,input_m.shape[1]) - nanOp.countNanInWindow(input_m, windowSize)
        dataNum1 = np.copy(dataNum)
        dataNum[dataNum == 0] = 1.
        
    dataNum = dataNum.astype(float)
    
    result = np.power(fastMProd(input_m, windowSize), 1.0/dataNum )
    
    result[dataNum1 == 0] = np.nan 

    return result

def linWeightedMovingSum(x, windowSize):
    
    result = np.zeros(x.shape)
    nanFiltered_x = np.copy(x)
    nanFiltered_x[np.isnan(nanFiltered_x)] = 0.0
    w = np.arange(windowSize) + 1
    w = mb.repmat(w, x.shape[1], 1).T 
    
    for i in range(x.shape[0]):
        if i < windowSize:
            result[i] = np.sum(nanFiltered_x[:i+1] * w[::-1][:i+1][::-1], axis = 0)
        else:
            result[i] = np.sum(nanFiltered_x[i - windowSize+1 : i+1] * w, axis = 0)
    
    return result
    
def linWeightedMovingAverage(x, windowSize, nanIsZero = True):
    
    result = np.zeros(x.shape)
    dataNum = np.zeros(x.shape)
    
    for i in range(dataNum.shape[0]):
        if i < windowSize:
            dataNum[i,:] = (windowSize * (windowSize + 1) - (windowSize - i - 1) * (windowSize - i))/2.0 
        else:
            dataNum[i,:] = windowSize * (windowSize + 1) / 2.0
    
    s = linWeightedMovingSum(x, windowSize)
    result = s / dataNum
    
    if (not nanIsZero):
        w = np.arange(windowSize)+1 
        w = mb.repmat(w,x.shape[1],1).T
        for di in range(x.shape[0]):
            wd = np.copy(w)
            if di < windowSize:
                wd = wd[::-1][:di+1][::-1]                
                wd[~np.isfinite(x[:di+1,:])] = 0
                result[di] = s[di] / np.sum(wd,axis=0)
            else:
                wd[~np.isfinite(x[di+1-windowSize:di+1,:])] = 0
                result[di] = s[di] / np.sum(wd,axis=0)
    
    return result

#def weightedMovingAverage(x, windowSize, nanIsZero = False):
#    
#    result = np.zeros(x.shape)
#    nanFiltered_x = np.copy(x)
#    nanFiltered_x[np.isnan(nanFiltered_x)] = 0.0
#    decay_v = np.arange(windowSize)+1
#    decay_v = mb.repmat(decay_v,x.shape[1],1).T
#    for di in range(x.shape[0]):
#        if di < windowSize:
#            d = np.copy(decay_v[::-1][:di+1][::-1])
#            if nanIsZero:
#                result[di] = np.sum(nanFiltered_x[:di+1]*d, axis=0)/ np.sum(d,axis=0)
#            else:
#                d[~np.isfinite(x[:di+1,:])] = 0
#                result[di] = np.sum(nanFiltered_x[:di+1]*d, axis=0)/ np.sum(d,axis=0)
#        else:
#            if nanIsZero:
#                result[di] = np.sum(nanFiltered_x[di+1-windowSize:di+1]*decay_v, axis=0)/ np.sum(decay_v, axis=0)
#            else:
#                d = np.copy(decay_v)
#                d[~np.isfinite(x[di+1-windowSize:di+1,:])] = 0
#                result[di] = np.sum(nanFiltered_x[di+1-windowSize:di+1]*d, axis=0)/ np.sum(d, axis=0)
#    
#    result[~np.isfinite(result)] = np.nan
#    
#    return result

def expWeightedMovingSum(x, decay_factor, windowSize = None, threshold = 0.01):
    
    result = np.zeros(x.shape)
    nanFiltered_x = np.copy(x)
    nanFiltered_x[np.isnan(nanFiltered_x)] = 0.0
    result[0] = nanFiltered_x[0]
    
    assert (windowSize is not None) | (threshold is not None) 
    
    if windowSize is None:
        windowSize = np.ceil(np.log(threshold)/ np.log(decay_factor))
    
    windowSize = np.int(windowSize)
    
    for i in range(1,windowSize):
        result[i] = result[i-1] * decay_factor + nanFiltered_x[i]
    
    for i in range(windowSize,x.shape[0]):
        result[i] = result[i-1] * decay_factor + nanFiltered_x[i] - np.power(decay_factor, windowSize) * nanFiltered_x[i - windowSize]
            
    return result

def expWeightedMovingAverage(x, decay_factor = None, windowSize = None, threshold = 0.01, nanIsZero = False):
    
    result = np.zeros(x.shape)
    assert (threshold is not None) | (windowSize is not None)  # cannot be both None
    
    if windowSize is None:
        windowSize = np.ceil(np.log(threshold)/ np.log(decay_factor))
    
    windowSize = np.int(windowSize)
    
    dataNum = np.zeros(x.shape)
    
    for i in range(dataNum.shape[0]):
        if i < windowSize:
            dataNum[i,:] = (1 - np.power(decay_factor, i+1))/ (1 - decay_factor)
        else:
            dataNum[i,:] = (1 - np.power(decay_factor, windowSize))/ (1 - decay_factor)
    
    s = expWeightedMovingSum(x, decay_factor, windowSize, threshold)
    result = s / dataNum
    
    if (not nanIsZero):
        w = np.ones(windowSize)
        for i in range(windowSize - 2, -1, -1):
            w[i] = w[i+1]* decay_factor
        w = mb.repmat(w,x.shape[1],1).T
        for di in range(x.shape[0]):
            wd = np.copy(w)
            if di < windowSize:
                wd = wd[::-1][:di+1][::-1]                
                wd[~np.isfinite(x[:di+1,:])] = 0
                result[di] = s[di] / np.sum(wd,axis=0)
            else:
                wd[~np.isfinite(x[di+1-windowSize:di+1,:])] = 0
                result[di] = s[di] / np.sum(wd,axis=0)
    
    return result

def movingVariance(input_m, windowSize=5, nanIsZero=False):
    dataNumShape = [input_m.shape[0], 1] if len(input_m.shape) == 2 else [input_m.shape[0]]
    dataNum = np.array([min(di + 1, windowSize) for di in range(input_m.shape[0])], dtype=input_m.dtype).reshape(
        dataNumShape)
    if (not nanIsZero):
        dataNum = dataNum - nanOp.countNanInWindow(input_m, windowSize)
        dataNum1 = np.copy(dataNum)
        dataNum[dataNum == 0] = 1.0
    
    dataNum = dataNum.astype(float)
    
    firstMoment = fastMSum(input_m, windowSize) / dataNum
    secondMoment = fastMSum(input_m ** 2, windowSize) / dataNum
    result = secondMoment - firstMoment ** 2
    
    result[dataNum1 == 0] = np.nan  
    result[result<0] = 0.   
    
    return result

def ts_sum(X_m, d, fillna=True):

    if fillna:
        x_m = np.nan_to_num(X_m)
    else:
        x_m = X_m

    x_csum = np.cumsum(x_m, axis=0)

    result = x_csum[d-1:] - np.pad(x_csum[:-d], ((1,0),(0,0)), 'constant')

    result = np.pad(result, ((d-1,0),(0,0)), 'edge')

    return result

def ts_count_isfinite(X, d):
    isfinite_m = np.isfinite(X)
    result_m = ts_sum(isfinite_m, d)
    return result_m
    
def ts_mean(X, d, fillna=False):
    '''
    Mean of time-series over past d days
    '''
    if fillna:
        return ts_sum(X, d)/np.float(d)
    else:
        return ts_sum(X, d)/ts_count_isfinite(X, d)
        

def ts_var(X, d, fillna=False):
    '''
    variance of time-series over past d days
    '''
    if fillna:
        return ts_sum(X**2, d)/np.float(d) - (ts_sum(X, d)/np.float(d))**2
    else:
        return ts_sum(X**2, d)/ts_count_isfinite(X, d) - (ts_sum(X, d)/ts_count_isfinite(X, d))**2

def ts_std(X, d, fillna=False):
    '''
    std deviation over past d days

    '''
    return np.sqrt(ts_var(X, d, fillna))

#def ts_Zscore(X, d, fillna = True):
    
   #return (X - ts_mean(X, d, fillna))/ np.power(ts_var(X, d, fillna), 0.5)

#def ts_rank(X_m, fillna=True, normalize = True):
#    '''
#
#    '''
#    if fillna:
#        x_m = np.nan_to_num(X_m)
#    else:
#        x_m = X_m
#
#    result = np.empty(X_m.shape)
#
#    for di in range(result.shape[1]):
#        result[:, di] = stats.rankdata(x_m[:, di], method='average')
#    
#    mx = np.nanmax(result, axis=0)
#    mx = mb.repmat(mx, X_m.shape[0], 1)
#    if normalize == True:
#        result = result/ mx
#    
#    return result
#
#def ts_rank1(X_m, d, fillna=True):
#    '''
#
#    '''
#    if fillna:
#        x_m = np.nan_to_num(X_m)
#    else:
#        x_m = X_m
#
#    result = np.empty((X_m.shape[0]-d, X_m.shape[1]))
#
#    x_csum = np.cumsum(x_m, axis=0)
#    x_sma = (x_csum[d-1:] - np.pad(x_csum[:-d], ((1,0),(0,0)), 'constant')) / d
#
#    for di in range(result.shape[0]):
#        result[di,:] = stats.rankdata(x_sma[di,:], method='average')
#
#    result = np.pad(result, ((d,0),(0,0)), 'edge')
#
#    return result

def delay(input_m, period=1):
    result = np.copy( input_m )
    # result[ period:, : ] = result[ :-period, : ]
    result = np.pad(result, [(period, 0), (0, 0)], mode='edge')[:-period]
    
    return result


#def expMovingAverage(input_m, windowSize):
#    
#    assert windowSize > 0
#    
#    dataNumShape = [input_m.shape[0], 1] if len(input_m.shape) == 2 else [input_m.shape[0]]
#    dataNum = np.array([min(di + 1, windowSize) for di in range(input_m.shape[0])], dtype=input_m.dtype).reshape(
#        dataNumShape)
#    dataNum = mb.repmat(dataNum,1,input_m.shape[1]) - nanOp.countNanInWindow(input_m, windowSize)
#    dataNum1 = np.copy(dataNum)
#    dataNum[dataNum == 0] = 1.
#    
#    nanFiltered_m = np.copy(input_m)
#    nanFiltered_m[np.isnan(nanFiltered_m)] = 0.0
#    
#    result = np.zeros(input_m.shape)
#    alpha = 2./(windowSize + 1)
#    result[:windowSize] = np.cumsum(nanFiltered_m[:windowSize], axis = 0) / dataNum[:windowSize]
#    
#    for di in range(windowSize, input_m.shape[0]):
#        result[di] = alpha * nanFiltered_m[di] + (1-alpha) * result[di-1]
#    
#    result[dataNum1 == 0] = np.nan         
#    
#    return result

#def ewma(x, factor, ffill = False):
#    
#    if ffill:
#        x = nanOp.fillNan(x)
#        
#    ewma = np.zeros(x.shape)
#    nanFiltered_m = np.copy(x)
#    nanFiltered_m[np.isnan(nanFiltered_m)] = 0.0
#    ewma[0] = nanFiltered_m[0]
#    
#    for i in range(1,x.shape[0]):
#        ewma[i] = (1 - factor) * ewma[i - 1] + factor * nanFiltered_m[i]
#        
#    return ewma
#
#def ewmv(x, factor, alpha, ffill = False):
#    
#    if ffill:
#        x = nanOp.fillNan(x)
#    
#    mean = np.zeros(x.shape)
#    mean = ewma(x, factor, ffill) 
#    ewmv = np.zeros(x.shape)
#    nanFiltered_m = np.copy(x)
#    nanFiltered_m[np.isnan(nanFiltered_m)] = 0.0
#    ewmv[0] = 0.0
#    
#    for i in range(1,x.shape[0]):
#        ewmv[i] = alpha * ewmv[i-1] + (1 - alpha) * ((nanFiltered_m[i] - mean[i-1] )**2)
#        
#    return ewmv    


def pw_corrcoef(x, y, axis = 0):
 
    n = x.shape[0]
    r = (n * np.sum(x*y,axis = axis) - np.sum(x, axis=axis) * np.sum(y, axis=axis))/ np.sqrt((n * np.sum(x**2, axis=axis) - (np.sum(x, axis=axis))**2 ) * (n * np.sum(y**2, axis=axis) - (np.sum(y, axis=axis))**2 ) )
    return r

def movingCorr1(x, y, axis = 0, windowSize = 5):
    
    assert x.shape == y.shape
    result = np.zeros(x.shape)
    for di in range(x.shape[0]):
        if di < windowSize:
            result[di] = pw_corrcoef(x[:di+1],y[:di+1],axis=axis)
        else:
            result[di] = pw_corrcoef(x[di+1-windowSize:di+1],y[di+1-windowSize:di+1],axis=axis)
            
    return result


def movingSCorr(x, y, windowSize = 20):
    # calculate the moving Spearman correlation coefficient of x and y column vectors
    assert x.shape == y.shape
    result = np.zeros(x.shape)
    for di in range(len(x)):
        if di < windowSize:
            for ii in range(x.shape[1]):
                result[di, ii] = stats.spearmanr(x[:di+1, ii],y[:di+1, ii])[0]
        else:
            for ii in range(x.shape[1]):
                result[di, ii] = stats.spearmanr(x[di+1-windowSize:di+1, ii],y[di+1-windowSize:di+1, ii])[0]
    return result

def movingSkew(x, windowSize = 20):
    
    result = np.zeros(x.shape)
    for di in range(len(x)):
        if di < windowSize:
            result[di] = stats.skew(x[:di+1])
        else:
            result[di] = stats.skew(x[di+1-windowSize:di+1])
    return result

def movingKurt(x, windowSize = 20):
    
    result = np.zeros(x.shape)
    for di in range(len(x)):
        if di < windowSize:
            result[di] = stats.kurtosis(x[:di+1])
        else:
            result[di] = stats.kurtosis(x[di+1-windowSize:di+1])
    return result

def movingCovar(x, y, windowSize = 20):
    # calculate the moving covariance of x and y column-wise
    assert x.shape == y.shape
    result = np.zeros(x.shape)
    for di in range(x.shape[0]):
        if di < windowSize:
            for ii in range(x.shape[1]):
                result[di,ii] = np.cov(x[:di+1, ii],y[:di+1, ii])[0,1]
        else:
            for ii in range(x.shape[1]):
                result[di,ii] = np.cov(x[di+1-windowSize:di+1, ii],y[di+1-windowSize:di+1, ii])[0,1]
    return result                                 

def chn_norm(X_m, windowSize=5, eps = 0.0001):
    '''

    Channel normalization, i.e equivalent to a high pass filter with detrending
    '''
    x_m = np.nan_to_num(X_m)

    min_m = ts_min(x_m, windowSize)
    max_m = ts_max(x_m, windowSize)

    min_m = np.nan_to_num(min_m)
    max_m = np.nan_to_num(max_m)

    return (x_m - min_m) / ((max_m - min_m) + eps)

def convolution(X_m, Y_m, fillna=True):

    return None


#============================== TA =====================================================================================#

def movingMedian(X, d, fillna=True):
    '''
    Time series median across last d days
    '''
    if fillna:
        x = np.nan_to_num(X)
    else:
        x = X

    result = np.empty((X.shape[0]-d+1, X.shape[1]),dtype=np.float32)

    for di in range(X.shape[0]-d+1):
        result[di] = np.median(x[di:di+d-1],axis=0)

    result = np.pad(result, ((d-1,0),(0,0)), 'edge')
    return result

def movingProd(X_m, d, fillna=True):
    '''
    Implements the time series product, which is simply a cumulative product of the past d days

    :param X_m:
    :param d:
    :param fillna:
    :return:
    '''

    if fillna:
        x_m = np.nan_to_num(X_m)
    else:
        x_m = X_m

    x_cprod = np.cumprod(x_m, axis=0)

    result = x_cprod[d-1:] / np.pad(x_cprod[:-d], ((1,0),(0,0)), 'constant', constant_values=1)

    result = np.pad(result, ((d-1,0),(0,0)), 'edge')

    return result

def std_away(input_m, days = 20):
    ma = movingAverage(input_m, days)
    std = np.power(movingVariance(input_m, days), 0.5)
    result = (input_m - ma)/ std
    return result

def rsi(X, days = 14, fillna = False):
    
    if fillna:
        x = np.nan_to_num(X)
    else:
        x = X

    up_m = (x[1:] > x[:-1]) 
    gain = up_m * np.fabs(x[1:] - x[:-1])
    loss = ~up_m * np.fabs(x[1:] - x[:-1])

    ave_gain = movingAverage(gain, days)
    ave_loss = movingAverage(loss, days)
    
    result = ave_gain/(ave_gain + ave_loss) * 100
    result = np.pad(result, [(1, 0), (0, 0)], mode='edge')
    
    return result

def fisher_rsi(X, day1 = 5, day2 = 9, fillna = False):
    
    if fillna:
        x = np.nan_to_num(X)
    else:
        x = X
        
    result = rsi(X, day1, fillna)
    
    result = 0.1 * ( result - 50 )
    result = linWeightedMovingAverage(result, day2)
    result = (np.exp(2*result) - 1)/ (np.exp(2*result) + 1)
    
    return result

def sto_rsi(X, days = 14, fillna = True):
    
    if fillna:
        x = np.nan_to_num(X)
    else:
        x = X
    
    R = rsi(x, days, fillna)
    result = (R - ts_min(R, days, fillna)) / (ts_max(R, days, fillna) - ts_min(R, days, fillna))
    
    return result

def bollinger(input_m, days = 20, factor = 2.0):
    ma = movingAverage(input_m, days)
    std = np.power(movingVariance(input_m, days), 0.5)
    upper_band = ma + factor*std
    lower_band = ma - factor*std
    return upper_band, lower_band

def ema(input_m, ratio = 0.9):
    alpha_v = np.copy(input_m)
    alpha_v = np.nan_to_num(alpha_v)
    for i in range(1,len(alpha_v)):
        v1 = np.nan_to_num(alpha_v[i-1])
        alpha_v[i]=alpha_v[i]*(1-ratio)+alpha_v[i-1]*ratio
        alpha_v[i,np.isnan(input_m[i-1,:])] = input_m[i,np.isnan(input_m[i-1,:])]
        alpha_v[i,np.isnan(input_m[i,:])] = input_m[i,np.isnan(input_m[i,:])]
    alpha_v = np.nan_to_num(alpha_v)
    alpha_v += (alpha_v == 0) * delay(alpha_v,1)
    return alpha_v

def ema1(input_m, ratio_m):
    alpha_v = np.copy(input_m)
    alpha_v = np.nan_to_num(alpha_v)
    for i in range(1,len(alpha_v)):
        v1 = np.nan_to_num(alpha_v[i-1])
        alpha_v[i]=alpha_v[i]*(1-ratio_m[i])+alpha_v[i-1]*ratio_m[i]
        alpha_v[i,np.isnan(input_m[i-1,:])] = input_m[i,np.isnan(input_m[i-1,:])]
        alpha_v[i,np.isnan(input_m[i,:])] = input_m[i,np.isnan(input_m[i,:])]
    alpha_v = np.nan_to_num(alpha_v)
    alpha_v += (alpha_v == 0) * delay(alpha_v,1)
    return alpha_v

def macd(input_m, ratio_fast = 0.1, ratio_slow = 0.4):
    fast = ema(input_m,ratio_fast)
    slow = ema(input_m,ratio_slow)
    return (fast - slow)

def ts_sto_k(close_m, high_m, low_m, days):
    high = ts_max(high_m, days)
    low = ts_min(low_m, days)
    result = 100*(close_m - low)/ (high - low)
    return result

def ts_sto_d(close_m, high_m, low_m, days1, days2):
    result1 = ts_sto_k(close_m, high_m, low_m, days1)
    result2 = ts_mean(result1, days2)
    return result2

def ts_sto(high_m, low_m, close_m, days1 = 14, days2 = 10, days3 = 20):
    fast = ts_mean( ((close_m - low_m)/ (high_m - low_m)), days1)
    slow1 = ts_mean(fast, days2)
    slow2 = ts_mean(fast, days3)
    output=[]
    output.append(slow1)
    output.append(slow2)
    return output

def obv(close_m, volume_m):
    result_m = np.zeros(volume_m.shape)
    result_m[0] = volume_m[0]
    for i in range(1, result_m.shape[0]):
        result_m[i] = result_m[i-1] + volume_m[i] * (close_m[i] > close_m[i-1]) - volume_m[i] * (close_m[i] < close_m[i-1])
    return result_m

def obv_rolling(close_m, volume_m, days = 14):
    close_prev_m = delay(close_m, 1)
    vol1 = volume_m * (close_m > close_prev_m)
    vol2 = -volume_m * (close_m < close_prev_m)
    vol = vol1 + vol2 
    return movingSum(vol, days)

def mfi(close_m, high_m, low_m, volume_m, days = 14):  
    tp_m = 1./3 * (close_m + high_m + low_m)
    mf = tp_m * volume_m
    p_mf = np.zeros(mf.shape)
    n_mf = np.zeros(mf.shape)
    for i in range(1, mf.shape[0]):
        p_mf[i] = p_mf[i] + mf[i] * (tp_m[i] > tp_m[i-1])
        n_mf[i] = n_mf[i] + mf[i] * (tp_m[i] < tp_m[i-1]) 
    p_mf_sum = fastMSum(p_mf, days)
    n_mf_sum = fastMSum(n_mf, days)
    result_m = 100.0 * p_mf_sum/ (p_mf_sum + n_mf_sum)
    return result_m

def adl(close_m, high_m, low_m, volume_m):
    clv_m = ((close_m - low_m) - (high_m - close_m))/ (high_m - low_m)
    mfv_m = clv_m * volume_m
    result_m = np.cumsum(mfv_m, axis=0)
    return result_m

def adl_rolling(close_m, high_m, low_m, volume_m, days = 14):
    clv_m = ((close_m - low_m) - (high_m - close_m))/ (high_m - low_m)
    mfv_m = clv_m * volume_m
    result_m = movingSum(mfv_m, days)
    return result_m

def cmf(close_m, high_m, low_m, volume_m, days = 20):
    clv_m = ((close_m - low_m) - (high_m - close_m))/ (high_m - low_m)
    mfv_m = clv_m * volume_m
    result_m = fastMSum(mfv_m, days)/ fastMSum(volume_m, days) 
    return result_m

def pvt(close_m, volume_m):
    result_m = np.zeros(volume_m.shape)
    result_m[0] = 0
    close_prev = delay(close_m, 1)
    for i in range(1, result_m.shape[0]):
        result_m[i] = (close_m[i] - close_prev[i])/ close_m[i] * volume_m[i]
    return result_m

def pvt_rolling(return_m, volume_m, days = 14):
    return movingSum(return_m * volume_m, days)

def mfv(high, low, close, volume):
    mfv = ((close - low) - (high - close))/ (high - low) * volume 
    return mfv

def obv1(return_m, volume_m):
    result = np.sign(return_m) * volume_m
    return result

def demark(high_m, low_m, d=5):
    high_prev = delay(high_m, 1)
    low_prev = delay(low_m, 1)
    maxx = np.maximum(high_m - high_prev, np.zeros(high_m.shape))
    minn = np.maximum(low_prev - low_m, np.zeros(low_m.shape))
    result = movingAverage(maxx, d) / (movingAverage(maxx, d) + movingAverage(minn, d))
    return result

def cmo(X, d, fillna=False):
    if fillna:
        x = np.nan_to_num(X)
    else:
        x = X
    X_d = delay(X, 1)
    pos = (X - X_d) * ((X - X_d) >= 0)
    neg = np.fabs(X - X_d) * ((X - X_d) < 0)
    result = (fastMSum(pos, d) - fastMSum(neg, d))/  (fastMSum(pos, d) + fastMSum(neg, d))
    return result 

def will_r(close_m, high_m, low_m, days = 5):
    result = (close_m - ts_max(high_m, days))/ (ts_max(high_m, days) - ts_min(low_m, days))
    return result

def pvo(volume_m, ratio_fast = 0.1, ratio_slow = 0.4):
    result = (ema(volume_m, ratio_fast) - ema(volume_m, ratio_slow))/ ema(volume_m, ratio_slow)
    return result

def vrc(volume_m, days = 14):
    result = volume_m/ delay(volume_m, days) - 1
    return result

def cci(high, low, close, d=20):
    high = np.nan_to_num(high)
    low = np.nan_to_num(low)
    close = np.nan_to_num(close)
    tp = 1./3*(high + low + close)
    ma = movingAverage(tp, d)
    std = fastMSum(np.fabs(tp - ma),  d)/ d
    result = (tp - ma)/ (0.15 * std)
    return result

def vma(close, ratio = 0.1, d1=10):
    direction = np.fabs(close - delay(close, d1))
    daily_change = np.fabs(close - delay(close, 1))
    volatility = movingSum(daily_change, d1)
    effciency_ratio = direction/ volatility
    result = ema1(close, ratio * effciency_ratio)
    #result = np.copy(close)
    #for i in range(1, len(result)):
        #result[i] = result[i-1] + SC[i] * (close[i] - result[i-1])
    return result

def kama(close, ratio = 0.1, d1=10, d2=2, d3=30):
    direction = np.fabs(close - delay(close, d1))
    daily_change = np.fabs(close - delay(close, 1))
    volatility = movingSum(daily_change, d1)
    effciency_ratio = direction/ volatility
    fast_SC = 2./ (d2 + 1) * np.ones(ER.shape)
    slow_SC = 2./ (d3 + 1) * np.ones(ER.shape)
    SC = (ER * (fast_SC - slow_SC) + slow_SC)
    SC = np.power(SC, 2)
    result = ema1(close, SC*ratio)
    #result = np.copy(close)
    #for i in range(1, len(result)):
        #result[i] = result[i-1] + SC[i] * (close[i] - result[i-1])
    return result

def divergence_index(close, d1=10, d2=40):
    daily_change = np.fabs(close - delay(close, 1)) 
    result = ((close - delay(close, d1)) * (close - delay(close, d2)))/ movingVariance(daily_change, d2)
    return result

def TR(high, low, close):
    close_prev = delay(close, 1)
    result = np.maximum(high - low, np.fabs(high - close_prev))
    result = np.maximum(result, np.fabs(low - close_prev))
    return result 

def ATR(high, low, close, ratio = 0.1):
    tr = TR(high, low, close)
    result = ema(tr, ratio)
    
    #result = np.copy(TR)
#    result[: days -1] = np.nan
#    result[days - 1] = np.mean(TR[:days], axis=0)
#    for i in range(days, TR.shape[0]):
#        result[i] = (result[i-1]*(days-1) + TR[i])/ day

    return result 

def directional_movement(high, low):
    result_1 = (high - delay(high, 1)) * (high >  delay(high, 1)) * ((high - delay(high, 1)) > (delay(low, 1) - low))
    result_2 = (delay(low, 1) - low) * (delay(low, 1) > low) * ((high - delay(high, 1)) < (delay(low, 1) - low))
    return result_1, result_2

def directional_index(close, high, low, ratio = 0.1):
    atr = ATR(high, low, close, ratio)
    DM_pos, DM_neg = directional_movement(high, low)
    DI_pos = ema(DM_pos/ atr, ratio)
    DI_neg = ema(DM_neg/ atr, ratio)
    DX = np.fabs(DI_pos - DI_neg)/ (DI_pos + DI_neg) * 100.
    ADX = ema(DX, ratio)
    return DI_pos, DI_neg, DX, ADX

def vortex(close, high, low, days = 14):
    vm_pos = np.fabs(high - delay(low, 1))
    vm_neg = np.fabs(low - delay(high, 1))
    vm_pos_sum = movingSum(vm_pos, days)
    vm_neg_sum = movingSum(vm_neg, days)
    tr = TR(high, low, close)
    tr_sum = movingSum(tr, days)
    vi_pos = vm_pos_sum/ tr_sum
    vi_neg = vm_neg_sum/ tr_sum
    return vi_pos, vi_neg

    
#==========================================================================================================================================    

def decay( alpha_m, ratio=0.1):
    alpha_v = np.copy(alpha_m)
    for i in range(1,len(alpha_v)):
        v1 = np.nan_to_num(alpha_v[i-1])
        alpha_v[i]=alpha_v[i]*(1-ratio)+v1*ratio
    return alpha_v

def movingCorr(X_m, Y_m, d=5, fillna=True):
    '''
    Compute correlation between corresponding columns in X_m and Y_m, with a window size of d

    :param X_v: Vector X
    :param Y_v: Vector Y
    :param d: Past d days to look back for computation
    :return: Correlation coefficient r
    '''
    if fillna:
        x_m = np.nan_to_num(X_m)
        y_m = np.nan_to_num(Y_m)
    else:
        x_m = X_m
        y_m = Y_m

    x_cumsum = np.cumsum(x_m, axis=0)
    y_cumsum = np.cumsum(y_m, axis=0)
    x_mov_sum = x_cumsum[d-1:] - np.pad(x_cumsum[:-d], ((1,0),(0,0)), 'constant')
    y_mov_sum = y_cumsum[d-1:] - np.pad(y_cumsum[:-d], ((1,0),(0,0)), 'constant')

    x_sq_cumsum = np.cumsum(x_m**2, axis=0)
    y_sq_cumsum = np.cumsum(y_m**2, axis=0)

    x_mov_sum_sq = x_sq_cumsum[d-1:] - np.pad(x_sq_cumsum[:-d], ((1,0),(0,0)), 'constant')
    y_mov_sum_sq = y_sq_cumsum[d-1:] - np.pad(y_sq_cumsum[:-d], ((1,0),(0,0)), 'constant')

    prod_xy = x_m * y_m
    prod_xy_cumsum = np.cumsum(prod_xy, axis=0)

    mov_sum_xy = prod_xy_cumsum[d-1:] - np.pad(prod_xy_cumsum[:-d], ((1,0),(0,0)), 'constant')

    result = (d*mov_sum_xy - (x_mov_sum * y_mov_sum))/np.sqrt((d*x_mov_sum_sq-x_mov_sum**2)*(d*y_mov_sum_sq-y_mov_sum**2))

    return np.pad(result, ((d-1,0),(0,0)), 'edge')

def movingDelayCorr(X_m, Y_m, lag=1, d=5, fillna=True):
    '''
    Compute delayed correlation between corresponding columns in X_m and Y_m, with a window size of d

    :param X_v: Vector X
    :param Y_v: Vector Y
    :param d: Past d days to look back for computation
    :return: Correlation coefficient r
    '''
    if fillna:
        x_m = np.nan_to_num(X_m)
        y_m = np.nan_to_num(Y_m)
    else:
        x_m = X_m
        y_m = Y_m
    
    x_m = delay(x_m, period=lag)
    
    result = movingCorr(x_m, y_m, d=d) # x lead y
    
    return result

def movingAutoCorr(X_m, lag=1, d=5, fillna=True):
    
    result = movingDelayCorr(X_m, X_m, lag=lag, d=d, fillna=fillna)
    
    return result

def ts_argmax(X, d, fillna=True):
    '''
    Returns argmax (which day) within the vector the maximum occurred.
    E.g for X_v = [1,2,3,1,1,2]:

        ts_argmax(X_v, 3) = 2
        ts_argmax(X_v, 5) = 1

    :param X_v:
    :param d: Past d days to look back for computation
    :return:
    '''
    if fillna:
        x = np.nan_to_num(X)
    else:
        x = X
    result = np.copy(x)
    for i in range(d, x.shape[0]+1):
        result[i-1,:] = np.argmax(x[i-d:i], axis=0)
    result[:d-1] = np.nan 
    return result
 

def ts_argmin(X, d, fillna=True):
    if fillna:
        x = np.nan_to_num(X)
    else:
        x = X
    result = np.copy(x)
    for i in range(d, x.shape[0]+1):
        result[i-1,:] = np.argmin(x[i-d:i], axis=0)
    result[:d-1] = np.nan 
    return result

def covariance(X_m, Y_m, d, fillna=True):
    '''
    Computes the covariance between correspond columns in X_m and Y_m based on previous d samples
        (Have some doubts whether this is the actual way rolling covariance is calculated)
        (!!! Something wrong with the calculated values, to be fixed)
    '''

    assert X_m.shape == Y_m.shape

    if fillna:
        x_m = np.nan_to_num(X_m)
        y_m = np.nan_to_num(Y_m)
    else:
        x_m = X_m
        y_m = Y_m

    x_cumsum = np.cumsum(x_m, axis=0)
    y_cumsum = np.cumsum(y_m, axis=0)


    E_x = (x_cumsum[d:] - np.pad(x_cumsum[:-d-1],((1,0),(0,0)), 'constant')).astype(np.float) / d
    E_y = (y_cumsum[d:] - np.pad(y_cumsum[:-d-1],((1,0),(0,0)), 'constant')).astype(np.float) / d

    xy_cumsum = np.cumsum(x_m * y_m, axis=0)
    E_xy = (xy_cumsum[d:] - np.pad(xy_cumsum[:-d-1],((1,0),(0,0)), 'constant')).astype(np.float) / d

    cov = E_xy - E_x*E_y
    cov = np.pad(cov, ((d,0),(0,0)), 'edge')

    return cov

#======= time series stats =============================================================================================

def hurst(ts, lags = 5):
    
    # calculate hurst exponent for multiple time series (column_wise)
    tau = np.zeros((lags-2, ts.shape[1]))
    for lag_ix in range(2,lags):
        tau[lag_ix-2] = np.sqrt(np.std(ts[lag_ix:,:] - ts[:-lag_ix,:], axis = 0))  
    y = log(tau)
    x = np.log(np.array(range(2,lags)))
    x = mb.repmat(x, y.shape[1], 1).T
    
    result = 2.*np.sum((y - np.mean(y, axis=0))*(x - np.mean(x, axis = 0)), axis = 0)/ np.sum((x - np.mean(x, axis=0))**2, axis =0)
    
    return result
    
def rolling_hurst(X, d, ffill = True):
    if ffill:
        x = nanOp.fillNan(X)
    else:
        x = np.copy(X)
    result = np.copy(x) * np.nan
    for i in range(d, close_m.shape[0]):
        result[i] = hurst(close_m[i - d: i])
    result = np.pad(result, ((d,0), (0,0)), mode='edge')
    
    return result

def mAdfuller(ts, lags = 1):
    """return the p value of ADF test for multiple time series column vectors"""
    
    result = []
    for i in range(ts.shape[1]):
        res = tss.adfuller(ts[:,i])[1]
        result.append(res)
    
    result = np.array(result)
    
    return result
    
def HalfLife(ts):
    
    ts_diff = np.diff(ts, axis = 0)
    ts = ts[1:]
    coefs = np.sum((ts_diff - np.mean(ts_diff, axis=0))*(ts - np.mean(ts, axis = 0)), axis = 0)/ np.sum((ts - np.mean(ts, axis=0))**2, axis =0)
    result = np.log(2)/coefs
    return result

##########################################  ops to handle date ##########################################################################

def date_diff(date_m1, date_m2): # date is in integer format
    result = np.copy(date_m1) * np.nan
    y1 = date_m1/ 10000; y2 = date_m2/ 10000;
    m1 = (date_m1 - y1 * 10000)/ 100; m2 = (date_m2 - y2 * 10000)/ 100;
    d1 = date_m1 - y1 * 10000 - m1 * 100; d2 = date_m2 - y2 * 10000 - m2 * 100;
    for i in range(date_m1.shape[0]):
        for j in range(date_m1.shape[1]):
            try:
                date1 = date(y1[i,j], m1[i,j], d1[i,j])
                date2 = date(y2[i,j], m2[i,j], d2[i,j])
                result[i,j] = (date1 - date2).days
            except:
                result[i,j] = np.nan
                continue
    return result

def biz_date_diff(date_m1, date_m2): # date is in integer format
    result = np.copy(date_m1) * np.nan
    y1 = date_m1/ 10000; y2 = date_m2/ 10000;
    m1 = (date_m1 - y1 * 10000)/ 100; m2 = (date_m2 - y2 * 10000)/ 100;
    d1 = date_m1 - y1 * 10000 - m1 * 100; d2 = date_m2 - y2 * 10000 - m2 * 100;
    for i in range(date_m1.shape[0]):
        for j in range(date_m1.shape[1]):
            try:
                date1 = date(y1[i,j], m1[i,j], d1[i,j])
                date2 = date(y2[i,j], m2[i,j], d2[i,j])
                result[i,j] = np.busday_count( date2, date1 )
            except:
                result[i,j] = np.nan
                continue
    return result

def limit(x, high = 30, low=3, out_is_nan = True):
    result = np.copy(x) * np.nan
    result[np.bitwise_and(x < high, x > low)] = x[np.bitwise_and(x < high, x > low)]
    if out_is_nan is False: 
        result[x >= high] = high
        result[x <= low] = low
    return result

def period_diff(period_1, period_2): # fiscal period is in form of YYYYQQ
    y1 = np.round(period_1/ 100)
    y2 = np.round(period_2/ 100)
    q1 = period_1 - y1 * 100
    q2 = period_2 - y2 * 100
    result = (y1 - y2)*4 + (q1 - q2)
    return result

def is_year_first_date(date, date_list):
    assert date in date_list
    ii = date_list.index(date)
    prev_date = date_list[ii-1]
    year = date/10000
    prev_year = prev_date/10000
    if year > prev_year:
        return True
    else:
        return False

def is_month_first_date(date, date_list):
    assert date in date_list
    if is_year_first_date(date, date_list):
        return True
    else:
        ii = date_list.index(date)
        prev_date = date_list[ii-1]
        month = (date - date/10000 * 10000)/100
        prev_month = (prev_date - prev_date/10000 * 10000)/100
        if month > prev_month:
            return True
        else:
            return False



    
