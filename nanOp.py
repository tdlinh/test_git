''' library to handle NaN '''

import numpy as np
import pandas as pd

def gt( x, y ):
    return np.ma.masked_greater( x, y ).mask

def lt( x, y ):
    return np.ma.masked_less( x, y ).mask

def geq( x, y ):
    return np.ma.masked_greater_equal( x, y ).mask

def leq( x, y ):
    return np.ma.masked_less_equal( x, y ).mask

def nanToValue( input_m, value, copy = True ):
    if copy:
        return np.where( np.isnan( input_m ), value, input_m )
    else:
        return np.putmask( input_m, np.isnan( input_m ), value )

def nonFiniteToValue( input_m, value, copy = True ):
    if copy:
        return np.where( ~np.isfinite( input_m ), value, input_m )
    else:
        return np.putmask( input_m, ~np.isfinite( input_m ), value )

def countNanInWindow( nan_m, windowSize ):
    assert windowSize > 0
    zeroNan_m = np.copy( nan_m )
    isNan = np.isnan( zeroNan_m )
    zeroNan_m[ isNan ] = 1.0
    zeroNan_m[ ~isNan ] = 0.0
    result = np.cumsum( zeroNan_m , 0 )
    result[ windowSize: ] -= np.copy( result[ :-windowSize ] )
    return result

def fillNan( input_m, method = 'ffill', axis = 0):
    result = np.copy(input_m)
    
    input_df = pd.DataFrame(result)
    input_df = pd.DataFrame.fillna(input_df, method = method, axis = axis)
   
    return input_df.values

def fillForward(input_m):
    for tickerIndex in xrange(input_m.shape[1]):
        lastValue = 0.0
        for dateIndex in xrange(input_m.shape[0]):
            item = input_m[dateIndex, tickerIndex]
            if ~np.isnan(item):
                lastValue = item
            else:
                input_m[dateIndex, tickerIndex] = lastValue
                
def ffill_window(X, windowSize = 5):
    
    result = np.copy(X)
    con_nan_count = np.zeros(X.shape)  # count consecutive NaN (except for string of NaN at the beginning)
    for ix in range(1, X.shape[0]):
        con_nan_count[ix, ~np.isnan(X[ix])] = 0
        con_nan_count[ix, np.isnan(X[ix])] = con_nan_count[ix-1, np.isnan(X[ix])] + 1
    bol = np.zeros(X.shape).astype(bool)
    bol[np.bitwise_and(np.isnan(X),con_nan_count <= windowSize)] = True
    for ix in range(result.shape[0]):
        if ix>0:
            result[ix,bol[ix]] = result[ix-1,bol[ix]]
    return result

def ffill_exp_decay(X, factor = np.exp(-1.0)):
    result = np.copy(X)
    bol = np.isnan(X)
    for ix in range(1, result.shape[0]):
        if ix>0:
            result[ix,bol[ix]] = result[ix-1,bol[ix]] * factor 
    return result

def ffill_lin_decay_limit_window(X, windowSize = 5):
    
    result = np.copy(X)
    con_nan_count = np.zeros(X.shape)  # count consecutive NaN (except for string of NaN at the beginning)
    for ix in range(1, X.shape[0]):
        con_nan_count[ix, ~np.isnan(X[ix])] = 0
        con_nan_count[ix, np.isnan(X[ix])] = con_nan_count[ix-1, np.isnan(X[ix])] + 1
    bol = np.zeros(X.shape).astype(bool)
    bol[np.bitwise_and(np.isnan(X),con_nan_count <= windowSize)] = True
    for ix in range(result.shape[0]):
        if ix>0:
            result[ix,bol[ix]] = result[ix-1,bol[ix]] * (windowSize - con_nan_count[ix,bol[ix]])/ (windowSize - con_nan_count[ix-1,bol[ix]])
    return result

def ffill_exp_decay_limit_window(X, windowSize = 5, factor = np.exp(-1)):
    
    result = np.copy(X)
    con_nan_count = np.zeros(X.shape)  # count consecutive NaN (except for string of NaN at the beginning)
    for ix in range(1, X.shape[0]):
        con_nan_count[ix, ~np.isnan(X[ix])] = 0
        con_nan_count[ix, np.isnan(X[ix])] = con_nan_count[ix-1, np.isnan(X[ix])] + 1
    bol = np.zeros(X.shape).astype(bool)
    bol[np.bitwise_and(np.isnan(X),con_nan_count <= windowSize)] = True
    for ix in range(result.shape[0]):
        if ix>0:
            result[ix,bol[ix]] = result[ix-1,bol[ix]] * factor
    return result

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

def ffill_ti_limit(input_t, limit = 5):
    # ffill along ti axis of minute bar price with limit to certain number of intervals 
    input_t = input_t.copy()
    
    mask_t = ~np.isfinite(input_t)
    nan_count_t = np.cumsum(mask_t, axis=1)
    
    for ti in range(1, input_t.shape[1]):
        for di in range(input_t.shape[0]):
            mask_v = mask_t[di, ti, :]
            nan_count_v = nan_count_t[di, ti, :]

            for ii, mask in enumerate(mask_v):
                if mask and nan_count_v[ii] <= limit:
                    input_t[di, ti, ii] = input_t[di, ti - 1, ii]

    return input_t