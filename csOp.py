"""
Created on Thu Jan  6 15:21:18 2022

@author: linh.trinh

this is library to perform cross sectional operation such as cross sectional rank
"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats.mstats import winsorize


def vectorRank( input_v, normalize = True):
    '''
    use sp.stats.rankdata to rank a vector and normalize after rank
    '''
    
    result = np.copy( input_v )
    result = sp.stats.rankdata( result, method = 'average' )
        
    if normalize:
        result /= np.amax( result )     
    
    return result

def groupRank(alpha_m, universe_m = None, group_m = None, normalize = True):
    '''
    cross-sectional rank each row of alpha_m within the universe (optional) and in each group (optional)
    assume grouping data will be in positive integer 
    '''

    alpha_m = alpha_m.copy()
    result = np.full_like(alpha_m, np.nan)
    
    if universe_m is None:
        valid = np.isfinite( alpha_m )
    else:
        valid = np.bitwise_and( universe_m, np.isfinite( alpha_m ) )
    
    if group_m is None:
        print ('there is no group')
        group_m = valid.astype('int32')

    groupValid_m = np.copy(group_m)
    groupValid_m[ ~valid ] = -1

    for di, dailyAlpha in enumerate( alpha_m ):
        dailyUniv = valid[di, :]
        dailyGroup = groupValid_m[di, :]
        dailyAlpha[~dailyUniv] = np.nan
        
        uniqueGroup_v = np.unique(dailyGroup[np.isfinite(dailyGroup)])
        uniqueGroup_v = uniqueGroup_v[uniqueGroup_v >= 0] 

        for gi, groupIx in enumerate(uniqueGroup_v):
            groupPick = np.bitwise_and(dailyUniv, dailyGroup == groupIx)
            dailyAlpha[groupPick] = vectorRank( dailyAlpha[groupPick], normalize = normalize)
        
        result[di] = dailyAlpha
    
    #result[~valid] = np.nan
    #result[~np.isfinite(group_m)] = np.nan
        
    return result

def groupNeutralize( alpha_m, universe_m = None, group_m = None):
    '''
    cross-sectional de-mean each row of alpha_m within the universe (optional) and in each group (optional)
    assume grouping data will be in positive integer 
    '''
    
    alpha_m = alpha_m.copy()
    neutAlpha_m = np.full_like(alpha_m, np.nan)

    if universe_m is None:
        valid = np.isfinite( alpha_m )
    else:
        valid = np.bitwise_and( universe_m, np.isfinite( alpha_m ) )
    
    if group_m is None:
        print ('there is no group')
        group_m = valid.astype('int32')

    groupValid_m = np.copy(group_m)
    groupValid_m[ ~valid ] = -1
    
    for di, dailyAlpha in enumerate( alpha_m ):
        dailyUniv = valid[di, :]
        dailyGroup = groupValid_m[di, :]
        dailyAlpha[~dailyUniv] = np.nan
        
        uniqueGroup_v = np.unique(dailyGroup[np.isfinite(dailyGroup)])
        uniqueGroup_v = uniqueGroup_v[uniqueGroup_v >= 0] 

        for gi, groupIx in enumerate( uniqueGroup_v ):
            groupPick = np.bitwise_and( dailyUniv, dailyGroup == groupIx )
            dailyAlpha[ groupPick ] -= np.mean( dailyAlpha[ groupPick ] )
            
        neutAlpha_m[ di ] = dailyAlpha

    #neutAlpha_m[~valid] = np.nan
    #neutAlpha_m[~np.isfinite(group_m)] = np.nan
     
    return neutAlpha_m

def scale( alpha_v, bookSize ):
    '''
    scale a vector to booksize i.e sum(abs(alpha_v)) = bookSize
    '''
    
    result = np.copy( alpha_v )
    scaleFactor = bookSize / np.nansum( np.fabs( result ) )
    if np.isfinite( scaleFactor ):
        result *= scaleFactor

    return result
        
def scaleAll( alpha_m, bookSize, universe_m = None):
    '''
    scale each row of alpha_m to bookSize
    NaN becomes 0
    '''
    
    result = np.copy( alpha_m )
    if not ( universe_m is None ):
        assert alpha_m.shape == universe_m.shape
        result[ ~universe_m ] = 0.

    for di, alpha_v in enumerate( result ):
        validIx = np.isfinite( alpha_v )
        alpha_v[ ~validIx ] = 0.0
        alpha_v = scale( alpha_v, bookSize )
        result[ di, : ] = alpha_v
    return result

def remap(alpha, universe = None, booksize=20e6, scale = True):
    '''
    remap to make long-short neutral by balance positive and negative positions, then daily scale to booksize
    '''
    
    alpha = pd.DataFrame(alpha)
    if universe is None:
        universe = np.ones(alpha.shape).astype(bool)
    universe = pd.DataFrame(universe)
    
    result = alpha[universe].copy()
    adj_factor = -result[result>0].sum(axis=1)/result[result<0].sum(axis=1)
    result[result<0] = result[result<0].multiply(adj_factor, axis=0)
    if scale:
        result = result.divide(result.abs().sum(axis=1), axis=0)*booksize
    
    res = result.values
    res[~np.isfinite(res)] = 0.
    
    return res

def dampen(alpha_m, barrierValue, bookSize = None, universe_m = None, noiseFactor = 0.0):
    '''
    dampen the trade to 0 if it is smaller than certain barrier value 
    this operation is to reduce TVR (and i.e. cost) but it will incur path-dependency
    '''
    
    if universe_m is None:
        valid = np.isfinite( alpha_m )
    else:
        valid = np.bitwise_and( universe_m, np.isfinite( alpha_m ) )
    
    result = np.copy( alpha_m )

    for di in range( 1, result.shape[0] ):
        alphaDiff_v = result[ di ] - result[ di - 1 ]
        tempValid_v = np.bitwise_or( valid[ di ], valid[ di - 1 ] )
        if bookSize is None:
            barrierDollarSize = np.fabs(barrierValue * np.sum(np.fabs(result[ di, tempValid_v ])))
        else:
            barrierDollarSize = barrierValue * bookSize
                                     
        buyIx_v = np.bitwise_and( alphaDiff_v >= barrierDollarSize, tempValid_v )
        sellIx_v = np.bitwise_and( alphaDiff_v <= - barrierDollarSize, tempValid_v )
        noiseIx_v = np.bitwise_and( ~np.bitwise_or( buyIx_v, sellIx_v ), tempValid_v )
        
        if noiseFactor == 0:
            alphaDiff_v[noiseIx_v] = 0
        else:
            alphaDiff_v[ noiseIx_v ] *= noiseFactor
        
        alphaDiff_v[ buyIx_v ] -= barrierDollarSize
        alphaDiff_v[ sellIx_v ] += barrierDollarSize
        result[ di ] = result[ di - 1 ] + alphaDiff_v

    return result


def vectorZscore( input_v ):
    '''
    calculate z-score of each element in input_v to mean(input_v)
    '''
    
    result = np.copy( input_v )
    result = (result - np.nanmean(result))/ np.nanstd(result)
        
    return result

def groupZscore(alpha_m, universe_m = None, group_m = None):
    '''
    cross-sectional rank each row of alpha_m within the universe (optional) and in each group (optional)
    assume grouping data will be in positive integer 
    '''

    alpha_m = alpha_m.copy()
    result = np.full_like(alpha_m, np.nan)
    
    if universe_m is None:
        valid = np.isfinite( alpha_m )
    else:
        valid = np.bitwise_and( universe_m, np.isfinite( alpha_m ) )
    
    if group_m is None:
        print ('there is no group')
        group_m = valid.astype('int32')

    groupValid_m = np.copy(group_m)
    groupValid_m[ ~valid ] = -1

    for di, dailyAlpha in enumerate( alpha_m ):
        dailyUniv = valid[di, :]
        dailyGroup = groupValid_m[di, :]
        dailyAlpha[~dailyUniv] = np.nan
        
        uniqueGroup_v = np.unique(dailyGroup[np.isfinite(dailyGroup)])
        uniqueGroup_v = uniqueGroup_v[uniqueGroup_v >= 0] 

        for gi, groupIx in enumerate(uniqueGroup_v):
            groupPick = np.bitwise_and(dailyUniv, dailyGroup == groupIx)
            dailyAlpha[groupPick] = vectorZscore( dailyAlpha[groupPick])
        
        result[di] = dailyAlpha
    
    #result[~valid] = np.nan
    #result[~np.isfinite(group_m)] = np.nan
        
    return result

def vectorWinsorize( input_v , limits = 0.5):
    '''
    calculate z-score of each element in input_v to mean(input_v)
    '''
    
    result = np.copy( input_v )
    result = winsorize(result, limits = limits)
        
    return result

def groupWinsorize(alpha_m, universe_m = None, group_m = None, limits = 0.5):
    '''
    cross-sectional rank each row of alpha_m within the universe (optional) and in each group (optional)
    assume grouping data will be in positive integer 
    '''

    alpha_m = alpha_m.copy()
    result = np.full_like(alpha_m, np.nan)
    
    if universe_m is None:
        valid = np.isfinite( alpha_m )
    else:
        valid = np.bitwise_and( universe_m, np.isfinite( alpha_m ) )
    
    if group_m is None:
        print ('there is no group')
        group_m = valid.astype('int32')

    groupValid_m = np.copy(group_m)
    groupValid_m[ ~valid ] = -1

    for di, dailyAlpha in enumerate( alpha_m ):
        dailyUniv = valid[di, :]
        dailyGroup = groupValid_m[di, :]
        dailyAlpha[~dailyUniv] = np.nan
        
        uniqueGroup_v = np.unique(dailyGroup[np.isfinite(dailyGroup)])
        uniqueGroup_v = uniqueGroup_v[uniqueGroup_v >= 0] 

        for gi, groupIx in enumerate(uniqueGroup_v):
            groupPick = np.bitwise_and(dailyUniv, dailyGroup == groupIx)
            dailyAlpha[groupPick] = vectorWinsorize( dailyAlpha[groupPick], limits = limits)
        
        result[di] = dailyAlpha
    
    #result[~valid] = np.nan
    #result[~np.isfinite(group_m)] = np.nan
        
    return result