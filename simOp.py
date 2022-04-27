"""
Created on Thu Jan  6 15:21:18 2022

@author: linh.trinh

this is library to perform back-test simulation
"""

import numpy as np
import matplotlib.pyplot as plt

def maxDrawDown(pnl_v):
    
    ''' 
    calculate MDD for a return time series
    '''
    
    equity = pnl_v.cumsum()
    equity_max = -999
    MDD = -999
    
    for k in range(len(pnl_v)):
        equity_max = np.maximum(equity_max, equity[k])
        DD = equity_max - equity[k]
        MDD = np.maximum(MDD, DD)
    
    return MDD

def sim(alpha_v, return_v, TC_c_bps = None, TC_ba_bps = None, iAnnualFactor = 252, show_stats = True):
    
    assert len(alpha_v) == len(return_v)
    
    pnl_v = alpha_v * return_v
    pnl_bc_v = pnl_v.copy()
    ave_profit_per_trade_bc = np.nanmean(pnl_v[np.fabs(pnl_v) > 0])
    
    if TC_c_bps is not None:
        pnl_v = pnl_v - np.fabs(alpha_v) * (TC_c_bps/ 1e4) * 2. # round trip everyday
    if TC_ba_bps is not None:
        pnl_v = pnl_v - np.fabs(alpha_v) * (TC_ba_bps/ 1e4) # half spread, round trip everyday
    ave_profit_per_trade_ac = np.nanmean(pnl_v[np.fabs(pnl_v) > 0])
    hit_ratio = np.sum(pnl_v > 0)/ np.sum(np.fabs(pnl_v) > 0)
    trading_freq = np.sum(np.fabs(alpha_v) > 0)/ len(alpha_v)
    
    annual_strat_ret = np.nanmean(pnl_v) * iAnnualFactor
    annual_strat_vol = np.nanstd(pnl_v) * np.sqrt(iAnnualFactor)
    annual_sharpe = annual_strat_ret/ annual_strat_vol
    mdd = maxDrawDown(pnl_v)
    
    if show_stats:
        print('Number of backtest days: ', len(pnl_v))
        print('Trading frequency:  {0:.2f}%'.format(trading_freq*100))
        print('Annualized return:  {0:.2f}%'.format(annual_strat_ret*100))
        print('Annualized volatility: {0:.2f}%'.format(annual_strat_vol*100))
        print('Annualized Sharpe: {0:.2f}'.format(annual_sharpe))
        print('Hit Ratio: {0:.2f}'.format(hit_ratio))
        print('Max Drawdown: {0:.2f}%'.format(mdd*100))
        print('Average profit per trade before cost: {0:.2f}%'.format(ave_profit_per_trade_bc*100))
        print('Average profit per trade after cost: {0:.2f}%'.format(ave_profit_per_trade_ac*100))
    
    return pnl_v, pnl_bc_v

def show_strategy_plot(pnl_v, dates_v, ticker = None):

    assert len(pnl_v) == len(dates_v)
    
    dates_v = np.array(dates_v).astype(int)
    years = np.unique(dates_v//10000)
    xTickPos = np.zeros(len(years))
    for yi, year in enumerate(years):
        xTickPos[yi] = np.where(dates_v == dates_v[dates_v > year*10000][0])[0]
    plt.figure()
    ax = plt.subplot(111)
    if ticker is not None:
        title = 'Cumulative Return of Strategy for ' + ticker
    else:
        title = 'Cumulative Return of Strategy'
    ax.set_title(title)
    ax.plot(np.cumsum(pnl_v))
    ax.set_xlim([0,len(pnl_v)])
    ax.xaxis.grid(True, which='major')
    plt.xticks(xTickPos,years, rotation='vertical')
    plt.show()


