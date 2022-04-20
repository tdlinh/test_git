''' 
this is library to simulate, summarize and plot result of trading strategies
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

def sim(alpha_m, return_m, delay = 0, bookSize = 20e6, tpnl_m = None, bc_m = None, slippage_m = None, isTvr = True, isIndividual = False):
    '''
    calculate daily pnl of constant booksize strategies
    default is calculation of only holding pnl (close-to-close return) with no cost
    tpnl_m is trading pnl which can be input (daily fill-to-close return)
    bc_m can be input (borrowing cost), slippage can be input (spread cost)
    '''
    
    alpha_m = alpha_m.copy()
    return_m = return_m.copy()
    if delay > 0:
        alpha_m[delay:,:] = alpha_m[:-delay,:]
        alpha_m[:delay] = np.nan
    
    pnl_m = np.zeros(alpha_m.shape, dtype = np.float32)
    alphaDiff = np.diff(alpha_m, axis=0)
    
    # pnl is position times return
    pnl_m = alpha_m * return_m
    
    # add trading pnl
    if not (tpnl_m is None):
        pnl_m = pnl_m + tpnl_m
        
    # subtract borrowing cost 
    # assume it is based on yesterday holding
    if not (bc_m is None):
        cost_m = alpha_m * bc_m
        cost_m[cost_m > 0.] = 0. # no bc for long position
        cost_m[~np.isfinite(cost_m)] = 0.
        pnl_m[1:] = pnl_m[1:] + cost_m[:-1]
    
    # subtract slippage cost
    if not (slippage_m is None):
        slippage_cost_m = slippage_m[1:] * np.fabs(alphaDiff)
        slippage_cost_m[~np.isfinite(slippage_cost_m)] = 0.
        pnl_m[1:] = pnl_m[1:] - slippage_cost_m
        
    pnl_m[~np.isfinite(pnl_m)] = 0.0
    
    # calculate TVR
    if isTvr:
        tvr_m = np.zeros(alpha_m.shape, dtype=np.float32)
        if bookSize is None:
            bookSize_v = np.sum(np.fabs(alpha_m), axis = 1)
            tvr_m[1:, :] = (np.fabs(alphaDiff).T / bookSize[:-1]).T
        else:
            tvr_m[1:, :] = np.fabs(alphaDiff) / bookSize
    else:
        tvr_m = 0.0
        
    if not isIndividual:
        pnl_m = np.sum(pnl_m, axis=1)
        if isTvr:
            tvr_m[~np.isfinite(tvr_m)] = 0.0
            tvr_m = np.sum(tvr_m, axis=1)

    return (pnl_m, tvr_m)


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

def alphaConcentration(alpha_m, percentile = 0.5):
    
    ''' 
    compute how many instruments make up a given percentile of portofolio
    calculate from the biggest position ascending down
    '''
    
    bookConcent = np.zeros(alpha_m.shape[0])
    
    for di in range(alpha_m.shape[0]):
        alpha_v = alpha_m[di]
        bookSize = np.sum(np.fabs(alpha_v))
        alpha_v_desc = np.fabs(alpha_v[np.argsort(np.fabs(alpha_v))[::-1]])/bookSize
        alpha_v_desc = alpha_v_desc[alpha_v_desc>0]
        cumBook = 0
        for cnt, alpha_s in enumerate(alpha_v_desc):
            cumBook += alpha_s
            if cumBook > percentile: 
                bookConcent[di] = cnt+1
                break
    return np.median(bookConcent)


def pnlGraph(pnl_v, label_v = False, bookSize=20e6, xLabel = 'Time', yLabel = 'Return', xTickPos = [], xTickLabel = [], title = '', legendTitle = '', loc = None):
    
    '''
    plot graph of time series pnl_v
    '''
    
    fig = plt.figure()
    ax = plt.subplot(111)
    for pi, pnl in enumerate(pnl_v):
        ax.plot(pnl.cumsum()/bookSize,label=label_v[pi])
    ax.set_title(title)
    ax.legend(loc='best', bbox_to_anchor=(1, 0.5), title=legendTitle)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.xaxis.grid(True, which='major')
    #ax.yaxis.grid(True, which='major')
    ax.set_xlim([0,len(pnl_v[0])])
    if len(xTickPos) >0:
        plt.xticks(xTickPos,xTickLabel, rotation='vertical')
    if loc is not None:
        for loc_i in loc:
            plt.axvline(loc_i, color='r')
    plt.show()   

class alphaStat(object):
    
    ''' 
    calculate common stats for alpha
    '''
    
    def __init__(self, alpha_m, pnl_v, tvr_v, dates_v, bookSize=20e6):
        self.IR = np.mean( pnl_v) / np.std( pnl_v )
        self.annualReturn = np.mean( pnl_v ) / bookSize * 250 * 100
        self.MDD = maxDrawDown( pnl_v ) * 100 / bookSize
        self.turnover = np.mean( tvr_v ) * 100
        self.concent50 = alphaConcentration(alpha_m, percentile = 0.5)
        self.concent90 = alphaConcentration(alpha_m, percentile = 0.9)
        self.winning_pct = 1.* np.sum(pnl_v > 0) / len(pnl_v) * 100
  
    def getResult(self):
        return [self.IR, self.annualReturn, self.MDD, self.turnover, self.concent50, self.concent90, self.winning_pct]
    
    
def summary(alpha_m, pnl_v, tvr_v, dates_v, bookSize=20e6, isAnnualSummary = True, loc = None, plot=True):
    
    '''
    summarize yearly stats for alpha and plot graph 
    '''
    
    # segregate into each year
    years = np.unique(dates_v//10000)
    xTickPos = np.zeros(len(years))
    for yi, year in enumerate(years):
        xTickPos[yi] = np.where(dates_v == dates_v[dates_v > year*10000][0])[0]
    segDates = [dates_v//10000 == years[yi] for yi in range(len(years))]
    segAlpha = [ alpha_m[segDates[yi]] for yi in range(len(years))]
    segPnl = [ pnl_v[segDates[yi]] for yi in range(len(years))]
    segTvr = [ tvr_v[segDates[yi]] for yi in range(len(years))]
    
    # calcualte stats for each year
    yearlyStat = [alphaStat( segAlpha[si], segPnl[si], segTvr[si], segDates[si], bookSize = bookSize) for si in range(len(years))]
    
    # plot and summarize
    result = np.zeros([len(years)+1,7]) # number of years and one row for whole period summary, 7 is number of stats in alphaStat
    for yi in range(len(years)):
        result[yi] = yearlyStat[yi].getResult()
    result[len(years)] = alphaStat( alpha_m, pnl_v, tvr_v, dates_v, bookSize = bookSize).getResult()
    if plot == True:
        message = 'Annualized Sharpe = %.3f , Annualized Return = %.2f, MDD = %.2f, TVR = %.2f' %(result[len(years),0] * np.sqrt(365), result[len(years),1], result[len(years),2], result[len(years),3])
        pnlGraph([pnl_v],[''], xTickPos = xTickPos, xTickLabel = years,title=message, loc = loc)
    print ('\033[0m')
    pd.options.display.float_format = '{:,.2f}'.format
    resultTable = pd.DataFrame(result, index=np.append(years,'Total'), columns=['IR','Return(%)','MDD(%)', 'Turnover(%)', 'Concent50', 'Concent90','Winning(%)'])
    resultTable['IR'] = resultTable['IR'].map('{:,.3f}'.format)
    resultTable['Concent50'] = resultTable['Concent50'].map('{:,.0f}'.format)
    resultTable['Concent90'] = resultTable['Concent90'].map('{:,.0f}'.format)
    resultTable['Winning(%)'] = resultTable['Winning(%)'].map('{:,.2f}'.format)
    
    print (resultTable)


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        