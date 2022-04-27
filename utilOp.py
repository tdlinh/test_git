"""
Created on Thu Jan  6 15:21:18 2022

@author: linh.trinh

this is other ops for this strategy
"""


import numpy as np
from datetime import timedelta

def transform_num_to_instr_code(num, contract_code_map):
    ''' 
    transform an instrument number to instrument codes
    e.g 2017.09 will be transformed to U17
    '''
    month = int(num*100)%100
    year = int(num)%100
    return contract_code_map[month] + str(year)

def convert_date_to_int(dt):
    return int(dt.year * 10000 + dt.month * 100 + dt.day)

def convert_time_to_int(ts):
    return ts.time().hour * 100 + ts.time().minute

def convert_weekend_to_weekday_date(dt):
    ''' 
    if weekday of date is mon-fri, keep the same
    elif saturday, convert to monday (for EU comdty)
    elif sunday, convert to friday (for US comdty)
    '''
    if dt.date().weekday() == 6:
        dt_ = dt - timedelta(days = 2)
    elif dt.date().weekday() == 5:
        dt_ = dt + timedelta(days = 2)
    else:
        dt_ = dt 
    
    return dt_

def convert_index_to_tz(return_df, timezone, convert_weekend_to_weekday = True):
    return_df = return_df.copy()
    return_df.index = return_df.index.tz_convert(timezone)
    if convert_weekend_to_weekday:
        return_df.index = return_df.index.map(convert_weekend_to_weekday_date)
    return return_df

def get_qf2_id(product_id, generic_no, futures_definition_dict):
    bbg_id = futures_definition_dict[product_id]['bbg_id']
    bbg_ylw_key = futures_definition_dict[product_id]['bbg_ylw_key']
    QF2_id = bbg_id + '-' + str(generic_no) + '-' + bbg_ylw_key
    return QF2_id


def get_single_instrument_di_ti_return_dataframe(product_id, generic_no, all_return_df, futures_definition_dict, convert_weekend_to_weekday = True):

    '''
    input is return_df with index in UTC_time
    output return matrix with index in local time of the product_id (convert weekend date accordingly)
    '''

    all_return_df = all_return_df.copy()
    qf2_id = get_qf2_id(product_id, generic_no, futures_definition_dict)
    return_series = all_return_df[[qf2_id]]
    timezone = futures_definition_dict[product_id]['timezone']
    return_df = convert_index_to_tz(return_series, timezone, convert_weekend_to_weekday)
    return_df['Date'] = return_df.index.map(lambda x: convert_date_to_int(x))
    return_df['Time'] = return_df.index.map(lambda x: convert_time_to_int(x))
    return_df = return_df.pivot(index = 'Date', columns = 'Time', values = qf2_id)
    
    return return_df

def get_intraday_return(return_m, time_m, start_time, end_time, return_type='geometric', including_start_time=True, including_end_time =True):
    
    '''
    calculate intraday return from start_time to end_time
    if start_time or end_time is negative, it means they are time from previous day
    '''
        
    assert np.sign(end_time) >= np.sign(start_time)
    if start_time < 0 and end_time < 0:
        assert abs(end_time) > abs(start_time)
        res_lag_m = return_m.copy()
        res_lag_m[1:] = res_lag_m[:-1]
        res_lag_m[0] = np.nan
        res_lag_m[time_m < abs(start_time)] = np.nan
        res_lag_m[time_m > abs(end_time)] = np.nan
        if not including_start_time: res_lag_m[time_m <= abs(start_time)] = np.nan
        if not including_end_time: res_lag_m[time_m >= abs(end_time)] = np.nan
        if return_type == 'geometric':
            res_v = np.nanprod(res_lag_m + 1, axis=1) - 1
        else:
            res_v = np.nansum(res_lag_m, axis=1)
    elif start_time < 0 and end_time >= 0:
        res_m = return_m.copy()
        res_lag_m = res_m.copy()
        res_lag_m[1:] = res_lag_m[:-1]
        res_lag_m[0] = np.nan
        res_m[time_m > abs(end_time)] = np.nan
        res_lag_m[time_m < abs(start_time)] = np.nan
        if not including_start_time: res_lag_m[time_m <= abs(start_time)] = np.nan
        if not including_end_time: res_m[time_m >= abs(end_time)] = np.nan
        if return_type == 'geometric':
            res_v = np.nanprod(res_m + 1, axis=1) * np.nanprod(res_lag_m + 1, axis=1) - 1
        else:
            res_v = np.nansum(res_m, axis=1) + np.nansum(res_lag_m, axis=1)
    elif start_time >= 0 and end_time >= 0:
        res_m = return_m.copy()
        res_m[time_m < abs(start_time)] = np.nan
        res_m[time_m > abs(end_time)] = np.nan
        if not including_start_time: res_m[time_m <= abs(start_time)] = np.nan
        if not including_end_time: res_m[time_m >= abs(end_time)] = np.nan
        if return_type == 'geometric':
            res_v = np.nanprod(res_m + 1, axis=1) - 1
        else:
            res_v = np.nansum(res_m, axis=1)
    return res_v


