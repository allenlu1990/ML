# -*- coding: utf-8 -*-
"""
Created on Sat Sep 09 22:12:35 2017

@author: PJ
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:40:49 2017

@author: PJ
"""

import pandas as pd
import numpy as np
import os
# from features_format import features_format
import logging
log=logging.getLogger('main.entrance.FeatureMining.SingleUserFeatures')

#correl
def correl(dataL,dataR,method = 'whole'):
    """calculate the correl relation between dataL and dataR """
    cf = np.abs(dataL.corr(dataR,method = 'pearson'))
    if method=='whole':
        return cf
    elif method =='each':
        return cf/len(dataL)
#    #area_diifference
def area_difference(dataL,dataR,method = 'whole'):
    """calculate the difference of area dataL,dataR """
    area_diff = np.abs(np.trapz(dataL) - np.trapz(dataR))
    if method == 'whole':
        return area_diff
    elif method =='each':
        return area_diff/len(dataL)
    #standard_deviation_ratio
def standard_deviation_ratio(dataL,dataR,method = 'whole'):
    """ """
    try:
        if method =='whole':
            sdr = max(np.float64((dataL.std()/dataR.std())),np.float64((dataR.std()/dataL.std())))
            return sdr
        elif method == 'each':
            sdr = max(np.float64((dataL.std()/dataR.std())),np.float64((dataR.std()/dataL.std())))/len(dataL)
            return sdr
    except ZeroDivisionError,e:
        print e.message

def abs_max_diff_temp(dataL,dataR):
    """ """
    sdr = np.abs((dataL-dataR).max())
    return sdr
 
def abs_min_diff_temp(dataL,dataR):
    """  """
    sdr = np.abs((dataL-dataR).min())
    return sdr
        
def max_temp(temp):
    """return the max temp """
    sdr = np.max(temp)
    return sdr

def min_temp(temp):
    """return the min temp """
    sdr = np.min(temp)
    return sdr

def avg_temp(temp):
    """return the avg temp """
    sdr = np.mean(temp)
    return sdr

def overtemp_time(temp,threshold=36):
    '''      '''
    sdr = len(temp[temp>36])
    return sdr
       
def one_side_temp_area(data,method = 'whole'):
    """ """
    try:
        if method =='whole':
            sdr = np.abs(np.trapz(data))
            return sdr
        elif method == 'each':
            sdr = np.abs(np.trapz(data))/len(data)
            return sdr
#        print sdr
    except ZeroDivisionError,e:
        print e.message
        
def sum_temp_gradient(data,method = 'whole'):
    """ """
    try:
        if method =='whole':
            sdr = np.abs(np.diff(data)).sum()
            return sdr
        elif method == 'each':
            sdr = np.abs(np.diff(data)).sum()/len(data)
            return sdr
#        print sdr
    except ZeroDivisionError,e:
         log.critical(e.message)
        
def cross_numbers(dataL,dataR,threshold=0.1):
    """
     cross numbers
    """
    diff_data = np.abs(dataL-dataR)
    filter_data = diff_data[diff_data<=threshold]
    return len(filter_data) 
        
def features_Max_whole(features,**input_dit):
#    print 'aa'*20
#    print features
#    print input_dit.keys()
    features_dit = {}
    for i in input_dit:
        features_dit[i] = features[i].max()
    features_dt = pd.DataFrame(data = features_dit,index=[features.index[0]])
#    print 'bb'*20
#    print features_dt
    return features_dt

def features_Max_single(features,**input_dit):
#    print 'aa'*20
#    print features
#    print input_dit.keys()
    features_dit = {}
    features_lst = features.columns
    if 'Temp1Area' in features_lst:
        feature_prior = 'Temp1Area'
    elif 'Temp1Area_each' in features_lst:
        feature_prior = 'Temp1Area_each'
    elif 'Temp2Area' in features_lst:
        feature_prior = 'Temp2Area'
    elif 'Temp2Area_each' in features_lst:
        feature_prior = 'Temp2Area_each'
    elif 'Temp1_abs_diff' in features_lst:
        feature_prior = 'Temp1_abs_diff'
    elif 'Temp2_abs_diff' in features_lst:
        feature_prior = 'Temp2_abs_diff'
#    elif 'Temp1_abs_dif_each' in features_lst:
#        feature_prior = 'Temp1_abs_diff_each'
#    elif 'Temp2_abs_diff_each' in features_lst:
#        feature_prior = 'Temp2_abs_diff_each'        
    for i in input_dit:
#        print 'bb'*20
#        print features[i]
        features_dit[i] = features[i][features[feature_prior]==features[feature_prior].max()].iloc[0]
#        print 'cc'*20
#        print features_dit[i]
#        print features_dit[i]
    features_dt = pd.DataFrame(data = features_dit,index=[features.index[0]])
        
#    print 'bb'*20
#    print features_dt
    return features_dt
    
def features_Min(features,**input_dit):
    features_dit = {}
    for i in input_dit:
        features_dit[i] = features[i].min()
    features_dt = pd.DataFrame(data = features_dit,index=[features.index[0]])
    return features_dt

def features_Avg(features,**input_dit):
    features_dit = {}
    for i in input_dit:
        features_dit[i] = features[i].mean()
    features_dt = pd.DataFrame(data = features_dit,index=[features.index[0]])
    return features_dt

def features_Sum(features,**input_dit):
    features_dit = {}
    for i in input_dit:
        features_dit[i] = features[i].sum()
    features_dt = pd.DataFrame(data = features_dit,index=[features.index[0]])
    return features_dt

def data_bytime(data,start,end,side='both'):
    """return data frame from start time to end time"""
    if side == 'both':
        manipulating = data
    else:
        log.critical('wrong args for side')
        raise 'wrong args for side'
#    print manipulating
    return manipulating.loc[manipulating.time.between(start,end)]

def str2Timestamp(time_data,time_str):
    ''' time_data : Timestamp format
        hour_minute_str:  string format
        example:  time_data =Timestamp('2016-01-13 23:00:00')
        time_str = '01:00'
        return    Timestamp('2016-01-13 01:00:00')
    '''
    time_ts = pd.to_datetime(time_str).replace(year=time_data.year,
                                month=time_data.month,day=time_data.day)
    return time_ts   

def avg_data(valid_data,seg_batch):
    data_num = len(valid_data)
#    print 'data_num:%d'%data_num
    each_seg_data = int(data_num)/int(seg_batch)
#    print 'each_seg_data:%d'%each_seg_data
    valid_seg_dit = {}
#    valid_data = valid_data.reindex()
    valid_data = valid_data.reset_index(drop=True)
#    print valid_data
    for i in range(seg_batch):
        valid_seg_dit['seg'+str(i)] = valid_data[(valid_data.index<(i+1)*each_seg_data) & 
                      (valid_data.index>=i*each_seg_data)]
#    print valid_seg_dit
    return valid_seg_dit

def hour_data(valid_data,hour_num=pd.to_timedelta(1,unit='h')):
    
    time_sr = pd.Series([x.replace(minute=0,second=0) for x in pd.to_datetime(valid_data.time)])
#    print time_sr
    time_sr_unique = list(pd.Series([str(x) for x in time_sr]).unique())
#    print time_sr_unique
    valid_seg_dit = {}
    for i in time_sr_unique:
#        print type(pd.to_datetime(i)),pd.to_datetime(i)
        valid_seg_dit[i] = data_bytime(valid_data,pd.to_datetime(i),
                     pd.to_datetime(i)+hour_num-pd.to_timedelta(1,unit='s'))
#    print valid_seg_dit
#    print valid_seg_dit
    return valid_seg_dit

def data_slice_by_sep(data,sep_time=pd.to_timedelta(1,unit='h'),side='both'):
    """return a dict of data"""
    data_slices = []
    data_hour = {}
    data_rows = data.time.shape[0]
    measure_time = data.time
    delta_time = list(measure_time.diff()) # use list to accel iteration
    time_sr = list(pd.Series([x.hour for x in pd.to_datetime(data.time)]).unique())
    last_sep_loc = 0
    for i in range(1,data_rows):
        if(delta_time[i] > sep_time):
            data_slices.append(data[last_sep_loc:i])
            last_sep_loc = i
        if(i - last_sep_loc != 0):
            data_slices.append(data[last_sep_loc:i])
#    print data_slices
    for j in range(len(data_slices)):
        data_hour[time_sr[j]] = data_slices[j]
    return data_hour


feature_method= {'features_Max_single':features_Max_single,
                 'features_Max_whole':features_Max_whole,
                 'features_Min':features_Min,
                 'features_Avg':features_Min,
                 'features_Sum':features_Sum}

class SingleUserFeatures(object):
    # __data_format__ = features_format()
    __one_day__ = pd.to_timedelta(60*60*24,unit='s')
    __four_hour__ = pd.to_timedelta(4*60*60,unit='s')
    __one_second__ = pd.to_timedelta(1,unit='s')
    def __init__(self,user_id,data,side = 'both'):
        """ """
        self.user_id = user_id
        self.data = data
        self.data_ts = data.copy()
        self.data_ts['time'] = pd.to_datetime(self.data_ts.time)
        self.data.time =[str(x) for x in self.data.time]
#        self.data.time_ts = pd.to_datetime(self.data.time)
        self.data.sort_values(['time'])
        self.origin_start_time = self.data_ts.time.min()
        self.origin_end_time = self.data_ts.time.max()

    
    def count_hour_data(self):
        _date_time = [x.split(':')[0] for x in self.data.time]
        _hour_lst = pd.Series(_date_time).unique()
        _count_data = {}
        for i in _hour_lst:
            _count_data[i] = _date_time.count(i)
        dt = pd.DataFrame(data = _count_data, index = [self.user_id])
        return _date_time,dt
    
    
    def data_slice_by_day(self):
        """return a list of data frame from slicing origin data frame by day"""
        #TODO
        data_slices = []
#        print type(self.origin_start_time)
        slice_start_time = self.origin_start_time.replace(hour=0,minute=0,second=0)-self.__one_second__
        while(slice_start_time < self.origin_end_time):
            d = data_bytime(self.data_ts,slice_start_time,slice_start_time + self.__one_day__)
            if(not d.empty):
                data_slices.append(d)
            slice_start_time = slice_start_time + self.__one_day__
        return data_slices
    

    
    def select_valid_data(self,each_valid_num,valid_seg):
        _date_time,_seg_data = self.count_hour_data()
        _seg_data_filter = _seg_data[_seg_data.loc[:,:]>each_valid_num]
        _seg_data_filter = _seg_data_filter.dropna(axis=1)
        _valid_hour_lst = _seg_data_filter.columns
        if len(_valid_hour_lst)<valid_seg:
            print "this user does not have enough segmental data:%d"%(_valid_hour_lst)
            return None
        _group_data = self.data.copy()
        _group_data['seg_time'] = _date_time
        return _date_time,list(_valid_hour_lst),_group_data
    
    def save_format(self,result_path,features,data_name='features'):
        if type(features)==dict:
            for i in features.keys():
                features[i].to_csv(result_path+self.user_id+'_'+i.replace(' ','_').replace('/','_')
                +data_name+'.csv')
        elif type(features)==pd.DataFrame:
            features.to_csv(result_path+self.user_id+'_'+data_name+'.csv')
        else:
            print "the input_data is not dict or DF"
            return None           
        
    def save_features(self,result_path,save_flag,features,valid_seg_data):
        '''save_flag = 1: save features only
           save_flag = 2: save datas for calc features
           save_flag = 3: save both data and features 
        '''
#        print type(arg_dit)
        if os.path.exists(result_path) == False:
            os.makedirs(result_path)
        if save_flag==0:
            return None
        elif save_flag == 1:
            self.save_format(result_path,features)
        elif save_flag == 2:
            self.save_format(result_path,valid_seg_data,data_name='seg_data')
        elif save_flag == 3:
            self.save_format(result_path,features)
            self.save_format(result_path,valid_seg_data,data_name='seg_data')
        else:
            return None
    def exception_of_time(self,valid_begin,valid_end,start,end):
        if start<valid_begin:
            log.warning("the select start time is earlier than vailid begin time")
            return None
        elif start>valid_end:
            log.warning("the select start time is later than vailid end time")
            return None
        elif end<valid_begin:
            log.warning("the select end time is earlier than vailid begin time")
            return None
        elif end>valid_end:
            log.warning("the select end time is later than vailid end time")
            return None
        else:
            return True
        
    def extract_data_by_time(self,seg_data,start_point,end_point):
        
        _valid_begin = pd.to_datetime(seg_data.time.min())
        _valid_end = pd.to_datetime(seg_data.time.max())
        _use_time_begin = _valid_begin + pd.to_timedelta(start_point,unit='m')
        _use_time_end = _valid_begin + pd.to_timedelta((end_point-1),unit='m')
        valid_seg_data = seg_data[pd.to_datetime(seg_data.time).between(_use_time_begin,
                                  _use_time_end)]
        if self.exception_of_time(_valid_begin,_valid_end,_use_time_begin,_use_time_end)!=True:
            return None
        return valid_seg_data
#    def str2Timestamp(self,time_data,hour_minute_str):
#        ''' time_data : Timestamp format
#            hour_minute_str:  string format
#            example:  time_data =Timestamp('2016-01-13 23:00:00')
#                      hour_minute_str = ('01:00')
#            return    Timestamp('2016-01-13 01:00:00')
#        '''
#        time_data_str = str(time_data).split(' ')[0]
#        time_str = time_data_str+' '+hour_minute_str
#        time_ts = pd.to_datetime(time_str)
#        return time_ts 
    
  
    
    def match_time_data(self,start,end,cday='single'):
        '''start : string format 
           end: string format 
           cday: compatible 
        '''
        days_data_bytime = {}
        if cday == 'single':
            days_data = self.data_slice_by_day()
            for i in range(len(days_data)):
                start_ts = str2Timestamp(days_data[i].time.iloc[0],start)
                end_ts = str2Timestamp(days_data[i].time.iloc[0],end)
                days_data_bytime[str(start_ts).split(' ')[0]]=data_bytime(days_data[i],start_ts,end_ts) 
          
        #%%
        else:
            
#            print '*'*15
#            print self.data.time
#            print '*'*15
#            print type(self.data.time)
            time_sr = pd.Series([x.split(' ')[0] for x in self.data.time])
            user_days_lst = list(time_sr.unique()) #
            user_days_diff = pd.Series([pd.to_datetime(i) for i in user_days_lst]).diff()
            date_exist_yesterday = pd.Series(user_days_lst)[user_days_diff==pd.to_timedelta(1,unit='D')]
            date_exist_yesterday = date_exist_yesterday.reset_index(drop=True) #object format
            start_today_ts = pd.to_datetime(start)
            end_today_ts = pd.to_datetime(end)
#            self.data.time = pd.to_datetime(self.data.time)
            for i in date_exist_yesterday:
               start_ts = (pd.to_datetime(i) - self.__one_day__).replace(hour=start_today_ts.hour,
                           minute=start_today_ts.minute,second=start_today_ts.second)
               end_ts  = pd.to_datetime(i).replace(hour=end_today_ts.hour,
                           minute=end_today_ts.minute,second=end_today_ts.second)
               days_data_bytime[str(start_ts).split(' ')[0]+'--'
                                +str(end_ts).split(' ')[0]]=data_bytime(self.data_ts,start_ts,end_ts)
            
            #%%
            if cday == 'compatible':
#                print type(self.data.time)
                time_sr = pd.Series([x.split(' ')[0] for x in self.data.time])
                user_days_lst = list(time_sr.unique())
                user_days_diff = pd.Series([pd.to_datetime(i) for i in user_days_lst]).diff()
                single_days_index = list(user_days_diff.index[user_days_diff>pd.to_timedelta(1,unit='D')])
                single_days = []
                if single_days_index:
                    if 1 in single_days_index:
                        single_days_index[0]=0
                    for i in range(len(single_days_index)):
                        single_days.append(user_days_lst[single_days_index[i]])
                    for j in single_days:
                        days_data_bytime[j] = data_bytime(self.data_ts,pd.to_datetime(j),
                                        pd.to_datetime(j).replace(hour=23,minute=59,second=59))
        return days_data_bytime
    
    #%%
#    def match_data(self,start_ts,end_ts,
#                   head_threshold=3,tail_threshold= 3):
#        '''start_ts: Timestamp format
#           end_ts:   Timestamp format
#        '''
#        origin_start = pd.to_datetime(self.data.time.min())
#        origin_end = pd.to_datetime(self.data.time.max())
#        head_threshold_td = pd.to_timedelta(head_threshold,unit='h')
#        tail_thershold_td = pd.to_timedelta(tail_threshold,unit='h')
#        
#        if origin_start - start_ts > head_threshold_td:
#            log.warning("the select start time is 3 hours earlier than the cleaned data")
#            return None
#        if end_ts - origin_end > tail_thershold_td:
#            log.warning("the select end time is 3 hours later than the cleaned data")
#            return None
#        if origin_start - end_ts >=pd.to_timedelta(0,unit='h'):
#            log.warning("the select end time is before the cleanned start time")
#            return None
#        if start_ts - origin_end >=pd.to_timedelta(0,unit='h'):
#            log.warning("the select start time is after the cleanned start time")
#            return None  
#        
#        origin_data_ts = pd.to_datetime(self.data.time)
#        match_start = self.data[(origin_data_ts-start_ts)>pd.to_timedelta(0,unit='h')]
#        match_start_ts = pd.to_datetime(match_start.time)
#        match_end = match_start[(match_start_ts-end_ts)<pd.to_timedelta(0,unit='h')] 
#    
#        return match_end
#    
    #%%
    def calc_indice(self,seg_data,**input_dit):
        features = {}
#        print (input_dit.keys())
        for i in input_dit.keys():
#            print i
            if i == 'Temp1Area':
                features[i] = input_dit[i](seg_data['tm1_L'],seg_data['tm1_R'])
            elif i == 'Temp2Area':
                features[i] = input_dit[i](seg_data['tm2_L'],seg_data['tm2_R'])
            elif i =='HRArea':
                features[i] = input_dit[i](seg_data['HR_L'],seg_data['HR_R'])
            elif i=='Temp1Correl':
                features[i] = input_dit[i](seg_data['tm1_L'],seg_data['tm1_R'])
            elif i=='Temp2Correl':
                features[i] = input_dit[i](seg_data['tm2_L'],seg_data['tm2_R'])
            elif i=='HRCorrel':
                features[i] = input_dit[i](seg_data['HR_L'],seg_data['HR_R'])
            elif i=='Temp1Std':
                features[i] = input_dit[i](seg_data['tm1_L'],seg_data['tm1_R'])
            elif i=='Temp2Std':
                features[i] = input_dit[i](seg_data['tm2_L'],seg_data['tm2_R'])    
            elif i=='HRStd':
                features[i] = input_dit[i](seg_data['HR_L'],seg_data['HR_R'])
                
            elif i == 'Temp1_abs_diff':
                features[i] = input_dit[i](seg_data['tm1_L'],seg_data['tm1_R'])
            elif i == 'Temp2_abs_diff':
                features[i] = input_dit[i](seg_data['tm2_L'],seg_data['tm2_R'])  
                
            elif i == 'Temp1Area_left':
                features[i] = input_dit[i](seg_data['tm1_L'])  
            elif i == 'Temp1Area_right':
                features[i] = input_dit[i](seg_data['tm1_R']) 
            elif i == 'Temp2Area_left':
                features[i] = input_dit[i](seg_data['tm2_L']) 
            elif i == 'Temp2Area_right':
                features[i] = input_dit[i](seg_data['tm2_R'])
            elif i == 'Temp1_sum_diff_left':
                features[i] = input_dit[i](seg_data['tm1_L']) 
            elif i == 'Temp1_sum_diff_right':
                features[i] = input_dit[i](seg_data['tm1_R']) 
            elif i == 'Temp2_sum_diff_left':
                features[i] = input_dit[i](seg_data['tm2_L']) 
            elif i == 'Temp2_sum_diff_right':
                features[i] = input_dit[i](seg_data['tm2_R'])     
            elif i == 'T1_cross_numbers':
                features[i] = input_dit[i](seg_data['tm1_L'],seg_data['tm1_R'])
            elif i == 'T2_cross_numbers':
                features[i] = input_dit[i](seg_data['tm2_L'],seg_data['tm2_R'])
                
            elif i == 'T1L_max_temp':
                features[i] = input_dit[i](seg_data['tm1_L'])
            elif i =='T1R_max_temp':
                features[i] = input_dit[i](seg_data['tm1_R'])
            elif i == 'T2L_max_temp':
                features[i] = input_dit[i](seg_data['tm2_L'])
            elif i =='T2R_max_temp':
                features[i] = input_dit[i](seg_data['tm2_R'])
                
            elif i == 'T1L_min_temp':
                features[i] = input_dit[i](seg_data['tm1_L'])
            elif i =='T1R_min_temp':
                features[i] = input_dit[i](seg_data['tm1_R'])
            elif i == 'T2L_min_temp':
                features[i] = input_dit[i](seg_data['tm2_L'])
            elif i =='T2R_min_temp':
                features[i] = input_dit[i](seg_data['tm2_R'])
            
            elif i == 'T1L_avg_temp':
                features[i] = input_dit[i](seg_data['tm1_L'])
            elif i =='T1R_avg_temp':
                features[i] = input_dit[i](seg_data['tm1_R'])
            elif i == 'T2L_avg_temp':
                features[i] = input_dit[i](seg_data['tm2_L'])
            elif i =='T2R_avg_temp':
                features[i] = input_dit[i](seg_data['tm2_R'])
            
            elif i == 'T1L_overtemp_time':
                features[i] = input_dit[i](seg_data['tm1_L'])
            elif i =='T1R_overtemp_time':
                features[i] = input_dit[i](seg_data['tm1_R'])
            elif i == 'T2L_overtemp_time':
                features[i] = input_dit[i](seg_data['tm2_L'])
            elif i =='T2R_overtemp_time':
                features[i] = input_dit[i](seg_data['tm2_R'])
            
            
            
            
                
                
            elif i=='Temp1Area_each':
                features[i] = input_dit[i](seg_data['tm1_L'],seg_data['tm1_R'],method='each')
            elif i==  'Temp2Area_each' :
                features[i] = input_dit[i](seg_data['tm2_L'],seg_data['tm2_R'],method='each')
            elif i == 'HRArea_each':
                features[i] = input_dit[i](seg_data['HR_L'],seg_data['HR_R'],method='each')
#            elif i == 'Temp1Correl_each':
#                features[i] = input_dit[i](seg_data['tm1_L'],seg_data['tm1_R'],method='each')
#            elif i == 'Temp2Correl_each':
#                features[i] = input_dit[i](seg_data['tm2_L'],seg_data['tm2_R'],method='each') 
#            elif i =='HRCorrel_each':
#                features[i] = input_dit[i](seg_data['HR_L'],seg_data['HR_R'],method='each')
#            elif i == 'Temp1Std_each':
#                features[i] = input_dit[i](seg_data['tm1_L'],seg_data['tm1_R'],method='each')
#            elif i == 'Temp2Std_each':
#                features[i] = input_dit[i](seg_data['tm2_L'],seg_data['tm2_R'],method='each')
#            elif i ==  'HRStd_each':
#                features[i] = input_dit[i](seg_data['HR_L'],seg_data['HR_L'],method='each')
            elif i == 'Temp1Area_left_each':
                features[i] = input_dit[i](seg_data['tm1_L'],method ='each')  
            elif i == 'Temp1Area_right_each':
                features[i] = input_dit[i](seg_data['tm1_R'],method='each') 
            elif i == 'Temp2Area_left_each':
                features[i] = input_dit[i](seg_data['tm2_L'],method='each') 
            elif i == 'Temp2Area_right_each':
                features[i] = input_dit[i](seg_data['tm2_R'],method='each')
            elif i == 'Temp1_sum_diff_left_each':
                features[i] = input_dit[i](seg_data['tm1_L'],method='each') 
            elif i == 'Temp1_sum_diff_right_each':
                features[i] = input_dit[i](seg_data['tm1_R'],method='each') 
            elif i == 'Temp2_sum_diff_left_each':
                features[i] = input_dit[i](seg_data['tm2_L'],method='each') 
            elif i == 'Temp2_sum_diff_right_each':
                features[i] = input_dit[i](seg_data['tm2_R'],method='each')    

#            elif i == 'Temp1_abs_diff_each':
#                features[i] = input_dit[i](seg_data['tm1_L'],seg_data['tm1_R'],method='each')
#            elif i == 'Temp2_abs_diff_each':
#                features[i] = input_dit[i](seg_data['tm2_L'],seg_data['tm2_R'],method='each')  
                
            features_dt = pd.DataFrame(data=features,index=[self.user_id])
        return features_dt
    
    #calculate_features        
    def calculate_features_(self,side ='both',start='0:00:00',
                            end='6:00:00',cday='single',seg_method='oneDay',
                            hour_num = 1,extact_method= 'features_Max',
                            seg_batch=3,save_flag=0,
                            features_path=os.path.join(os.getcwd(),'result'),**input_dit): 
        ''' '''
#        print 'aaaaaaaaaaaaa'
        state=True
        if self.data[self.data.isnull().values ==True].empty == False:
            log.warning('there has nan in data')
            state=False
            return state, None
        
        start_end_diff = pd.to_datetime(start) - pd.to_datetime(end)
#        print start_end_diff, start, '-', end
        if cday == 'single':
            if start_end_diff > pd.to_timedelta(1,unit='s'):
                log.warning("the start time and the end time must "+" in the same day under single mode")
                state=False
                return state, None
        
        valid_data = self.match_time_data(start,end,cday)
        for i in valid_data.keys():
            if valid_data[i].empty:
                del valid_data[i]
        
        if len(valid_data.values())==0:
            log.warning(self.user_id+": not enough data")
            state=False
            return state, None

        hour_num = pd.to_timedelta(hour_num*3600,unit='s')
        features =pd.DataFrame()
        
#        if seg_method =='whole' or seg_method =='oneDay':
        if  seg_method =='oneDay':   
            for i in valid_data.keys():
                features_part = self.calc_indice(valid_data[i],**input_dit)
                features_part['start_time'] = str(valid_data[i].time.min())
                features_part['end_time'] = str(valid_data[i].time.max())
                features = features.append(features_part)
#            if seg_method =='whole':
#                features_rt =  features_Sum(features,**input_dit)  
#                result_path = os.path.join(features_path,cday)+os.path.sep
#            else:
                features_rt =  feature_method[extact_method](features,**input_dit) 
                result_path = os.path.join(features_path,cday,seg_method,extact_method)+os.path.sep
            self.save_features(result_path,save_flag,features_rt,valid_data)
            
#        elif seg_method =='seg_avg' or seg_method =='hour':
#            valid_seg_dit = {}
#            seg_data_dit = {}
#            for i in valid_data.keys():
#                if seg_method=='seg_avg':
#                    seg_data_dit = avg_data(valid_data[i],seg_batch)
#                elif seg_method=='hour':
#                    seg_data_dit = hour_data(valid_data[i],hour_num)
#                for j in seg_data_dit.keys():
#                    valid_seg_dit[i+j] = seg_data_dit[j]
#                    
#            for jj in valid_seg_dit.keys():
#                features_part = self.calc_indice(valid_seg_dit[jj],**input_dit)
#                features_part['start_time'] = str(valid_seg_dit[jj].time.min())
#                features_part['end_time'] = str(valid_seg_dit[jj].time.max())    
#                features = features.append(features_part)
#            result_path = os.path.join(features_path,cday,seg_method,extact_method)+os.path.sep
#            self.save_features(result_path,save_flag,features,valid_data)
#            features_rt =  feature_method[extact_method](features,**input_dit) 
##            result_path = os.path.join(features_path,cday,seg_method,extact_method)+os.path.sep
            
        else:
            log.warning("seg_method is not qualified")
            state=False
            return state, None
        
#        self.save_features(result_path,save_flag,features_rt,valid_data)

        return state, features_rt

def cac_all(user_id,Temper):
    
    a = SingleUserFeatures(user_id,Temper,side = 'both')
    state, feat_df = a.calculate_features_(side ='both',start='0:00',
                                           end='6:00',cday='single',
                                           seg_method='oneDay',hour_num = 5,
                                           extact_method = 'features_Max_single',
                                           seg_batch=10,save_flag=0,
                                           features_path=os.path.join(os.getcwd(),'result'),
                                           Temp1Area_each=area_difference, 
                                           Temp2Area_each=area_difference,
                                           Temp1_abs_diff=abs_max_diff_temp,
                                           Temp2_abs_diff=abs_max_diff_temp,
                                           HRArea = area_difference,
                                           HRCorrel = correl,
                                           HRStd = standard_deviation_ratio,
                                           Temp1Area=area_difference,
                                           Temp2Area=area_difference,
                                           Temp1Std = standard_deviation_ratio,
                                           Temp2Std = standard_deviation_ratio,
                                           Temp1Correl = correl,
                                           Temp2Correl = correl,
                                           Temp1Area_left_each = one_side_temp_area,
                                           Temp1Area_right_each = one_side_temp_area,
                                           Temp1_sum_diff_left_each = sum_temp_gradient,
                                           Temp1_sum_diff_right_each = sum_temp_gradient,
                                           Temp2Area_left_each = one_side_temp_area,
                                           Temp2Area_right_each = one_side_temp_area,
                                           Temp2_sum_diff_left_each = sum_temp_gradient,
                                           Temp2_sum_diff_right_each = sum_temp_gradient,
                                           T1_cross_numbers = cross_numbers,
                                           T2_cross_numbers = cross_numbers,
                                           T1L_max_temp = max_temp,
                                           T1R_max_temp = max_temp,
                                           T2L_max_temp = max_temp,
                                           T2R_max_temp = max_temp,                                           
                                           T1L_min_temp = min_temp,
                                           T1R_min_temp = min_temp,
                                           T2L_min_temp = min_temp,
                                           T2R_min_temp = min_temp,                                               
                                           T1L_avg_temp = avg_temp,
                                           T1R_avg_temp = avg_temp,
                                           T2L_avg_temp = avg_temp,
                                           T2R_avg_temp = avg_temp,                                             
                                           T1L_overtemp_time = overtemp_time,
                                           T1R_overtemp_time = overtemp_time,
                                           T2L_overtemp_time = overtemp_time,
                                           T2R_overtemp_time = overtemp_time
                                           )
    return state, feat_df
        
if __name__ == "__main__":
    import glob
    file_path = os.path.join(os.getcwd(),'cleaned')+os.path.sep
    lst = glob.glob(file_path + "*.csv")
    features_dt = pd.DataFrame()
    for i in lst:
        user_id = os.path.splitext(os.path.basename(i))[0]
        print user_id
        data = pd.read_csv(i)  
#        print data
        a = SingleUserFeatures(user_id,data)
#        print features
        '''
            side = 'both''
            start = 'H:M:S'
            end = 'H:M:S'
            cday = 'single': start end in oneday
            cday = 'compatible': if user's data has two continous day,
                                    choose it first;otherwise, use single day
            
           seg_method = 'whole': user's whole data
           seg_method = 'oneDay': user's one-day data
           seg_method = 'seg_avg': average segmental each day's data
           seg_method = 'hour': use  hour data
           save_flag = 0:  do not save  
           save_flag = 1: save features only
           save_flag = 2: save datas for calc features
           save_flag = 3: save both data and features 
           save_path = save_path
           side='both'
           valid_seg=3,each_valid_num=20,start_point=0,end_point=20
           **arg_dict: {Temp1Area,Temp2Area=area_difference,
                        Temp1Std,Temp2Std,HRStd=standard_deviation_ratio,
                        Temp1Correl,Temp2Correl,HRCorrel = correl}
        '''
        state,features = a.calculate_features_(side ='both',start='0:00',
                                        end='5:00',cday='single',
                                        seg_method='seg_avg',hour_num = 1,
                                        extact_method = 'features_Max_single',
                                        seg_batch=3,save_flag=0,
                                        features_path=os.path.join(os.getcwd(),'result'),
                                        Temp1Area_each=area_difference, 
                                        Temp2Area_each=area_difference,
                                        Temp1_abs_diff=abs_max_diff_temp,
                                        Temp2_abs_diff=abs_max_diff_temp,
                                        HRArea = area_difference,
                                        HRCorrel = correl,
                                        HRStd = standard_deviation_ratio,
                                        Temp1Area=area_difference,
                                        Temp2Area=area_difference,
                                        Temp1Std = standard_deviation_ratio,
                                        Temp2Std = standard_deviation_ratio,
                                        Temp1Correl = correl,
                                        Temp2Correl = correl)
        
        features_dt = features_dt.append(features)
