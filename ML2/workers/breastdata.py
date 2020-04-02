#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:37:38 2017
Single User Data for Well Diagnosis Breast Cancer
@author: wangzilong
"""
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
import copy
import logging
log=logging.getLogger('main.entrance.preprocess.breastdata')

#%%
class SingleUserData(object):
    __one_day__ = pd.to_timedelta(60*60*24,unit='s')
    __four_hour__ = pd.to_timedelta(4*60*60,unit='s')
    
    def __init__(self,file_dir,phone_number):
        """Read data frame froma csv file"""
        # Data frame should have a columns include measure_time,
        # server_record_time,devicePart,temperature,room_temperature
        csvfile="{}{}.csv".format(file_dir, phone_number)  # repaired by zhuxw
        self.__data = pd.read_csv(csvfile)
        self.__data.dropna(axis=0)  # added by zhuxw
        self.__data['time'] = pd.to_datetime(self.__data.measure_time,unit='s')
        self.__data['time'] = self.__data['time'] + pd.to_timedelta(8*60*60,unit='s')
        self.__data.sort_values(['time'])
        self.start_time = self.__data['time'].min()
        self.end_time = self.__data['time'].max()
        self.phone_number = phone_number
        self.data_all = self.__data
        self.data_left = self.__data.loc[self.__data.devicePart == 0]
        self.data_right = self.__data.loc[self.__data.devicePart == 1]
#        print self.data_left
#        print self.data_right
    
    def data_bytime(self,start,end,side='both'):
        """return data frame from start time to end time"""
        if side == 'both':
            manipulating = self.data_all
        elif side.startswith('l'):
            manipulating = self.data_left
        elif side.startswith('r'):
            manipulating = self.data_right
        else:
            raise Exception('wrong args for side')
        return manipulating.loc[manipulating.time.between(start,end)]
    
    def data_bytime2(self,data,start,end):
        """return data frame from start time to end time"""
#        if side == 'both':
#            manipulating = self.data_all
#        elif side.startswith('l'):
#            manipulating = self.data_left
#        elif side.startswith('r'):
        manipulating = data
#        else:
#            raise 'wrong args for side'
        return manipulating.loc[manipulating.time.between(start,end)]
    
    def data_slice_by_day(self):
        """return a list of data frame from slicing origin data frame by day"""
        #TODO
        data_slices = []
        slice_start_time = self.start_time.replace(hour=0,minute=0,second=0) - self.__four_hour__
        while(slice_start_time < self.end_time):
            d = self.data_bytime(slice_start_time,slice_start_time + self.__one_day__)
            if(not d.empty):
                data_slices.append(d)
            slice_start_time = slice_start_time + self.__one_day__
        return data_slices
    
    def data_slice_by_day2(self,data):
        """return a list of data frame from slicing origin data frame by day"""
        #TODO
        data_slices = []
        start_time = data['time'].min()
        end_time = data['time'].max()
        slice_start_time = start_time.replace(hour=0,minute=0,second=0)
        while(slice_start_time < end_time):
            d = self.data_bytime2(data,slice_start_time,slice_start_time + self.__one_day__)
            if(not d.empty):
                data_slices.append(d)
            slice_start_time = slice_start_time + self.__one_day__
        return data_slices
    
    def data_slice_by_sep(self,sep_time=pd.to_timedelta(600,unit='s'),side='both'):
        """return a list of data"""
        data_slices = []
        data_rows = self.__data.shape[0]
        measure_time = self.__data.time
        delta_time = list(measure_time.diff()) # use list to accel iteration
        last_sep_loc = 0
        for i in range(1,data_rows):
            if(delta_time[i] > sep_time):
                data_slices.append(self.__data.iloc[last_sep_loc:i])
                last_sep_loc = i
        if(i - last_sep_loc != 0):
            data_slices.append(self.__data.iloc[last_sep_loc:i])
        return data_slices
#            df = pd.DataFrame({'time':time,'tm1_L':tm1_L,'HR_L':HR_L,'tm2_L':tm2_L,
#                           'tm1_R':tm1_R,'HR_R':HR_R,'tm2_R':tm2_R},
#            columns = ['time','tm1_L','HR_L','tm2_L','tm1_R','HR_R','tm2_R'])

    def adjust_dataframe(self,dt):
#        print dt
        del dt['bid']
        del dt['devicePart']
        del dt['measure_time']
        del dt['server_record_time']
#        del dt.
        dt.rename(columns={'humidity':'HR','temperature':'tm1',
                           'room_temperature':'tm2'},inplace = True)
        return dt
    
    def manage_day_data_two_side(self):
        '''     '''
        left_dt = copy.deepcopy(self.data_left)
        right_dt = copy.deepcopy(self.data_right)
#        left_lst=data_slice_by_day2(left_dt)
#        right_lst=data_slice_by_day2(right_dt)
        left_dt = self.adjust_dataframe(left_dt)
        right_dt = self.adjust_dataframe(right_dt)
#        print
#        print left_dt
        
        left_lst=self.data_slice_by_day2(left_dt)
        right_lst=self.data_slice_by_day2(right_dt)
#        print left_lst
        
        return left_lst,right_lst
#        columns = ['time','tm1_L','HR_L','tm2_L','tm1_R','HR_R','tm2_R']
#        data = data_slice_by_day2(self)
        
#%%  
class DataCleanerFactory(object):
    TM0 = 30
    sep_seconds = 600 
    #sep_time=pd.to_timedelta(600,unit='s')
    sigma_threshold = 2
    diff_threshold = 0.0035
    length_threshold_multiplier = 3
    humidity_threshold = 30
    #length_threshold = 10
    
    def __init__(self,parameters={}):
        for k,v in parameters.items():
            self.k = v
        self.sep_time = pd.to_timedelta(self.sep_seconds,unit='s') # one minutes
        self.length_threshold = self.sep_time * self.length_threshold_multiplier
        return
    
    def clean_single_side(self,single_side_data,fun=lambda x:x):
        """clean a single side data to list of data frames
        using temperature threshold, n-sigma threshold, gradient threshold and
        humidity threshold. then re-cut data into pieces and drop small pieces"""
        #remove temperature below 30
        data = single_side_data[ single_side_data.temperature > self.TM0 ]
        
        #if too small
        if(len(data) < 10):
            log.warning("Input data is TOO Small")
            return None
        
        #cut into slices
        data_slices = []
        data_rows = data.shape[0]
        measure_time = data.time
        delta_time = list(measure_time.diff()) # use list to accel iteration
        last_sep_loc = 0
        for i in range(1,data_rows):
            if(delta_time[i] > self.sep_time):
                data_slices.append(data.iloc[last_sep_loc:i])
                last_sep_loc = i
        if(i - last_sep_loc != 0):
            data_slices.append(data.iloc[last_sep_loc:i])
        
        # n-sigma
        new_data_slices = []
        for x in data_slices:
            #print x
            mu = x.temperature.mean()
            sigma = x.temperature.std()
            z = (x.temperature - mu) / sigma
            new_data_slices.append(x[z.between(-self.sigma_threshold,self.sigma_threshold)])
        data_slices = new_data_slices
        
        #remove small slices less than 3x length threshold
        data_slices = [pieces for pieces in data_slices 
                       if pieces.time.max()-pieces.time.min()  >self.length_threshold]
        
        
        #gradient(the micro quotient)
        new_data_slices = []
        for x in data_slices:
            diff_vec = x.temperature.diff(2).shift(-1)
            diff_time = x.time.diff(2).shift(-1)
            diff_vec = diff_vec / [t.seconds for t in diff_time]
            diff_vec.iloc[-1] = diff_vec.iloc[-2]
            diff_vec.iloc[0] = diff_vec.iloc[1]
            new_data_slices.append(
                    x[diff_vec.between(-self.diff_threshold,
                                       self.diff_threshold)])
        data_slices = new_data_slices
        
        #humidity > 30
        new_data_slices = []
        for x in data_slices:
            new_data_slices.append(x[x.humidity > self.humidity_threshold])
        
        
        #recut into slices
#        print data_slices
        if(data_slices):
            data_slices = pd.concat(data_slices)
        else:
            return None
        data_rows = data.shape[0]
        measure_time = data.time
        delta_time = list(measure_time.diff()) # use list to accel iteration
        last_sep_loc = 0
        for i in range(1,data_rows):
            if(delta_time[i] > self.sep_time):
                data_slices.append(data.iloc[last_sep_loc:i])
                last_sep_loc = i
        if(i - last_sep_loc != 0):
            data_slices.append(data.iloc[last_sep_loc:i])
        data_slices = new_data_slices
        
        #remove small slices less than 3x length threshold
        data_slices = [pieces for pieces in data_slices 
                       if pieces.time.max()-pieces.time.min()  >self.length_threshold]
        
        #return list of data frames
        data_slices = fun(data_slices)
        return data_slices
    
    def interpolation(self,t1,v1,t2,v2,t3):
        if(t1 == t3):
            return v1
        if(t2 == t3):
            return v2
        return v1 + (v2-v1) / (t2 - t1).seconds * (t3 - t1).seconds
    
    def interpolate_column(self,time_column,value_column,i,time):
        return self.interpolation(time_column.iat[i-1],value_column.iat[i-1],
                                  time_column.iat[i],value_column.iat[i],time)
    
    def merge(self,df_left,df_right,start,end):
        time = []
        tm1_L = []
        HR_L = []
        tm2_L = []
        tm1_R = []
        HR_R = []
        tm2_R =[]
        one_minute = pd.to_timedelta(60,unit='s')
        t =start
        df_l_t = df_left.time
        df_r_t = df_right.time
        df_l_i = 0
        df_r_i = 0
#        df_l_t_max = df_l_t.max()
#        df_r_t_max = df_r_t.max()
        df_l_i_max = len(df_l_t) -1
        df_r_i_max = len(df_r_t) -1
        while t < end:
            while df_l_t.iat[df_l_i] <= t and df_l_i< df_l_i_max:
                df_l_i += 1
            
            while df_r_t.iat[df_r_i] <= t and df_r_i< df_r_i_max:
                df_r_i += 1
    
            time.append(t)
            
            #left side
            z = self.interpolate_column(df_l_t,df_left.temperature,df_l_i,t)
            tm1_L.append(z)
            z = self.interpolate_column(df_l_t,df_left.humidity,df_l_i,t)
            HR_L.append(z)
            z = self.interpolate_column(df_l_t,df_left.room_temperature,df_l_i,t)
            tm2_L.append(z)
            
            #right side
            z = self.interpolate_column(df_r_t,df_right.temperature,df_r_i,t)
            tm1_R.append(z)
            z = self.interpolate_column(df_r_t,df_right.humidity,df_r_i,t)
            HR_R.append(z)
            z = self.interpolate_column(df_r_t,df_right.room_temperature,df_r_i,t)
            tm2_R.append(z)
            
            t = t + one_minute
            
        df = pd.DataFrame({'time':time,'tm1_L':tm1_L,'HR_L':HR_L,'tm2_L':tm2_L,
                           'tm1_R':tm1_R,'HR_R':HR_R,'tm2_R':tm2_R},
            columns = ['time','tm1_L','HR_L','tm2_L','tm1_R','HR_R','tm2_R'])
        return df
    
    def align_merge_and_interpolation(self,df_list_L,df_list_R):
        if(df_list_L == None or df_list_R == None):
            log.warning("None")
            return None
        merged_list = []
        for df in df_list_L:
            df_time_min = df.time.min()
            df_time_max = df.time.max()

            for conjugate_df in df_list_R:
                conjugate_df_time_min = conjugate_df.time.min()
                conjugate_df_time_max = conjugate_df.time.max()
                #df overlaps conjugate_df 
                if(conjugate_df_time_min < df_time_max
                   and conjugate_df_time_max > df_time_min):
                    start = max(df_time_min,conjugate_df_time_min)
                    end = min(df_time_max,conjugate_df_time_max)
                    merged_list.append(self.merge(df,conjugate_df,start,end))
        return merged_list
        
#%% functional
def read_single_clean_split(filepath,phone_number):
    a = SingleUserData(filepath,phone_number)
    my_data_cleaner = DataCleanerFactory()
    left = my_data_cleaner.clean_single_side(a.data_left)
    right = my_data_cleaner.clean_single_side(a.data_right)
    aligned = my_data_cleaner.align_merge_and_interpolation(left,right) 
    return (phone_number,aligned)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% output window %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_multi_clean_split(filepath,phone_numbers):
    result = {}
    for i in range(len(phone_numbers)):
        log.debug(phone_numbers[i])
        f = filepath
        p = phone_numbers[i]
        (p,pd_list) = read_single_clean_split(f,p)
        result[p] = pd_list
    return result
    
#%% test code
if __name__ == "__main__":
    
    
    FILE_DIR = r'./data-10-17/'
    FILE_PATH = r'./data-10-17/18601774731 .csv'
    PHONENUMBER ="18601774731 "
    f = pd.read_csv(FILE_DIR + PHONENUMBER + '.csv')
    
    
    
#    f = pd.read_csv(FILE_DIR + PHONENUMBER + '.csv')
#    FILE_DIR = r'./Redo/'
#    FILE_PATH = r'./Redo/13701602184.csv'
#    PHONENUMBER ="13956840600"
#    PHONENUMBER ="18801615792"
#    f = pd.read_csv(FILE_DIR + PHONENUMBER + '.csv')
#    def unit_test():
#        a = SingleUserData(FILE_DIR,PHONENUMBER)
#        my_data_cleaner = DataCleanerFactory({'sep_time':600})
#        left = my_data_cleaner.clean_single_side(a.data_left)
#        right = my_data_cleaner.clean_single_side(a.data_right)
#        aligned = my_data_cleaner.align_merge_and_interpolation(left,right) 
#        return aligned
#    unit_test()
#    a = SingleUserData(FILE_DIR,PHONENUMBER)
#    import cProfile
#    
#    my_data_cleaner = DataCleanerFactory({'sep_time':600})
#    left = my_data_cleaner.clean_single_side(a.data_left)
#    right = my_data_cleaner.clean_single_side(a.data_right)
#    
##    ts = pd.to_datetime('2016-01-13 23:00:00')
##    te = pd.to_datetime('2016-01-14 06:00:00')
##    left = my_data_cleaner.clean_single_side(a.data_bytime(ts,te,side='l'))
##    right = my_data_cleaner.clean_single_side(a.data_bytime(ts,te,side='r'))
##    
#    aligned = my_data_cleaner.align_merge_and_interpolation(left,right) 
##    
##    cProfile.run('my_data_cleaner.clean_single_side(a.data_left)')
##    cProfile.run('my_data_cleaner.align_merge_and_interpolation(left,right) ')
#    apd = pd.concat(aligned)
#    
#    f = open('merged.csv','w')
#    f.write(apd.to_csv())
#    f.close()
#    
#    
#    filedir = r'./Redo/'
#    read_single_clean_split(filedir,PHONENUMBER)
#    print 'single'
##%% test code
#    import os
#    filedir = r'./Redo/'
#    filelist = os.listdir(r'./Redo/')
#    filelist = [x for x in filelist if x.endswith('.csv')]
#    phone_numbers = [x.split('.')[0] for x in filelist]
#    results = read_multi_clean_split(filedir,phone_numbers)
#
#
##%% test code
#    PHONENUMBER = "13818108618"
#    PHONENUMBER = "18616834486"
#    filedir = r'./Redo/'
#    outfiledir = r'./cleaned/'
#    #read_single_clean_split(filedir,PHONENUMBER)
#    a = SingleUserData(filedir,PHONENUMBER)
#    my_data_cleaner = DataCleanerFactory()
#    left = my_data_cleaner.clean_single_side(a.data_left)
#    right = my_data_cleaner.clean_single_side(a.data_right)
#    aligned = my_data_cleaner.align_merge_and_interpolation(left,right) 
#    print 'single'
#    
#    z= ["18616834486","13818108618",'13818599899',
#        '13587768048','13002128060','13221986571','13588847596']
#    keys = [k for k,v in results.items() if v!=None and v!=[]]
#    for phone in keys:
#        with open(outfiledir+str(phone)+'.csv','w') as o:
#            if (len(results[phone])>1):
#                z = pd.concat(results[phone])
#            else:
#                print phone
#                z = results[phone][0]
#            o.write(z.to_csv())






