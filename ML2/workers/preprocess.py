#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:35:44 2017

@author: peiji
"""

import os
import pandas as pd
import logging
log=logging.getLogger('main.entrance.preprocess')
class preprocess(object):

    def __init__(self,FILEDIR):
        self.FILEDIR =FILEDIR
        self.dict_temper = {}
        self.exception_user={}
        self.valid_dict = {}  ##output dictionary
        
    def clean(self,method='clean'):
        log.debug('user_list:')
        FILELIST = os.listdir(self.FILEDIR)
        filelist = [x for x in FILELIST if x.endswith('.csv')]
        phone_numbers = [x.split('.')[0] for x in filelist]
        if method=='clean':
            import breastdata as bd
            self.dict_temper = bd.read_multi_clean_split(self.FILEDIR,phone_numbers)
        elif method=='read':
            for user_id in phone_numbers:
                self.dict_temper[user_id] = pd.read_csv(self.FILEDIR+user_id+'.csv')

    def judge(self,phones):
        log.debug('user_list:')
        for user_id in phones:
            log.debug(user_id)
            Temper=self.dict_temper[user_id]
            if Temper is None or len(Temper) == 0: 
                self.exception_user[user_id]='PE'
                del self.dict_temper[user_id]
                continue

    def user_day_split(self):
        log.debug('user_list:')
        for user_id in self.dict_temper.keys():
            log.debug(user_id)
            Temper=self.dict_temper[user_id]
            if len(Temper)==1:
                self.valid_dict[user_id]=Temper[0]
            else:
                for day in range(len(Temper)):
                    self.valid_dict[user_id+'-'+str(day)]=Temper[day]
    
    def process(self): 
        log.debug(''*20+'clean start'+'-'*20)
        self.clean()
        log.debug('-'*20 + 'clean end'+'-'*20)
        log.debug('')
        
        phones = self.dict_temper.keys()
        log.debug('-'*20+'judge start'+'-'*20)
        self.judge(phones)
        log.debug('-'*20+'judge finish'+'-'*20)
        log.debug('')
        
        log.debug('-'*20+'split start'+'-'*20)
        self.user_day_split()
        log.debug('-'*20+'split finish'+'-'*20)
        log.debug('')
        return self.valid_dict,self.exception_user
