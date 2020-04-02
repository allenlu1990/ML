#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:37:46 2017

@author: peiji
"""

import SingleUserFeatures as suf
import pandas as pd
#import copy
import logging
log=logging.getLogger('main.entrance.FeatureMining')
class FeatureMining(object):

    def __init__(self,valid_dict):
        self.valid_dict = valid_dict
        self.exception_user = {}
        self.user_features = pd.DataFrame()

    def calc_feature(self)  :   
        user_num = 0
        log.debug('user_list:')
        for user_id in self.valid_dict.keys():
            log.debug(user_id)
            Temper=self.valid_dict[user_id]
            state, feat_df = suf.cac_all(user_id,Temper)
            if not state:
                self.exception_user[user_id]= 'CE'
                continue
            if user_num==0:
                self.user_features = feat_df
            else:
                self.user_features = self.user_features.append(feat_df) 
            user_num = user_num+1
        return self.user_features,self.exception_user
        
