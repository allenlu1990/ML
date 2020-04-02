#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:38:20 2017

@author: peiji
"""
import pandas as pd
import numpy as np
#import 
import logging
log=logging.getLogger('main.entrance.predict')
class predict(object):  
    def __init__(self,model,user_features,exception_user1,exception_user2):
        self.clf = model['clf']
        self.user_features = user_features
        self.best_feature_lst = model['select_feature']
        self.patient_dt = pd.DataFrame()
        self.select_features = pd.DataFrame()
        self.Xtotal=[]
        self.ID_list=[]
        self.y_predict=[]
        self.y_predict_prob=[]
        self.exception_user = dict(exception_user1,**exception_user2)
        self.thresold = [0.5,0.7,0.9,0.99]
        
    def avg_score(self):
        patient_lst = self.score.index.tolist()
        real_lst = [x.split('-')[0] for x in patient_lst]
        self.score['phone'] = real_lst
        grouped = self.score.groupby('phone')
        return grouped.mean()
    
    def arrange_score(self):
        grade =[]
        for item in self.score['score']:
            if item<=self.thresold[0] and item>=0:
                grade.append(1)
            elif item>=self.thresold[0] and item<self.thresold[1]:
                grade.append(2)
            elif item>=self.thresold[1] and item<self.thresold[2]:
                grade.append(3)
            elif item>=self.thresold[2] and item<=self.thresold[3]:
                grade.append(4)
            else:
                grade.append(-1)

        self.score['grade'] = grade

    def solve_exception(self):
        self.exception_user = pd.DataFrame.from_dict(self.exception_user,orient='index')
        self.exception_user = self.exception_user.rename(columns={0:'score'})
        for ids in self.exception_user.index.tolist():
            if ids.split('-')[0] not in [x.split('-')[0] for x in self.score.index.tolist()]:
                self.score=self.score.append(self.exception_user.loc[ids].rename(ids.split('-')[0]))
                
    def score(self):
        
        self.select_features = self.user_features[self.best_feature_lst]
        self.Xtotal = self.select_features.values.tolist()
        self.ID_list = self.select_features.index.tolist()
        self.y_predict = self.clf.predict(self.Xtotal)
        y_predict_prob =  self.clf.predict_proba(self.Xtotal)
        self.y_predict_prob = y_predict_prob[:,1]
        y_predict_prob_round = np.round(self.y_predict_prob,4)
        self.score = pd.DataFrame(data=y_predict_prob_round,index=self.ID_list,columns=['score'])
#        self.score= self.avg_score()
        self.solve_exception()
        self.arrange_score()
        return self.score
    

