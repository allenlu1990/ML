#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:33:31 2017

@author: peiji
"""

import logging
import logging.config as logconf
import datetime
#import pandas as pd
import os
# ROOTDIR=os.getcwd()
ROOTDIR = os.path.join(os.getcwd(),"ML2/workers")
LOG_CONFIG_FILE='logging.conf'
LOG_SAVE_FILE='Prediction.log'


def produce_logconfig(logconfigfile, runmode="CRITICAL"):
    with open(logconfigfile, 'w') as lf:
        level_line='level={}\n'.format(runmode)
        lf.write('[loggers]\n')
        lf.write('keys=root,main\n\n')
        
        lf.write('[handlers]\n')
        lf.write('keys=consoleHandler,fileHandler\n\n')
        
        lf.write('[formatters]\n')
        lf.write('keys=fmt\n\n')
        
        lf.write('[logger_root]\n')
        lf.write(level_line)
        lf.write('handlers=consoleHandler\n\n')
        
        lf.write('[logger_main]\n')
        lf.write(level_line)
        lf.write('qualname=main\n')
        lf.write('handlers=fileHandler\n\n')
        
        lf.write('[handler_consoleHandler]\n')
        lf.write('class=StreamHandler\n')
        lf.write(level_line)
        lf.write('formatter=fmt\n')
        lf.write('args=(sys.stdout,)\n\n')
        
        lf.write('[handler_fileHandler]\n')
        lf.write('class=logging.handlers.RotatingFileHandler\n')
        lf.write(level_line)
        lf.write('formatter=fmt\n')
        lf.write('args=(\'{}\',\'a\',20000,5,)\n\n'.format(LOG_SAVE_FILE))
        
        lf.write('[formatter_fmt]\n')
        lf.write('format=%(name)s-%(levelname)s-%(message)s\n\n')

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-rootdir', type=str, default = None)
args = parser.parse_args()
ROOTDIR=args.rootdir
LOGFILE=os.path.join(ROOTDIR, LOG_CONFIG_FILE)
if not os.path.exists(LOGFILE):
    produce_logconfig(LOGFILE,"DEBUG")
logconf.fileConfig(LOGFILE)
#print ROOTDIR
log=logging.getLogger('main.entrance')
    
def ENTRANCE():
    Start = datetime.datetime.now()
    log.debug(str(Start))
    FILEDIR=os.path.join(ROOTDIR, 'Algorithm')
    FILEDIR=os.path.join(FILEDIR, '') ## add a '\' at the end of the path
    #%% preprocess 
    log.debug('~'*30+'preprocess start'+'~'*30)
    import preprocess as pre
    valid_dict, exception_user1 = pre.preprocess(FILEDIR).process()
#    print valid_dict
    log.debug('*'*30+'preprocess finish'+'*'*30)
    
    #%% Feature Mining 
    log.debug('~'*30+'Feature Mining start'+'~'*30)
    import FeatureMining as FM
    user_features,exception_user2 = FM.FeatureMining(valid_dict).calc_feature()
    log.debug('*'*30+'Feature Mining finish'+'*'*30)
        
    #%% predict
    log.debug('~'*30+'predict start'+'~'*30)
    full_model_name=os.path.join(ROOTDIR,'Model.mdl')
    from sklearn.externals import joblib
    model = joblib.load(full_model_name)
    import predict as pct
    c = pct.predict(model,user_features,exception_user1,exception_user2)
    score = c.score()
    log.debug('~'*30+'predict finish'+'~'*30)
        
    End = datetime.datetime.now()
    log.debug(str(End))
    second=(End-Start).seconds
    log.debug('Escape:'+str(second))
    return score

#if __name__=='__main__':
#
#    predict=main()
    #print predict
    
