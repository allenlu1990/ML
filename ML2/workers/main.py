from  entrance import *
import os, shutil


LOG_CONFIG_FILE='logging.conf'
LOG_SAVE_FILE='Prediction.log'

# ROOTDIR=os.getcwd()
ROOTDIR = os.path.join(os.getcwd(),"ML2/workers")
csvDIR=os.path.join(ROOTDIR, 'Algorithm')

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
        lf.write('format=%(levelname)s:%(name)s-%(message)s\n\n')



if  __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-rootdir', type=str, default = None)
    args = parser.parse_args()
    ROOTDIR=args.rootdir
    LOGFILE=os.path.join(ROOTDIR, LOG_CONFIG_FILE)
    if not os.path.exists(LOGFILE):
       produce_logconfig(LOGFILE,"DEBUG")
    logconf.fileConfig(LOGFILE)
    
    log=logging.getLogger('entrance')
 
    predit=ENTRANCE()

    print "========RESULT-S========="
    print predit.shape
    print predit
    print "========RESULT-E========="
    print "---------END-------------"
    if os.path.exists(csvDIR):shutil.rmtree(csvDIR)
    
