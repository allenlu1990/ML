import os, shutil
import logging
log=logging.getLogger('estimator')

# ROOTDIR=os.getcwd()
ROOTDIR = os.path.join(os.getcwd(),"ML2/workers")
csvDIR=os.path.join(ROOTDIR, 'Algorithm')
def estimatorRUN(id,data2d):# length, dev, measure_time, temp_edge,humid, temp_cent):
    NAME='{}.csv'.format(id)
    csvNAME=os.path.join(csvDIR, NAME)
    if os.path.exists(csvDIR):shutil.rmtree(csvDIR)
    os.makedirs(csvDIR)
    with open(csvNAME,'w') as tf:
#        line="bid,devicePart,measure_time,stemperature,humidity,room_temperature\n"
#        tf.write(line)
        for i in range(len(data2d)):
            line="{},{},{},{},{},{},{}\n".format(data2d[i][0], data2d[i][1],data2d[i][2],data2d[i][3],data2d[i][4],data2d[i][5],data2d[i][6])
            tf.write(line)
#    print "extimatorRUN:", ROOTDIR 
    mainNAME=os.path.join(ROOTDIR, 'main.py')
    command="python"+" "+mainNAME+" -rootdir "+ROOTDIR

##############################################################
    lines=os.popen(command).readlines()
    record=[]
    fills=True
    nrec=0
    for ls in lines:
        if ls.find("==RESULT-S==")>0: 
            fills=True
            continue
        if ls.find("==RESULT-E==")>0:break
        if fills:nrec=nrec+1
        if nrec>1 and len(ls)>0: record.append(ls)
    return '|'.join(record)
##############################################################        
            
#    os.system(command)
#    print "1111111111111111111"
#    #os.execfile(mainNAME)
#    log.info("finished! congratulations")
        
        
    
