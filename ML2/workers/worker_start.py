import os, shutil
import logging

from ML2.settings import BASE_DIR

log = logging.getLogger('estimator')
ROOTDIR = os.path.join(os.getcwd(),"ML2/workers")
csvDIR = os.path.join(ROOTDIR, 'Algorithm')

def run(path):  # length, dev, measure_time, temp_edge,humid, temp_cent):

    # NAME = '{}.csv'.format(id)
    file_name = os.path.basename(path)
    csvNAME = os.path.join(csvDIR, file_name)
    if os.path.exists(csvDIR): shutil.rmtree(csvDIR)
    os.makedirs(csvDIR)
    with open(csvNAME, 'w') as tf:
        for line in open(path).readlines():
            splits = line.split(",")
            write_line = "{},{},{},{},{},{},{}".format(splits[0],splits[1],splits[2],splits[3],splits[4],splits[5],splits[6])
            tf.write(write_line)

    mainNAME = os.path.join(ROOTDIR, 'main.py')
    command = "python" + " " + mainNAME + " -rootdir " + ROOTDIR

    ##############################################################
    lines = os.popen(command).readlines()
    record = []
    fills = True
    nrec = 0
    for ls in lines:
        if ls.find("==RESULT-S==") > 0:
            fills = True
            continue
        if ls.find("==RESULT-E==") > 0: break
        if fills: nrec = nrec + 1
        if nrec > 1 and len(ls) > 0: record.append(ls)
    return '|'.join(record)
##############################################################


if __name__ == '__main__':
    print BASE_DIR
    me = run("/Users/alu/CodeRepo/PythonCode/ML2/static/pic/13917596968.csv")
    print me




