import time
import secrets
import numpy
import os
import util
import numpy as np
import math

class DiskLogger:
    def __init__(self,folder,checkpointTime = None):
        self.folder = folder
        milis = str(math.floor(time.time()*10000))[-7:]
        self.instanceNumber = secrets.token_hex(15)+milis
        self.workingData = {}
        self.checkpointTime = checkpointTime
        self.startTime = time.time()

    def write(self,k,v):
        envoke_time = time.time()
        self.workingData[k]=v
        if self.checkpointTime and (envoke_time - self.startTime > self.checkpointTime):
            milis = str(math.floor(time.time()*10000))[-7:]
            self.instanceNumber = secrets.token_hex(15)+milis
            self.startTime = envoke_time
        util.ensure_folders(f'{self.folder}/{self.instanceNumber}')
        print(self.instanceNumber)
        np.save(f'{self.folder}/{self.instanceNumber}',self.workingData)
        
class DiskReader:
    def __init__(self,folder):
        self.folder = folder

    # shallow merge from all files by modify date
    def data(self):
        if not os.path.exists(self.folder):
            return {}
        files = os.popen(f'ls -tr {self.folder}/').read().strip().split('\n')
        obj = {}
        for f in files:
            # print("read ",f)
            if os.path.getsize(f'{self.folder}/{f}') == 0: continue
            obj.update(np.load(f'{self.folder}/{f}',allow_pickle=True)[()])
        return obj

if __name__ == '__main__':
    reader = DiskReader('./navigation_results/base_spl_slam')
    writer = DiskLogger('data_dump/dump')
    writer.write(1,2)
    print(reader.data())

