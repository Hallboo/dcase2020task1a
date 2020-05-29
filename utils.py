import os
import re
from hparams import hparams

def GetUttID(the_path):
    # airport-barcelona-0-4-a
    # [scene label]-[city]-[location id]-[segment id]-[device id].wav
    return re.findall(r'[^a-z]*([a-z_]+-[a-z_]+-[0-9]+-[0-9]+-[a-z][0-9]?).*', the_path)[0]

def getClassName(the_index):
    return hparams.labels[the_index]

def getDevice(label_str):
    return re.findall(r'[a-z_]+-[a-z_]+-[0-9]+-[0-9]+-([a-z][0-9]?).*', label_str)[0]

def get_label2index():
    label2index=dict()
    for i in range(len(hparams.labels)):
        label2index[hparams.labels[i]] = i
    return label2index

def get_index2label():
    index2label=dict()
    for i in range(len(hparams.labels)):
        index2label[i] = hparams.labels[i]
    return index2label

def get_indexOfDevice(dev_str):
    return hparams.devices.index(dev_str)

def get_numDevice():
    return len(hparams.devices)

if __name__ == "__main__":
    # print(GetUTTID("datasets/feature/airport-barcelona-0-6-a-R.npy"))
    # print(getClassName(1))
    for k,v in get_label2index().items():
        print(k,v)
    
    for k,v in get_index2label().items():
        print(k,v)
