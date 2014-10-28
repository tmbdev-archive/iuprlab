import sys,os
import random as pyrand
import pylab,numpy
from types import *
from math import *
from numpy import *
from scipy import mgrid,linalg,ndimage

################################################################
# data reading
################################################################

def read_columns(file):
    columns = None
    stream = open(file,"r")
    lines = stream.xreadlines()
    for line in lines:
        fields = line.split()
        if columns==None: 
            columns = [[] for i in range(len(fields))]
        assert len(fields)==len(columns)
        for i in range(len(fields)):
            columns[i].append(fields[i])
    for i in range(len(columns)):
        try:
            columns[i] = array([float(x) for x in columns[i]])
        except:
            pass
    return columns

def read_hcolumns(file):
    columns = None
    stream = open(file,"r")
    lines = stream.xreadlines()
    header = lines.next().split()
    for line in lines:
        fields = line.split()
        if columns==None: 
            columns = [[] for i in range(len(fields))]
        assert len(fields)==len(columns)
        for i in range(len(fields)):
            columns[i].append(fields[i])
    for i in range(len(columns)):
        try:
            columns[i] = array([float(x) for x in columns[i]])
        except:
            columns[i] = array(columns[i])
    assert len(header)==len(columns)
    result = {}
    for i in range(len(header)):
        result[header[i]] = columns[i]
    return result

################################################################
# output data in "SVM" storage format
################################################################

def write_svm(file,images,labels):
    stream = open(file,"w")
    lmax = amax(labels)
    for index in range(len(images)):
        label = labels[index]
        if label==0: lmax+10
        stream.write("%d"%label)
        input = images[index].ravel()
        for i in range(len(input)):
            stream.write(" %d:%.4g"%(i+1,input[i]))
        stream.write("\n")
    stream.close()

################################################################
# use HDF5 tables to store arrays quickly and easily
################################################################

class HDF5Store:
    def __init__(self,file=None,mode="r",title=None):
        self.h5 = 1
        if file!=None:
            self.open(file,mode,title=title)
    def open(self,file,mode,title=None):
        if not title: title = file
        self.h5 = tables.openFile(file,mode=mode,title=title)
    def close(self):
        self.h5.close()
    def __getattr__(self,item):
        value = self.h5.getNode(self.h5.root,item)
        self.__dict__[item] = value
        return value
    def __delattr__(self,item):
        try: self.h5.removeNode(self.h5.root,item)
        except tables.exceptions.NoSuchNodeError: pass
    def __setattr__(self,item,value):
        if item=="h5":
            self.__dict__["h5"] = value
            return
        try: self.h5.removeNode(self.h5.root,item)
        except tables.exceptions.NoSuchNodeError: pass
        if type(value)==type([]): value = array(value)
        result = self.h5.createArray(self.h5.root,item,value,item)
        self.__dict__[item] = result
        self.h5.flush()
    def __str__(self):
        return str(self.h5)
    def __repr__(self):
        return repr(self.h5)

