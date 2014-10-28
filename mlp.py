from __future__ import with_statement

__all__ = ["MLP"]

import os,sys,os.path,re,math
from numpy import *
from pylab import *
from scipy import *
from utils import *

verbose_examples = 0

def show_examples(classifier,data,cls,samples=None,n=30):
    if not verbose_examples: return
    print "   ",take(cls,samples[:30])
    print "   ",rowwise(classifier.classify,take(data,samples[:n],axis=0))

class MLP:
    def __init__(self):
        self.trained = None
        self.w1 = None
        self.verbose = 0
    def check_finite(self):
        "Ensure all weights are finite."
        assert finite(self.w1)
        assert finite(self.b1)
        assert finite(self.w2)
        assert finite(self.b2)
    def error_rate(self,data,cls,samples=None):
        "Compute the classificatino error for the given sample."
        n,d = data.shape
        if samples is None: samples = arange(n)
        errs = 0
        for i in samples:
            pred = self.classify(data[i])
            if pred!=cls[i]: errs += 1
        return errs * 1.0/len(samples)
    def output_error(self,data,cls,samples=None):
        "Compute the output error over the given sample."
        n,d = data.shape
        if samples is None: samples = arange(n)
        oerrs = 0
        errs = 0
        for i in samples:
            pred = self.forward(data[i])
            oerrs += dist(pred,unary(cls[i],self.nclass()))
            errs += (argmax(pred)!=cls[i])
        return oerrs*1.0/len(samples),errs*1.0/len(samples)
    def train_epoch(self,data,cls,samples=None,eta=None):
        "Train one epoch."
        n,d = data.shape
        self.set_learning_rate(eta)
        if samples is None: samples = range(n)
        for i in samples:
            self.train1(data[i],cls[i])
        self.check_finite()
    def train_epochs(self,data,cls,train=None,epochs=30,eta=0.1,eta1=None):
        "Train multiple epochs."
        n,d = data.shape
        if train is None: train = arange(n)
        if eta1 is None: eta1 = eta/10.0
        assert eta>eta1
        assert eta1>0
        if epochs==1: etas = [eta]
        else: etas = [10**r for r in frange(log10(eta),log10(eta1),npts=epochs)]
        assert finite(etas)
        epoch = 0
        for eta in etas:
            self.train_epoch(data,cls,eta=eta,samples=train)
            oerr,err = self.output_error(data,cls,samples=train)
            if self.verbose: print "[epoch %d] oerr=%g err=%g eta=%g"%(epoch,oerr,err,eta)
            show_examples(self,data,cls,train)
            epoch += 1
        self.check_finite()
    def train_cv(self,data,cls,train=None,test=None,eta=0.1,factor=0.9,auto=None,frac=0.1,
                 min_epochs=10,max_fail=5,min_improve=0.99):
        "Train with cross-validation."
        n,d = data.shape
        if train is None:
            train = arange(n)
        if test is None:
            ntest = int(n*frac)
            train = train.copy()
            random.shuffle(train)
            test = train[:ntest]
            train = train[ntest:]
        # assert disjoint_samples(data,train,test)
        epoch = 0
        best_net = None
        best_oerr = 1e38
        fail = 0
        while epoch<min_epochs or fail<max_fail:
            if auto: eta = self.find_learning_rate(data,cls,samples=test[:1000])
            self.train_epoch(data,cls,eta=eta,samples=train)
            oerr,err = self.output_error(data,cls,samples=train)
            toerr,terr = self.output_error(data,cls,samples=test)
            if self.verbose:
                print "[epoch %d] train: oerr=%g err=%g "+\
                    "test: oerr=%g err=%g eta=%g *%d"% \
                      (epoch,oerr,err,toerr,terr,eta,fail)
            show_examples(self,data,cls,test)
            if toerr<=min_improve*best_oerr:
                best_oerr = toerr
                best_net = self.copy()
                fail = 0
            else:
                fail += 1
            epoch += 1
            eta *= factor
        self.set(best_net)
    def find_learning_rate(self,data,cls,samples=None,
                           min_rate = 1e-4,max_rate = 1e0):
        "Find a good learning rate by sampling."
        logstep = 0.5
        etas = [10**r for r in frange(log10(min_rate),log10(max_rate),logstep)]
        if samples is None: samples = arange(len(data))
        deltas = []
        for eta in etas:
            aux = self.copy()
            before,_ = aux.output_error(data,cls,samples=samples)
            aux.train_epoch(data,cls,eta=eta,samples=samples)
            after,_ = aux.output_error(data,cls,samples=samples)
            delta = before-after
            # print delta,eta,len(samples)
            deltas.append((delta,eta))
        return max(deltas)[1]
    def ninput(self):
        "Dimension of input vector."
        return self.w1.shape[1]
    def nclass(self):
        "Number of output classes."
        return self.w2.shape[0]
    def classify(self,data):
        "Classify the given input vector."
        if self.w1 is None: raise Exception("must train first")
        return rowwise(lambda x:argmax(self.forward(x)),data)
    def discriminants(self,data):
        "Compute discriminant values."
        if self.w1 is None: raise Exception("must train first")
        return rowwise(self.forward,data)
    def posteriors(self,data):
        "Compute posterior probabilities."
        if self.w1 is None: raise Exception("must train first")
        result = rowwise(self.forward,data)
        n,d = result.shape
        result /= sum(result,axis=1).reshape(n,1)
        return result
    def train1(self,x,cls):
        "Train one sample."
        self.backward(x,unary(cls,self.nclass()))
    def copy(self):
        "Clone the network"
        result = MLP()
        result.set(self)
        return result
    def info(self):
        "Provide some information about the network."
        return [self.shape,
                min(amin(self.w1),amin(self.b1),amin(self.w2),amin(self.b2)),
                max(amax(self.w1),amax(self.b1),amax(self.w2),amax(self.b2))]
    def train(self,data,cls,nclass,nhidden=None,samples=None,
              epochs=10,eta=0.1,eta1=None):
        "Batch train on data."
        if self.trained: raise Exception("can train only once")
        n,d = data.shape
        if nhidden is None: nhidden = 3*nclass
        if not self.w1: self.create(d,nhidden,nclass)
        self.train_epochs(data,cls,train=samples,epochs=epochs,eta=eta,eta1=eta1)
        self.trained = 1
    def create(self,n1,n2,n3,eps=1.0):
        "Create a network with the given topology."
        self.w1 = random.uniform(-eps,eps,(n2,n1))
        self.b1 = random.uniform(-eps,eps,(n2,))
        self.w2 = random.uniform(-eps,eps,(n3,n2))
        self.b2 = random.uniform(-eps,eps,(n3,))
        self.eta = 0.1
        return self
    def set_net(self,w1,b1,w2,b2):
        "Set up a network with the given weights."
        assert w1.shape[0]==len(b1)
        assert w2.shape[0]==len(b2)
        assert w1.shape[0]==w2.shape[1]
        self.shape = (w1.shape[1],len(b1),len(b2))
        self.w1 = w1.copy()
        self.b1 = b1.copy()
        self.w2 = w2.copy()
        self.b2 = b2.copy()
    def set(self,other):
        "Set the weights in this network to be equal to the other network"
        self.set_net(other.w1,other.b1,other.w2,other.b2)
        self.set_learning_rate(other.eta)
    def set_learning_rate(self,eta):
        "Change the learning rate."
        self.eta = eta
    def forward(self,x):
        "Forward propagation step."
        hidden = sigmoid(dot(self.w1,x) + self.b1)
        output = sigmoid(dot(self.w2,hidden) + self.b2)
        assert finite(output)
        return output
    def backward(self,x,target):
        "Backward propagation step."
        assert amin(x)>-10 and amax(x)<10
        eta = self.eta
        hidden = sigmoid(dot(self.w1,x) + self.b1)
        output = sigmoid(dot(self.w2,hidden) + self.b2)
        delta2 = (output-target) * dsigmoidy(output)
        delta1 = dot(delta2,self.w2).transpose() * dsigmoidy(hidden)
        self.w2 -= outer(eta*delta2,hidden) # speedup: pull eta inside outer()
        self.w1 -= outer(eta*delta1,x)
        self.b2 -= eta * delta2
        self.b1 -= eta * delta1
        return output
    def save(self,stream):
        "Save the network to the stream."
        if isinstance(stream,basestring):
            with open(stream,"w") as stream: self.load(stream)
        self.w1.dump(stream)
        self.b1.dump(stream)
        self.w2.dump(stream)
        self.b2.dump(stream)
    def load(self,stream):
        "Load the network from the stream."
        if isinstance(stream,basestring):
            with open(stream) as stream: self.load(stream)
        self.w1 = load(stream)
        self.b1 = load(stream)
        self.w2 = load(stream)
        self.b2 = load(stream)

import unittest
from test_classifier import *

class TestMLP(TestBatchClassifier):
    factory = MLP

if __name__ == "__main__":
    unittest.main()
