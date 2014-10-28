__all__ = ["BoostedMLP"]

import os,sys,os.path,re,string,math
from pylab import *
from numpy import *
import mlp
import pickle

verbose = 0

def finite(x):
    return not isnan(x).any() and not isinf(x).any()

def perplexity(weights):
    weights = weights/sum(weights)
    return exp(-sum(weights*where(weights>0,log(weights),0.0)))

def weighted_sample(weights,n):
    weights = weights * 1.0 / sum(weights)
    weights = cumsum(weights)
    return searchsorted(weights,random.uniform(size=n))

def rowwise(f,data):
    n,d = data.shape
    l = [f(data[i]) for i in range(n)]
    return array(l)

class BoostedMLP:
    def __init__(self):
        self.list = None
    def train(self,data,cls,nclass,nstages=30,nhidden=None,
              eta=1.0,epochs=10,nsample=10000):
        assert self.list is None
        n,d = data.shape
        assert n==len(cls)
        assert (cls>=0).all()
        assert (cls<nclass).all()
        self.nclass = nclass
        weights = 1.0/(1+random.permutation(len(data)))
        weights /= sum(weights)
        list = []
        for i in range(nstages):
            if verbose: print "round",i
            net = mlp.MLP()
            # train on weighted sample
            samples = weighted_sample(weights,n=nsample)
            net.train(data,cls,nclass,samples=samples)
            # compute error on entire set
            pred = net.classify(data)
            if verbose:
                print "    err=",sum(pred!=cls)
                print "    werr=",sum((pred!=cls)*weights)/sum(weights)
                print "    sample perplexity=",perplexity(weights)
            err = sum((pred!=cls)*weights)/sum(weights)
            # SAMME update for multiclass boosting
            alpha = log((1.0-err)/err) + log(nclass-1.0)
            weights = weights*exp(alpha*(pred!=cls))
            weights /= sum(weights)
            list.append((net,alpha,weights,err))
        self.list = list
    def discriminants1(self,v,limit=9999):
        d = self.nclass
        totals = zeros(d)
        for index in range(min(limit,len(self.list))):
            comp = self.list[index]
            net = comp[0]
            alpha = comp[1]
            c = net.classify(v.reshape(1,len(v)))[0]
            totals[c] += alpha
        return totals
    def classify1(self,v,limit=9999):
        return argmax(self.discriminants1(v,limit=limit))
    def discriminants(self,data):
        return rowwise(self.discriminants1,data)
    def classify(self,data):
        return rowwise(self.classify1,data)
    def save(self,stream):
        pickle.dump((self.list,self.nclass),stream,protocol=2)
    def load(self,stream):
        self.list,self.nclass = pickle.load(stream)

class StackedMLP(BoostedMLP):
    def __init__(self):
        BoostedMLP.__init__(self)
        self.nclass = None
        self.stacked = None
    def train(self,data,cls,nclass,nstages,
              nhidden,eta=1.0,epochs=10,nsample=10000,
              snhidden=None,seta=1.0,sepochs=10):
        assert self.stacked is None
        if snhidden==None: snhidden = 3*nclass
        BoostedMLP.train(self,data,cls,nclass,
                         nstages=nstages,nhidden=nhidden,
                         eta=eta,epochs=epochs,nsample=nsample)
        sdata = rowwise(self.all_discriminants1,data)
        assert finite(sdata)
        # mlp.verbose = 1
        net = mlp.MLP()
        net.train(sdata,cls,nclass,nhidden=snhidden,eta=seta,epochs=sepochs)
        self.stacked = net
        self.nclass = nclass
    def all_discriminants1(self,v):
        assert self.list is not None
        assert v.ndim==1
        result = zeros((self.nclass * len(self.list)))
        i = 0
        for l in self.list:
            net = l[0]
            d = net.discriminants(v.reshape(1,len(v)))[0]
            result[i:i+len(d)] = d
            i += len(d)
        assert finite(d)
        return result
    def discriminants1(self,v):
        ps = self.all_discriminants1(v)
        assert finite(ps)
        result = self.stacked.posteriors(ps.reshape(1,len(ps)))[0]
        assert finite(result)
        return result
    def posteriors1(self,v):
        result = self.discriminants1(v)
        result /= max(1.0,sum(result))
        assert finite(result)
        return result
    def classify1(self,v):
        return argmax(self.discriminants1(v))
    def discriminants(self,data):
        assert finite(data)
        result = rowwise(self.discriminants1,data)
        assert finite(result)
        return result
    def posteriors(self,data):
        assert finite(data)
        result = rowwise(self.posteriors1,data)
        assert finite(result)
        return result
    def classify(self,data):
        assert finite(data)
        return rowwise(self.classify1,data)
    def save(self,stream):
        pickle.dump((self.list,self.nclass,self.stacked),stream,protocol=2)
    def load(self,stream):
        self.list,self.nclass,self.stacked = pickle.load(stream)

import unittest,fpectl
from test_classifier import *

class TestBoostedMLP(TestBatchClassifier):
    params = {"nstages":3,"epochs":1,"nsample":10}
    factory = BoostedMLP

class TestStackedMLP(TestBatchClassifier):
    params = {"nstages":3,"epochs":1,"nsample":10,"sepochs":1,"nhidden":3,"snhidden":3}
    factory = StackedMLP

if __name__ == "__main__":
    unittest.main()

def test_verbose():
    fpectl.turnon_sigfpe()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBoostedMLP)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStackedMLP)
    unittest.TextTestRunner(verbosity=1).run(suite)
