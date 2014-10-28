__all__ = ["Stump1", "BoostedStumps"]

import numpy
from pylab import *
from numpy import *

verbose = 0

def rowwise(f,data):
    l = [f(data[i]) for i in range(len(data))]
    return array(l)

def opt_thresh(vals,weights,cls,internal=False):
    """Finds the optimal threshold (val>=thresh) for classification
    of vals into cls, with errors weighted by weights.  Returns the
    error rate and the threshold. Note: this works correctly
    even if values are repeated."""
    assert len(vals)==len(weights) and len(vals)==len(cls)
    indexes = argsort(vals)
    vals = vals[indexes]
    weights = weights[indexes]
    cls = cls[indexes]
    cls = 0+(cls==1)
    c0 = (cls*weights).cumsum()
    c0 = concatenate(([0],c0))
    c1 = ((1-cls)*weights)[::-1].cumsum()[::-1]
    c1 = concatenate((c1,[0]))
    vals = concatenate((vals,[inf]))
    total = c0+c1
    n = len(c0)
    i = 0
    thresh = vals[i]
    mi = 0
    mv = total[mi]
    while i<n:
        while i<n and vals[i]==thresh: i+=1
        if i<n:
            if total[i]<mv:
                mi = i
                mv = total[i]
            thresh = vals[i]
    err = mv*1.0/sum(weights)
    if mi>0: thresh = (vals[mi]+vals[mi-1])/2.0
    else: thresh = -inf
    return err,thresh

def stump1(data,weights,cls):
    n,d = data.shape
    assert n==len(weights) and n==len(cls)
    stumps = [opt_thresh(data[:,i],weights,cls) for i in range(d)]
    best = argmin([v[0] for v in stumps])
    err,thresh = stumps[best]
    return (best,err,thresh)

def stump(data,weights,cls):
    """Compute the optimal decision stump for weighted data points.
    Returns the sign (-1 meaning <, +1 meaning >), the threshold, and
    the dimension on which the split is being carried out."""
    n,d = data.shape
    assert n==len(weights) and n==len(cls)
    index,err,thr = stump1(data,weights,cls)
    index2,err2,thr2 = stump1(-data,weights,cls)
    if err<err2:
        return (1,thr,index),err
    else:
        return (-1,thr2,index2),err2

class Stump1:
    """A simple decision stump.  This picks one feature, one threshold,
    and one comparison direction and classifies based on that."""
    threshold = None
    def train(self,data,cls,nclass=2,weights=None):
        assert nclass==2
        assert self.threshold is None
        if weights is None: weights = ones(len(data))
        classifier,err = stump(data,weights,cls)
        self.sign,self.threshold,self.dim = classifier
        self.err = err
        assert self.sign in [-1,1]
    def classify(self,data):
        features = data[:,self.dim]
        if self.sign==-1:
            result = array(data[:,self.dim]<self.threshold,dtype='i')
        else:
            result = array(data[:,self.dim]>=self.threshold,dtype='i')
        return result
    def save(self,stream):
        data = array([self.sign,self.threshold,self.dim])
        data.dump(stream)
    def load(self,stream):
        data = load(stream)
        self.sign = data[0]
        self.threshold = data[1]
        self.dim = data[2]

class BoostedStumps:
    """Boosted decision stumps.  This uses the Stump1 class (it isn't a generic
    boosting implementation.  In addition, it's only a binary classifier."""
    stumps = None
    alphas = []
    def train(self,data,cls,nclass=2,k=10):
        assert self.stumps==None
        assert nclass==2
        assert (cls>=0).all()
        assert (cls<nclass).all()
        n,d = data.shape
        assert n==len(cls)
        weights = 1.0/(1+arange(len(data)))
        stumps = []
        alphas = []
        for i in range(k):
            if verbose: print "round",i
            stump = Stump1()
            stump.train(data,cls,nclass,weights=weights)
            err = stump.err
            assert err<=0.5
            if err==0.0: err=0.1/n
            alpha = 0.5 * log((1-err)/err)
            pred = stump.classify(data)
            sign = (pred==cls)*2-1
            weights = weights * exp(-sign*alpha)
            weights /= sum(weights)
            stumps.append(stump)
            alphas.append(alpha)
        self.stumps = stumps
        self.alphas = array(alphas)
    def binary_discriminants(self,data):
        totals = zeros(len(data))
        for i in range(len(self.stumps)):
            stump = self.stumps[i]
            alpha = self.alphas[i]
            d = stump.classify(data)
            totals += (2*(d>0)-1)*alpha
        return totals
    def classify(self,data):
        return array(self.binary_discriminants(data)>0,dtype='i')
    def save(self,stream):
        self.alphas.dump(stream)
        assert len(self.alphas)==len(self.stumps)
        for stump in self.stumps: stump.save(stream)
    def load(self,stream):
        self.alphas = load(stream)
        self.stumps = []
        for i in range(len(self.alphas)):
            stump = Stump1()
            stump.load(stream)
            self.stumps.append(stump)

import unittest
from test_classifier import *

class TestStump1(TestBatchClassifierBinary):
    factory = Stump1

class TestBoostedStumps(TestBatchClassifierBinary):
    factory = BoostedStumps

class MiscStumpTests(unittest.TestCase):
    def testOptThresh1(self):
        # simple tests
        q,t = opt_thresh(array([1,2,3,4,5,6,7,8]),array([1,1,1,1,1,1,1,1]),
                     array([0,0,0,0,1,1,1,1]))
        assert q==0 and t==4.5
    def testOptThresh2(self):
        q,t = opt_thresh(array([1,2,3,4,5,6,7,8]),array([1,1,1,1,1,1,1,1]),
                     array([1,0,0,0,1,1,1,1]))
        assert q==0.125 and t==4.5
    def testOptThresh3(self):
        # test weights
        q,t = opt_thresh(array([1,2,3,4,5,6,7,8]),array([0.5,1.5,1,1,1,1,1,1]),
                     array([1,0,0,0,1,1,1,1]))
        assert q==0.0625 and t==4.5
    def testOptThresh4(self):
        # test out of order
        q,t = opt_thresh(array([7,2,3,4,5,6,1,8]),array([1,1,1,1,1,1,1,1]),
                     array([1,0,0,0,1,1,1,1]))
        assert q==0.125 and t==4.5
    def testOptThresh5(self):
        # test whether constant runs are handled correctly
        q,t = opt_thresh(array([0,0,1]),array([2,1,1]),
                     array([0,1,1]))
        assert q==0.25 and t==0.5
    def testStump1Fun1(self):
        data = array([[0,0],[1,0],[2,0],[3,0]])
        weights = array([1,1,1,1])
        cls = array([0,0,1,1])
        best,err,thresh = stump1(data,weights,cls)
        assert best==0 and err==0 and thresh==1.5
    def testStump1Fun2(self):
        data = array([[0,0],[0,1],[0,2],[0,3]])
        weights = array([1,1,1,1])
        cls = array([0,0,1,1])
        best,err,thresh = stump1(data,weights,cls)
        assert best==1 and err==0 and thresh==1.5
    def testStumpFun1(self):
        data = array([[0,0],[1,0],[2,0],[3,0]])
        weights = array([1,1,1,1])
        cls = array([0,0,1,1])
        (sign,thresh,index),err = stump(data,weights,cls)
        assert index==0 and err==0 and thresh==1.5 and sign==1
    def testStumpFun2(self):
        data = array([[0,0],[0,1],[0,2],[0,3]])
        weights = array([1,1,1,1])
        cls = array([0,0,1,1])
        (sign,thresh,index),err = stump(data,weights,cls)
        assert index==1 and err==0 and thresh==1.5 and sign==1
    def testStumpFun3(self):
        data = array([[0,0],[0,1],[0,2],[0,3]])
        weights = array([1,1,1,1])
        cls = array([1,1,0,0])
        (sign,thresh,index),err = stump(data,weights,cls)
        assert index==1 and err==0 and thresh==-1.5 and sign==-1
    def testStumpFun4(self):
        data = array([[0,0],[1,0],[2,0],[3,0]])
        weights = array([1,1,1,1])
        cls = array([1,1,0,0])
        (sign,thresh,index),err = stump(data,weights,cls)
        assert index==0 and err==0 and thresh==-1.5 and sign==-1

###
### try out the code
###
    
def plot_fun2d(f,x0=-1,y0=-1,x1=1,y1=1):
    xs,ys = mgrid[x0:x1:100j,y0:y1:100j]
    coords = c_[xs.ravel(),ys.ravel()]
    zs = reshape(f(coords),xs.shape)
    imshow(rot90(zs),extent=[x0,x1,y0,y1])

def demo():
    n = 100
    c1 = array(RandomArray.multivariate_normal([0,0],identity(2),n))
    c2 = array(RandomArray.multivariate_normal([2,2],identity(2),n))
    global data,cls,result
    data = array(concatenate((c1,c2),axis=0))
    cls = array(concatenate(([0]*n,[1]*n)))
    # note: fails miserably without randomization
    perm = RandomArray.permutation(2*n)
    cls = take(cls,perm)
    data = take(data,perm,axis=0)
    model = BoostedStumps()
    model.train(data,cls,2,k=100)
    plot_fun2d(model.classify,-1,-1,3,3)

if __name__ == "__main__":
    unittest.main()
