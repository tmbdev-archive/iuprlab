from __future__ import with_statement
import os,sys,os.path,re,string
from pylab import *
from numpy import *
import math
import unittest
import cPickle

__all__ = ["TestBatchDensityEstimator"]


def tempname(pattern):
    assert not ("/" in pattern)
    return "/tmp/"+pattern % str(random.randint(0,1000000))

class MockBatchDensityEstimator:
    def __init__(self):
        self.n = None
    def train(self,data,cls=None,k=10):
        """Train a density estimator on the data."""
        assert len(data)>=k
        assert self.n==None
        self.n,self.d = data.shape
        assert k==int(k)
        assert k>=2 and k<=10000000
        self.k = int(k)
        if cls!=None:
            assert (cls==floor(cls+0.5)).all()
            self.nclass = amax(cls)+1
    def loglikelihood(self,data):
        """Compute the log likelihood."""
        n,d = data.shape
        assert self.d==d
        sums = abs(sum(data,axis=1))
        return -log(maximum(sums,1e-6))
    def save(self,stream):
        """Save the quantizer to disk."""
        cPickle.dump(self.n,stream)
        cPickle.dump(self.d,stream)
        cPickle.dump(self.k,stream)
    def load(self,stream):
        """Load the quantizer from disk."""
        self.n = cPickle.load(stream)
        self.d = cPickle.load(stream)
        self.k = cPickle.load(stream)

class TestBatchDensityEstimator(unittest.TestCase):
    # override the class variables in subclasses
    params = {}
    factory = MockBatchDensityEstimator
    def set_factory(self,f):
        self.factory = f
        return self
    def setUp(self):
        random.seed(88)
        self.estimator = self.factory()
    def make_data(self):
        self.ntrain = random.randint(50,100)
        self.ntest = random.randint(3,19)
        self.d = random.randint(5,11)
        self.nclass = random.randint(2,12)
        self.train = random.uniform(size=(self.ntrain,self.d))
        self.cls = random.randint(0,self.nclass,size=self.ntrain)
        self.cls[0] = self.nclass-1
        self.test = random.uniform(size=(self.ntest,self.d))
    def testCheckPriorTraining(self):
        """Batchquantizers need to raise an error if they are asked to classify without
        prior training."""
        self.make_data()
        self.assertRaises(Exception,self.estimator.loglikelihood,ones((100,100)))
    def testCheckTrainOnce(self):
        """A quantizer can be trained only once."""
        self.make_data()
        self.estimator.train(self.train,**self.params)
        self.assertRaises(Exception,self.estimator.train,self.train,**self.params)
    def testCheckTrainingFormat1(self):
        """Batchquantizers need to check that training data is 2D and there are the same number
        of target classes."""
        self.make_data()
        self.assertRaises(Exception,self.estimator.train,self.train[1],**self.params)
    def testCheckTrainingFormat2(self):
        """Batchquantizers need to check that training data is 2D and there are the same number
        of target classes."""
        self.make_data()
        self.assertRaises(Exception,self.estimator.train,ones((self.ntrain,self.d,2)),**self.params)
    def testCheckTrainingFormat3(self):
        """Batchquantizers need to check that training data is 2D and there are the same number
        of target classes."""
        self.make_data()
        self.assertRaises(Exception,self.estimator.train,self.train[1,:],**self.params)
    def testQuantize(self):
        """quantize must return int array of the right size"""
        self.make_data()
        self.estimator.train(self.train,**self.params)
        pred = self.estimator.loglikelihood(self.test)
        assert pred.shape==(len(self.test),)
        assert pred.dtype=='float' or pred.dtype=='double'
    def testCheckLikelihoodFormat(self):
        """quantize may not accept non-2D arrays"""
        self.make_data()
        self.estimator.train(self.train,**self.params)
        self.assertRaises(Exception,self.estimator.loglikelihood,self.test[1])
        self.assertRaises(Exception,self.estimator.loglikelihood,117.0)
    def testSaveLoad(self):
        """saving and loading a model should work and give the same result"""
        self.make_data()
        self.estimator.train(self.train,**self.params)
        pred = self.estimator.loglikelihood(self.test)
        file = tempname("test%s.loglikelihoodr")
        with open(file,"w") as stream: self.estimator.save(stream)
        quantizer2 = self.factory()
        with open(file) as stream: quantizer2.load(stream)
        os.remove(file)
        pred2 = quantizer2.loglikelihood(self.test)
        assert (pred==pred2).all()


if __name__ == "__main__":
    unittest.main()
