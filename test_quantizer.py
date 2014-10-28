from __future__ import with_statement

__all__ = ["TestBatchQuantizer"]

import os,sys,os.path,re,string
from pylab import *
from numpy import *
import math
import unittest
import cPickle

def tempname(pattern):
    assert not ("/" in pattern)
    return "/tmp/"+pattern % str(random.randint(0,1000000))

class MockBatchQuantizer:
    """A mock implementation of a batch quantizer."""
    def __init__(self):
        self.n = None
    def train(self,data,cls=None,k=10):
        """Train a vector quantizer based on the data.  The quantizer may optionally
        be given class information, but it is up to the quantizer whether or how it
        uses it."""
        assert len(data)>=k
        assert self.n is None
        self.n,self.d = data.shape
        assert k==int(k)
        assert k>=2 and k<=10000000
        self.k = int(k)
        if cls is not None:
            assert (cls==floor(cls+0.5)).all()
            self.nclass = amax(cls)+1
    def prototype(self,i):
        """Return a vector representing a "prototype"
        corresponding to the quantization value i."""
        assert i>=0 and i<self.k
        return random.uniform(size=self.d)
    def quantize(self,data):
        """Quantize the data."""
        n,d = data.shape
        assert self.d==d
        sums = abs(sum(data,axis=1))
        return array(floor(sums) % self.k,dtype='i')
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

class TestBatchQuantizer(unittest.TestCase):
    params = {}
    factory = MockBatchQuantizer
    def set_factory(self,f):
        self.factory = f
        return self
    def setUp(self):
        random.seed(88)
        self.quantizer = self.factory()
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
        """BatchQuantizers need to raise an error if they are asked to classify without
        prior training."""
        self.make_data()
        self.assertRaises(Exception,self.quantizer.quantize,ones((100,100)))
    def testCheckTrainOnce(self):
        """A BatchQuantizer can be trained only once."""
        self.make_data()
        self.quantizer.train(self.train,**self.params)
        self.assertRaises(Exception,self.quantizer.train,self.train,**self.params)
    def testCheckTrainingFormat1(self):
        """BatchQuantizers need to check that training data is 2D and there are the same number
        of target classes."""
        self.make_data()
        self.assertRaises(Exception,self.quantizer.train,self.train[1],**self.params)
    def testCheckTrainingFormat2(self):
        """BatchQuantizers need to check that training data is 2D and there are the same number
        of target classes."""
        self.make_data()
        self.assertRaises(Exception,self.quantizer.train,ones((self.ntrain,self.d,2)),**self.params)
    def testCheckTrainingFormat3(self):
        """BatchQuantizers need to check that training data is 2D and there are the same number
        of target classes."""
        self.make_data()
        self.assertRaises(Exception,self.quantizer.train,self.train[1,:],**self.params)
    def testQuantize(self):
        """The quantize method must return int array of the right size."""
        self.make_data()
        self.quantizer.train(self.train,**self.params)
        pred = self.quantizer.quantize(self.test)
        assert pred.shape==(len(self.test),)
        assert pred.dtype=='int32' or pred.dtype=='int64'
    def testPrototype(self):
        """Prototype must return array of the right size and shape."""
        self.make_data()
        self.quantizer.train(self.train,**self.params)
        proto = self.quantizer.prototype(0)
        assert proto.shape==(self.d,)
        assert proto.dtype=='float' or proto.dtype=='double'
    def testCheckQuantizeFormat(self):
        """Quantize may not accept non-2D arrays."""
        self.make_data()
        self.quantizer.train(self.train,**self.params)
        self.assertRaises(Exception,self.quantizer.quantize,self.test[1])
        self.assertRaises(Exception,self.quantizer.quantize,117.0)
    def testSaveLoad(self):
        """Saving and loading a model should work and give the same result."""
        self.make_data()
        self.quantizer.train(self.train,**self.params)
        pred = self.quantizer.quantize(self.test)
        file = tempname("test%s.quantizer")
        with open(file,"w") as stream: self.quantizer.save(stream)
        quantizer2 = self.factory()
        with open(file) as stream: quantizer2.load(stream)
        os.remove(file)
        pred2 = quantizer2.quantize(self.test)
        assert (pred==pred2).all()


if __name__ == "__main__":
    unittest.main()
