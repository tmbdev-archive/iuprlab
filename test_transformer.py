from __future__ import with_statement

__all__ = ["TestBatchTransformer"]

import os,sys,os.path,re,string
from pylab import *
from numpy import *
import math
import unittest
import cPickle

def tempname(pattern):
    assert not ("/" in pattern)
    return "/tmp/"+pattern % str(random.randint(0,1000000))

class MockBatchTransformer:
    """A mock implementation of a batch transformer.  Batch transformer
    are trained on a batch of data and then perform some kind of transformation.
    Examples of batch transformers are PCA and ICA."""
    def __init__(self):
        self.n = None
    def train(self,data,k=2):
        """Perform unsupervised learning of a class that transforms the input
        data into output data in some way.  The dimensionality of the output data
        may be given as k, or it may be automatically selected by the algorithm."""
        assert self.n is None
        if k is None: k = random.randint(2,111)
        self.k = k
        self.n,self.d = data.shape
        assert k==int(k)
        assert k>=2 and k<=10000000
    def transform(self,data):
        """Transform the input data based on the trained model."""
        assert self.n is not None
        n,d = data.shape
        assert self.d==d
        sums = abs(sum(data,axis=1))%self.k
        centers = arange(self.k,dtype='f')
        probs = abs(subtract.outer(sums,centers))
        probs /= sum(probs,axis=1).reshape(n,1)
        return probs
    def reconstruct(self,data):
        """Optional method: given a transformed vector, reconstruct the input vector
        as closely as possible."""
        assert self.n is not None
        n,k = data.shape
        assert self.k==k
        return random.uniform(size=(n,self.d))
    def save(self,stream):
        """Save the transformer to disk."""
        cPickle.dump(self.n,stream)
        cPickle.dump(self.d,stream)
        cPickle.dump(self.k,stream)
    def load(self,stream):
        """Load the transformer from disk."""
        self.n = cPickle.load(stream)
        self.d = cPickle.load(stream)
        self.k = cPickle.load(stream)

class TestBatchTransformer(unittest.TestCase):
    """Unit testing for batch transformers."""
    params = {}
    factory = MockBatchTransformer
    def set_factory(self,f):
        self.factory = f
        return self
    def setUp(self):
        random.seed(88)
        self.k = random.randint(3,19)
        self.transformer = self.factory()
    def make_data(self):
        self.ntrain = random.randint(30,40)
        self.ntest = random.randint(3,19)
        self.d = random.randint(10,20)
        self.nclass = random.randint(2,12)
        self.train = random.uniform(size=(self.ntrain,self.d))
        self.cls = random.randint(0,self.nclass,size=self.ntrain)
        self.cls[0] = self.nclass-1
        self.test = random.uniform(size=(self.ntest,self.d))
    def testCheckPriorTraining(self):
        """BatchTransformers need to raise an error if they are asked to classify without
        prior training."""
        self.make_data()
        self.assertRaises(Exception,self.transformer.transform,ones((100,100)))
    def testCheckTrainOnce(self):
        """A transformer can be trained only once."""
        self.make_data()
        self.transformer.train(self.train,**self.params)
        self.assertRaises(Exception,self.transformer.train,self.train,**self.params)
    def testCheckTrainingFormat(self):
        """BatchTransformers need to check that training data is 2D and there are the same number
        of target classes."""
        self.make_data()
        self.assertRaises(Exception,self.transformer.train,self.train[0],**self.params)
    def testCheckTrainingFormat1(self):
        """BatchTransformers need to check that training data is 2D and there are the same number
        of target classes."""
        self.make_data()
        self.assertRaises(Exception,self.transformer.train,self.train[1],**self.params)
    def testCheckTrainingFormat2(self):
        """BatchTransformers need to check that training data is 2D and there are the same number
        of target classes."""
        self.make_data()
        self.assertRaises(Exception,self.transformer.train,ones((self.ntrain,self.d,2)),**self.params)
    def testCheckTrainingFormat3(self):
        """BatchTransformers need to check that training data is 2D and there are the same number
        of target classes."""
        self.make_data()
        self.assertRaises(Exception,self.transformer.train,self.train[1,:],**self.params)
    def testTransform(self):
        """The transform method must return float/double array of the right size."""
        self.make_data()
        self.transformer.train(self.train,**self.params)
        pred = self.transformer.transform(self.test)
        assert pred.shape[0]==len(self.test)
        assert pred.dtype=='float' or pred.dtype=='double'
    def testCheckTransformFormat(self):
        """The transform method may not accept non-2D arrays."""
        self.make_data()
        self.transformer.train(self.train,**self.params)
        self.assertRaises(Exception,self.transformer.transform,self.test[1,:])
        self.assertRaises(Exception,self.transformer.transform,117.0)
    def testSaveLoad(self):
        """Saving and loading a model should work and give a model that behaves
        the same way as the original."""
        self.make_data()
        self.transformer.train(self.train,**self.params)
        pred = self.transformer.transform(self.test)
        file = tempname("test%s.transformer")
        with open(file,"w") as stream: self.transformer.save(stream)
        transformer2 = self.factory()
        with open(file) as stream: transformer2.load(stream)
        os.remove(file)
        pred2 = transformer2.transform(self.test)
        assert ((pred-pred2)/maximum(maximum(abs(pred),abs(pred2)),0.1)<1e-6).all()
    def testReconstruction(self):
        """If a reconstruction method exists, it must return a vector in the
        original space."""
        if not callable(getattr(self.transformer,"reconstruct",None)): return
        self.make_data()
        self.transformer.train(self.train,**self.params)
        v = self.transformer.transform(self.test)
        w = self.transformer.reconstruct(v)
        assert w.shape==self.test.shape

if __name__ == "__main__":
    unittest.main()
