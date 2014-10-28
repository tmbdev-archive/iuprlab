from __future__ import with_statement

__all__ = ["TestBatchClassifier","TestBatchClassifierBinary"]

import os,sys,os.path,re,string
from pylab import *
from numpy import *
import math
import unittest
import cPickle

def tempname(pattern):
    assert not ("/" in pattern)
    return "/tmp/"+pattern % str(random.randint(0,1000000))

class MockBatchClassifier:
    """A mock implementation of a batch classifier."""
    def __init__(self):
        self.n = None
    def train(self,data,cls,nclass):
        """Use the rows of data and corresponding classifications in cls to train
        a classifier.  May only be called once."""
        assert self.n is None
        self.n,self.d = data.shape
        assert (cls==floor(cls+0.5)).all()
        assert (cls<nclass).all()
        self.nclass = nclass
    def posteriors(self,data):
        """Compute the posterior probabilities based on the training data.
        This method is optional. If it is present, it must be consistent
        with discriminants and classify output."""
        assert self.n is not None
        n,d = data.shape
        assert self.d==d
        sums = abs(sum(data,axis=1))%self.nclass
        centers = arange(self.nclass,dtype='f')
        probs = abs(subtract.outer(sums,centers))
        probs /= sum(probs,axis=1).reshape(n,1)
        return probs
    def discriminants(self,data):
        """Compute discriminant functions based on the training data.  This
        method is optional."""
        return self.posteriors(data)
    def classify(self,data):
        """Perform classification of the input data based on the training data."""
        return argmax(self.discriminants(data),axis=1)
    def save(self,stream):
        """Save the classifier to disk."""
        cPickle.dump(self.n,stream)
        cPickle.dump(self.d,stream)
        cPickle.dump(self.nclass,stream)
    def load(self,stream):
        """Load the classifier from disk."""
        self.n = cPickle.load(stream)
        self.d = cPickle.load(stream)
        self.nclass = cPickle.load(stream)

class TestBatchClassifier(unittest.TestCase):
    # these are class variables that are overridden in subclasses
    params = {}
    factory = MockBatchClassifier
    def set_factory(self,f):
        self.factory = f
        return self
    def make_data(self):
        self.ntrain = random.randint(30, 70)
        self.ntest = random.randint(10, 30)
        self.d = random.randint(3,19)
        self.nclass = random.randint(2,7)
        self.train = random.uniform(size=(self.ntrain,self.d))
        self.cls = random.randint(0,self.nclass-1,size=self.ntrain)
        self.cls[0] = self.nclass-1
        self.test = random.uniform(size=(self.ntest,self.d))
    def setUp(self):
        random.seed(88)
        self.classifier = self.factory()
    def testCheckPriorTraining(self):
        """BatchClassifiers need to raise an error if they are asked to classify without
        prior training."""
        if not callable(getattr(self.classifier,"posteriors",None)): return
        self.make_data()
        self.assertRaises(Exception,self.classifier.classify,ones((100,100)))
    def testCheckPriorTraining1(self):
        """BatchClassifiers need to raise an error if they are asked to classify without
        prior training."""
        if not callable(getattr(self.classifier,"posteriors",None)): return
        self.make_data()
        self.assertRaises(Exception,self.classifier.posteriors,ones((100,100)))
    def testCheckTrainOnce(self):
        """A classifier can be trained only once."""
        self.make_data()
        self.classifier.train(self.train,self.cls,self.nclass,**self.params)
        self.assertRaises(Exception,self.classifier.train,self.train,self.cls,self.nclass,**self.params)
    def testCheckTrainingFormat(self):
        """BatchClassifiers need to check that training data is 2D and there are the same number
        of target classes."""
        self.make_data()
        self.assertRaises(Exception,self.classifier.train,self.train[0],self.cls,self.nclass)
    def testCheckTrainingFormat1(self):
        """BatchClassifiers need to check that training data is 2D and there are the same number
        of target classes."""
        self.make_data()
        self.assertRaises(Exception,self.classifier.classify,self.train[1],self.cls,self.nclass)
    def testCheckTrainingFormat2(self):
        """BatchClassifiers need to check that training data is 2D and there are the same number
        of target classes."""
        self.make_data()
        self.assertRaises(Exception,self.classifier.classify,ones((self.ntrain,self.d,2)),self.cls,self.nclass)
    def testCheckTrainingFormat2(self):
        """BatchClassifiers need to check that training data is 2D and there are the same number
        of target classes."""
        self.make_data()
        self.assertRaises(Exception,self.classifier.classify,self.train[1,:],self.nclass)
    def testCheckNclass(self):
        """A classifier must check for consistency between nclass and the class labels."""
        self.make_data()
        self.assertRaises(Exception,self.classifier.train,self.train,self.cls,self.nclass-1)
    def testClassify(self):
        """Make sure the classifier returns consistent output."""
        self.make_data()
        self.classifier.train(self.train,self.cls,self.nclass,**self.params)
        pred = self.classifier.classify(self.test)
        assert pred.shape==(len(self.test),)
        assert pred.dtype=='int32' or pred.dtype=='int64'
        assert (pred>=0).all()
        assert (pred<self.nclass).all()
    def testClassifyFormat(self):
        """classify may only accept 2D data array"""
        self.make_data()
        self.classifier.train(self.train,self.cls,self.nclass,**self.params)
        self.assertRaises(Exception,self.classifier.classify,self.test[1,:])
        self.assertRaises(Exception,self.classifier.classify,117.0)
    def testDiscriminantsFormat(self):
        """discriminants may only accept 2D data array"""
        if not callable(getattr(self.classifier,"discriminants",None)): return
        self.make_data()
        self.classifier.train(self.train,self.cls,self.nclass,**self.params)
        self.assertRaises(Exception,self.classifier.discriminants,self.test[1,:])
        self.assertRaises(Exception,self.classifier.discriminants,117.0)
    def testPosteriorsFormat(self):
        """posteriors may only accept 2D data array"""
        if not callable(getattr(self.classifier,"posteriors",None)): return
        self.make_data()
        self.classifier.train(self.train,self.cls,self.nclass,**self.params)
        self.assertRaises(Exception,self.classifier.posteriors,self.test[1,:])
        self.assertRaises(Exception,self.classifier.posteriors,117.0)
    def testDiscriminants(self):
        """discriminants must return float/double array of the right size"""
        if not callable(getattr(self.classifier,"discriminants",None)): return
        self.make_data()
        self.classifier.train(self.train,self.cls,self.nclass,**self.params)
        pred = self.classifier.discriminants(self.test)
        assert pred.shape==(len(self.test),self.nclass)
        assert pred.dtype=='float' or pred.dtype=='double'
    def testPosteriors(self):
        """posteriors must return float/double array with normalized rows"""
        if not callable(getattr(self.classifier,"posteriors",None)): return
        self.make_data()
        self.classifier.train(self.train,self.cls,self.nclass,**self.params)
        pred = self.classifier.posteriors(self.test)
        assert pred.shape==(len(self.test),self.nclass)
        assert pred.dtype=='float' or pred.dtype=='double'
        assert (pred>=0).all()
        assert (pred<=1).all()
        assert (abs(sum(pred,axis=1)-1.0)<1e-4).all()
    def testPosteriorsConsistentWithDiscriminants(self):
        """posteriors and discriminants need to be consistent"""
        if not callable(getattr(self.classifier,"posteriors",None)): return
        self.make_data()
        self.classifier.train(self.train,self.cls,self.nclass,**self.params)
        pred = self.classifier.posteriors(self.test)
        pred2 = self.classifier.discriminants(self.test)
        assert pred.ndim==2
        assert pred2.ndim==2
        assert (argsort(pred,axis=1)==argsort(pred2,axis=1)).all()
    def testClassificationsConsistentWithDiscriminants(self):
        """posteriors and discriminants need to be consistent"""
        if not callable(getattr(self.classifier,"discriminants",None)): return
        self.make_data()
        self.classifier.train(self.train,self.cls,self.nclass,**self.params)
        pred = self.classifier.classify(self.test)
        pred2 = self.classifier.discriminants(self.test)
        assert pred.ndim==1
        assert pred2.ndim==2
        assert (pred==argmax(pred2,axis=1)).all()
    def testSaveLoad(self):
        """saving and loading a model should work and give the same result"""
        self.make_data()
        self.classifier.train(self.train,self.cls,self.nclass,**self.params)
        pred = self.classifier.classify(self.test)
        file = tempname("test%s.classifier")
        with open(file,"w") as stream: self.classifier.save(stream)
        classifier2 = self.factory()
        with open(file) as stream: classifier2.load(stream)
        pred2 = classifier2.classify(self.test)
        assert (pred==pred2).all()

class TestBatchClassifierBinary(TestBatchClassifier):
    def make_data(self):
        self.ntrain = random.randint(3,17)
        self.ntest = random.randint(3,19)
        self.d = random.randint(3,19)
        self.nclass = 2
        self.train = random.uniform(size=(self.ntrain,self.d))
        self.cls = random.randint(0,self.nclass,size=self.ntrain)
        self.cls[0] = self.nclass-1
        self.test = random.uniform(size=(self.ntest,self.d))


if __name__ == "__main__":
    unittest.main()
