__all__ = ["PCA", "pca", "pca_filter", "PCAGHA", "pca_gha_filter"]

import sys,os,random,math
import numpy,pylab,scipy
from numpy import *
from scipy import stats, mgrid, c_, reshape, random, rot90, linalg

def normalize(v):
    s = linalg.norm(v)
    if abs(s)<1e-6: return v
    return v / s

def normalize_rows(w):
    result = zeros(w.shape)
    for i in range(w.shape[0]):
        result[i,:] = normalize(w[i,:])
    return result

def fraction(evals,frac):
    p = evals/sum(evals)
    p.sort()
    p = cumsum(p)
    k = searchsorted(p,frac,side="left")
    return k

###
### PCA using linalg.eig
###

def pca_filter(data,k=None,frac=0.99,whiten=0):
    """Computes a PCA and (optionally) whitening of the data, then projects
    back into the original space.  The output data has the same
    dimensions as the input data."""
    n,d = data.shape
    data = data - average(data,axis=0).reshape(1,d)
    cov = dot(data.T,data)/n
    evals,evecs = linalg.eig(cov)
    if k is None: k = fraction(evals,frac)
    top = argsort(-evals)
    evals = evals[top[:k]]
    evecs = evecs.T[top[:k]]
    assert evecs.shape==(k,d)
    ys = dot(evecs,data.T)
    assert ys.shape==(k,n)
    if whiten: ys = dot(diag(sqrt(1.0/evals)),ys)
    whitened = dot(evecs.T,ys)
    assert data.shape == whitened.T.shape
    return whitened.T

def pca(data,k=None,frac=0.99,whiten=0):
    """Computes a PCA and a whitening.  The number of
    components can be specified either directly or as a fraction
    of the total sum of the eigenvalues.  The function returns
    the transformed data, the mean, the eigenvalues, and 
    the eigenvectors."""
    n,d = data.shape
    mean = average(data,axis=0).reshape(1,d)
    data = data - mean.reshape(1,d)
    cov = dot(data.T,data)/n
    evals,evecs = linalg.eig(cov)
    if k is None: k = fraction(evals,frac)
    top = argsort(-evals)
    evals = evals[top[:k]]
    evecs = evecs.T[top[:k]]
    assert evecs.shape==(k,d)
    ys = dot(evecs,data.T)
    assert ys.shape==(k,n)
    if whiten: ys = dot(diag(sqrt(1.0/evals)),ys)
    return (ys.T,mean,evals,evecs)

class PCA:
    """Perform a PCA transformation based on batch training."""
    mean = None
    def train(self,data,k=None,frac=0.99,whiten=1):
        """Train the PCA and (optional) whitening filter with data.  
        The number of components can be specified either directly 
        or as a fraction of the total sum of the eigenvalues.  This uses
        linalg.eig to do the PCA computation."""
        n,d = data.shape
        if self.mean: raise Exception("already trained")
        ys,mean,evals,evecs = pca(data,k=k,frac=frac,whiten=whiten)
        self.mean = mean
        self.w = evecs
    def transform(self,data):
        """Transform the data using the trained PCA."""
        n,d = data.shape
        data = data - average(data,axis=0).reshape(1,d)
        return dot(data,self.w.T)
    def reconstruct(self,data):
        """Given transformed data, reconstruct the closest original vector."""
        k,d = self.w.shape
        return dot(data,self.w)+self.mean.reshape(1,d)
    def save(self,stream):
        """Save the trained PCA to disk."""
        self.mean.dump(stream)
        self.w.dump(stream)
    def load(self,stream):
        """Load a PCA model from disk."""
        self.mean = load(stream)
        self.w = load(stream)

###
### incremental PCA using GHA
###

def pca_gha(data,k,eta=lambda x:0.01,niter=10000,mean=None,scale=None,verbose=0):
    """Compute PCA of the input data using a generalized Hebbian architecture.
    The input data must have mean zero, or the mean needs to be provided
    as an argument."""
    assert mean.shape==data.shape[1:]
    if mean is None: assert (abs(average(data,axis=0))<1e-2).all()
    n,m = data.shape
    if verbose: print "normalizing"
    w = normalize_rows(1.0+tril(ones((k,m))))
    if verbose: print "iterating"
    for i in range(niter):
        if verbose and i%1000==0:
            print "iter",i
        if i%100==0:
            if not numpy.isfinite(w).all(): raise "divergence"
        x = data[i%n,:]
        if mean is not None: x = x-mean.ravel()
        if scale is not None: x = x/scale
        y = dot(w,x)
        e = eta(i)
        w = e * outer(y,x)+ w - e * dot(tril(outer(y,y)),w)
    return w

def pca_gha_filter(data,k,eta=lambda x:0.01,niter=10000,mean=None,scale=None):
    """Perform PCA filtering of the data using a PCA computed using the
    generalized Hebbian architecture."""
    n,d = data.shape
    data = data - average(data,axis=0).reshape(1,d)
    a = pca_gha(data,k,eta,niter)
    return dot(dot(data,a.T),a)

class PCAGHA:
    """A PCA transformation of the data using a generalized Hebbian architecture
    learning method."""
    mean = None
    def train(self,data,k=None):
        """Train the PCA with data.  The number of 
        components can be specified either directly or as a fraction
        of the total sum of the eigenvalues.  This uses
        a generalized Hebbian architecture for training (essentially,
        gradient descent)."""
        if k is None: k = max(int(math.sqrt(data.shape[1])),1)
        n,d = data.shape
        assert self.mean is None
        self.mean = average(data,axis=0)
        self.w = pca_gha(data,k,niter=max(1000,n*4),mean=self.mean)
    def transform(self,data):
        """Transform the data using the trained PCA."""
        n,d = data.shape
        data = data - average(data,axis=0).reshape(1,d)
        return dot(data,self.w.T)
    def reconstruct(self,data):
        """Given transformed data, reconstruct the closest original vector."""
        k,d = self.w.shape
        return dot(data,self.w)+self.mean.reshape(1,d)
    def save(self,stream):
        """Save the trained PCA to disk."""
        self.mean.dump(stream)
        self.w.dump(stream)
    def load(self,stream):
        """Load a PCA model from disk."""
        self.mean = load(stream)
        self.w = load(stream)


import unittest
from test_transformer import *

class TestPCA(TestBatchTransformer):
    factory = PCA

class TestPCAGHA(TestBatchTransformer):
    factory = PCAGHA

if __name__ == "__main__":
    unittest.main()
