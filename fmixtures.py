from __future__ import with_statement

__all__ = ["FastGaussianMixtureFixed"]

import sys,os,random,math
import numpy,pylab,scipy,scipy.linalg
from numpy import *
from pylab import *

verbose = 0

def rchoose(k,n):
    "Randomly choose k distinct values from range(n)."
    return numpy.random.permutation(range(n))[:k]
def dist(u,v):
    "Euclidean distance of u and v."
    return linalg.norm(u-v)
def iternonzero(a):
    "Iterate over the non-zero elements of a."
    for index in range(len(a.flat)):
        if a.flat[index]!=0:
            yield unravel_index(index,a.shape)
def pairdistances(u,v):
    "Compute a matrix of all the pariwise distances of u and v."
    n,m = u.shape
    l,m1 = v.shape
    assert m==m1
    result = 0.0*zeros((n,l))
    for i in range(n):
        for j in range(l):
            d = dist(u[i],v[j])
            result[i,j] = d
    return result

### fast gaussian mixture for fixed variance

def fast_gaussian_mixture_fixed(data,k=2,maxiter=1000,sigma=1.0,auto_sigma=0):
    "Fast Gaussian mixture modeling for fixed sigma."
    n,d = data.shape
    means = data[rchoose(k,n),:]
    r = zeros((k,n),'d')
    oldmeans = means
    dists = pairdistances(means,data)
    err = zeros((k,n),'d')
    for iter in range(maxiter):
        oldmeans = means
        if sigma!=None:
            r = maximum(exp(-dists**2/2/sigma/sigma),1e-100)
            r /= sum(r,axis=0).reshape(1,n)
        else:
            r.fill(0.0)
            for j in range(n): r[argmin(dists[:,j]),j] = 1
        means = dot(r,data) / sum(r,axis=1).reshape(k,1)
        assert not isnan(means).any()
        shift = array([dist(means[i],oldmeans[i]) for i in range(k)]).reshape(k,1)
        err += shift
        # in c code, only compute this for reasonably small distances to speed things up
        lo = exp(-(dists+err)**2/2/sigma/sigma)
        hi = exp(-maximum(dists-err,0)**2/2/sigma/sigma)
        rel = amax(dists,axis=0); rel += (rel==0); rel = rel.reshape(1,n)
        needs_update = (abs(hi-lo)/rel>1e-3)
        for i,j in iternonzero(needs_update):
            dists[i,j] = dist(means[i],data[j])
            err[i,j] = 0
        if verbose:
            print "   ",iter,amax(shift,None),amax(err,None),\
                sum(needs_update,axis=None),sigma
        if amax(shift,None)<1e-6: break
        if auto_sigma:
            mindists = amin(dists,axis=0)
            sigma = sqrt(sum(pow(mindists,2))/len(mindists)/d)
    return means,sigma

class FastGaussianMixtureFixed:
    "Fast Gaussian mixture modeling for fixed sigma."
    means = None
    def train(self,data,k=2,maxiter=1000,sigma=1.0,auto_sigma=0):
        n,d = data.shape
        assert self.means is None
        means,sigma = fast_gaussian_mixture_fixed(data,k=k,maxiter=maxiter,sigma=sigma,auto_sigma=0)
        assert means.shape==(k,d)
        self.means = means
        self.sigma = sigma
    def loglikelihood(self,data):
        assert self.means is not None
        assert data.ndim==2
        if len(data.shape)==1: data = data.reshape(1,len(data))
        n,d = data.shape
        dists = pairdistances(self.means,data)
        r = -dists**2/(2*self.sigma**2)
        r = amax(r,axis=0)
        return r
    def bic(self,data):
        L = sum(self.log_likelihood(data))
        k = prod(self.means.shape)+1
        n = len(data)
        result = - 2 * abs(L) + k * log(n)
        return result
    def save(self,stream):
        self.means.dump(stream)
        array([self.sigma]).dump(stream)
    def load(self,stream):
        self.means = numpy.load(stream)
        self.sigma = numpy.load(stream)[0]


import unittest
from test_density import *

class TestFastGaussianMixtureFixed(TestBatchDensityEstimator):
    factory = FastGaussianMixtureFixed

if __name__ == "__main__":
    unittest.main()
