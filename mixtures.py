from __future__ import with_statement
import sys,os,random,math,numpy
import numpy,pylab,scipy,scipy.linalg
from numpy import *
from pylab import *

verbose = 0

def rchoose(k,n):
    "Choose k distinct numbers from range(n)"
    return numpy.random.permutation(range(n))[:k]
def dist(u,v):
    "Euclidean distance."
    return linalg.norm(u-v)
def pairdistances(u,v):
    "Compute all pairwise distances."
    n,m = u.shape
    l,m1 = v.shape
    assert m==m1
    result = 0.0*zeros((n,l))
    for i in range(n):
        for j in range(l):
            d = dist(u[i],v[j])
            result[i,j] = d
    return result
def rowwise(f,data,samples=None):
    "Apply f to the rows of data (selected optionally by samples)."
    assert data.ndim==2
    if samples is None: samples = range(len(data))
    return array([f(data[i]) for i in samples])

def gaussian_mixture_fixed(data,k,maxiter=1000,sigma=1.0):
    "Gaussian mixtures with fixed sigma."
    n,d = data.shape
    means = data[rchoose(k,n),:]
    last = zeros((k,n))
    for iter in range(maxiter):
        dists = pairdistances(means,data)
        if (abs(dists-last)<1e-5).all(): break
        last = dists
        r = exp(-dists**2/2/sigma/sigma)
        r /= sum(r,axis=0).reshape(1,n)
        oldmeans = means
        means = dot(r,data) / sum(r,axis=1).reshape(k,1)
        shift = array([dist(means[i],oldmeans[i]) for i in range(k)]).reshape(k,1)
        if verbose: print "   ",iter,amax(shift,None)
        if amax(shift,None)<1e-6: break
    assert means.shape==(k,d)
    assert means.ndim==2
    return means

def gaussian_mixture(data,k=2,maxiter=1000,start_sigma=1.0,
                     mode='diag',minsigma=0.1,always_update=1,thresh=1e-3):
    "Gaussian mixture with variable covariance matrix."
    n,d = data.shape
    means = data[rchoose(k,n),:]
    oldmeans = means
    dists = zeros((k,n),'d')
    if mode=='spherical':
        sigmas = array([start_sigma for i in range(k)])
    elif mode=='diag':
        sigmas = array([ones(d)*start_sigma for i in range(k)])
    elif mode=='full':
        sigmas = array([eye(d,d)*start_sigma for i in range(k)])
    else: raise "unknown mode (supported: spherical diag full)"
    for iter in range(maxiter):
        # compute inverse of the covariance matrix
        if mode=='spherical':
            assert sigmas.shape == (k,)
            sigmats = [eye(d,d)*1.0/max(minsigma,s) for s in sigmas]
        elif mode=='diag':
            assert sigmas.shape == (k,d)
            sigmats = [diag(1.0/maximum(minsigma,s)) for s in sigmas]
        elif mode=='full':
            assert sigmas.shape == (k,d,d)
            sigmats = [scipy.linalg.inv(maximum(minsigma*eye(d,d),s)) for s in sigmas]

        for i in range(k):
            for j in range(n):
                delta = data[j]-means[i]
                r = dot(delta,dot(sigmats[i],delta))
                dists[i,j] = exp(-r)

        # compute responsibilities
        responsibility = dists / maximum(sum(dists,axis=0).reshape(1,n),1e-10)

        # update the means
        oldmeans = means
        global norm
        norm = sum(responsibility,axis=1).reshape(k,1)
        means = dot(responsibility,data) / norm
        assert not isnan(means).any()

        # update the variances
        if mode=='spherical':
            assert sigmas.shape == (k,)
            sigmas.fill(0.0)
            for i in range(k):
                for j in range(n):
                    delta = data[j]-means[i]
                    r = dot(delta,delta)
                    sigmas[i] += responsibility[i,j]*r
            sigmas /= norm
        elif mode=='diag':
            assert sigmas.shape == (k,d)
            sigmas.fill(0.0)
            for i in range(k):
                for j in range(n):
                    delta = data[j]-means[i]
                    r = delta**2
                    sigmas[i] += responsibility[i,j]*r
            sigmas /= norm
        elif mode=='full':
            assert sigmas.shape == (k,d,d)
            sigmas.fill(0.0)
            for i in range(k):
                for j in range(n):
                    delta = data[j]-means[i]
                    sigmas[i] += responsibility[i,j]*outer(delta,delta)
            sigmas /= norm

        # compute shifts
        shift = array([dist(means[i],oldmeans[i]) for i in range(k)]).reshape(k,1)
        if verbose: print "   ",iter,amax(shift,None)

        # for s in sigmas: print "   ",s
        if amax(shift,None)<thresh: break
    return means,sigmas

class GaussianMixtureFixed:
    means = None
    def train(self,data,k=2,maxiter=1000,sigma=1.0):
        n,d = data.shape
        assert self.means is None
        means = gaussian_mixture_fixed(data,k=k,maxiter=maxiter,sigma=sigma)
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


def logadd(x,y): return x + log(1+exp(y-x))
def logsub(x,y): return x + log(1-exp(y-x))

class GaussianMixture:
    means=None
    def train(self,data,k=2,maxiter=1000,sigma=1.0):
        assert self.means is None
        means,sigma = gaussian_mixture(data,k=k,maxiter=1000,start_sigma=1.0,
                                       mode='diag',minsigma=0.1,always_update=1,thresh=1e-3)
        self.means = means
        if sigma.ndim==1:
            self.siginv = 1.0/sigma**2
        elif sigma.ndim==2:
            self.siginv = 1.0/sigma**2
        elif sigma.ndim==3:
            self.siginv = array([linalg.inv(m) for m in sigma])
    def loglikelihood1(self,x):
        assert x.ndim==1
        means = self.means
        k,d = means.shape
        siginv = self.siginv
        result = None
        if siginv.ndim==1:
            for i in range(k):
                r = linalg.norm(x-means[i])
                ll = -0.5 * r**2 * siginv[i]
                if result==None: result = ll
                else: result = logadd(result,ll)
        elif siginv.ndim==2:
            for i in range(k):
                delta = x-means[i]
                ll = sum(-0.5 * delta**2 * siginv[i])
                if result==None: result = ll
                else: result = logadd(result,ll)
        elif siginv.ndim==3:
            for i in range(k):
                delta = x-means[i]
                ll = -0.5 * dot(delta,dot(siginv[i],delta))
                if result==None: result = ll
                else: result = logadd(result,ll)
        return result
    def loglikelihood(self,data):
        return rowwise(self.loglikelihood1,data)
    def bic(self,data):
        L = sum(self.log_likelihood(data))
        k = prod(self.means.shape)+1
        n = len(data)
        result = - 2 * abs(L) + k * log(n)
        return result
    def save(self,stream):
        self.means.dump(stream)
        self.siginv.dump(stream)
    def load(self,stream):
        self.means = numpy.load(stream)
        self.siginv = numpy.load(stream)

################################################################
### test cases
################################################################

def example(n=100,k=2):
    clf()
    global data
    c1 = RandomArray.multivariate_normal([0,5],eye(2,2),shape=(n))
    c2 = RandomArray.multivariate_normal([5,0],eye(2,2),shape=(n))
    data = concatenate([c1,c2])
    scatter(data[:,0],data[:,1],c="blue")
    means,sigma = fast_gaussian_mixture(data,k=k,sigma=1)
    print means
    scatter(means[:,0],means[:,1],c="red")
    r = gm_likelihood(data,means,1.0)
    print r.shape
    assert len(r)==len(data)
    print r

def example2(n=1000,k=2,s=0.1):
    clf()
    global data
    c1 = RandomArray.multivariate_normal([0,5],eye(2,2)*pow(s,2),shape=(n))
    c2 = RandomArray.multivariate_normal([5,0],eye(2,2)*pow(s,2),shape=(n))
    data = concatenate([c1,c2])
    scatter(data[:,0],data[:,1],c="blue")
    means,sigma = fast_gaussian_mixture(data,k=k,sigma=1,auto_sigma=1)
    print means,sigma
    scatter(means[:,0],means[:,1],c="red")

def example3(n=1000,k=2,s=2.0):
    clf()
    global data
    c1 = RandomArray.multivariate_normal([0,5],eye(2,2)*pow(s,2),shape=(n))
    c2 = RandomArray.multivariate_normal([5,0],eye(2,2)*pow(s,2),shape=(n))
    c3 = RandomArray.multivariate_normal([5,5],eye(2,2)*pow(s,2),shape=(n))
    c4 = RandomArray.multivariate_normal([0,0],eye(2,2)*pow(s,2),shape=(n))
    data = concatenate([c1,c2,c3,c4])
    mix = best_mixture(data)
    scatter(data[:,0],data[:,1],c="blue")
    means,sigma = fast_gaussian_mixture(data,k=k,sigma=1,auto_sigma=1)
    print means,sigma
    scatter(means[:,0],means[:,1],c="red")

def example4(n=4000,k=2,s=2.0):
    clf()
    global data
    c1 = RandomArray.multivariate_normal([0,5],diag([1,3]),shape=(n))
    c2 = RandomArray.multivariate_normal([5,0],diag([3,1]),shape=(n))
    data = concatenate([c1,c2])
    means,sigmas = gaussian_mixture(data,mode='full')
    print means
    print sigmas
    scatter(data[:,0],data[:,1],c="blue")
    # scatter(means[:,0],means[:,1],c="red")

def load_mnist():
    global data,cls,tdata,tcls
    data = numpy.load("mnist-train-images-deskewed.dump")
    data.shape = (60000,225)
    cls = numpy.load("mnist-train-labels.dump")
    tdata = numpy.load("mnist-test-images-deskewed.dump")
    tdata.shape = (10000,225)
    tcls = numpy.load("mnist-test-labels.dump")

def example_mnist():
    global data,cls,tdata,tcls,boost,vdata,vnet,vtdata,vtpred
    nstages = 30
    load_mnist()
    global means,sigmas
    means,sigmas = gaussian_mixture(data[:1000,:],k=16,mode='diag',minsigma=1.0)

def example_mnist_show():
    global means,sigmas
    pylab.clf()
    w = 6
    h = 6
    for i in range(2*len(means)):
        pylab.subplot(w,h,i+1)
        if i%2==0:
            pylab.imshow(means[i/2].reshape(15,15))
        else:
            pylab.imshow(sigmas[i/2].reshape(15,15))

import unittest
from test_density import *

class TestGaussianMixtureFixed(TestBatchDensityEstimator):
    factory = GaussianMixtureFixed
    
class TestGaussianMixture(TestBatchDensityEstimator):
    factory = GaussianMixture

if __name__ == "__main__":
    unittest.main()
