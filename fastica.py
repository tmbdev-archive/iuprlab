__all__ = ["FastICA","fastica"]

import sys,os,random,math,numpy,pylab,scipy,scipy.linalg
from numpy import *
from scipy.linalg import norm
from pylab import imshow,show,clf,gray

verbose = 0

def project_perp(data,vs):
    "Project on the space perpendicular to the space spanned by the vs."
    if len(vs.shape)==1: vs = vs.reshape(1,len(vs))
    for v in vs:
        v = v/norm(v)
        for i in range(len(data)):
            data[i] -= dot(v,data[i]) * v

def fraction(evals,frac):
    "Find a quantile (fractile)."
    p = evals/sum(evals)
    p.sort()
    p = cumsum(p)
    k = searchsorted(p,frac,side="left")
    return k

def pca(data,k=None,frac=0.99,whiten=1):
    """Compute the PCA and whitening of the input data.
    Returns the transformed data, the mean, the eigenvalues,
    and the eigenvectors."""
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

def g(x):
    "Nonlinear function for fastica."
    return x**3
def gp(x):
    "Derivative of nonlinear function for fastica."
    return 3*x**2

def fastica1(data,maxiter=1000,g=g,gp=gp,eps=1e-3):
    """Perform fast-ICA for one component using the given g function;
    gp must be the derivative of g."""
    n,d = data.shape
    iter = 0
    index = argmax([norm(x) for x in data])
    w = data[index]
    w /= norm(w)
    while iter<maxiter:
        iter += 1
        w1 = zeros(d)
        for i in range(len(data)):
            x = data[i]
            dp = dot(w,x)
            v = g(dp)*x - gp(dp)*w
            w1 += v
        w1 /= norm(w1)
        if average(w1)<0: w1 = -w1
        delta = 1.0 - abs(dot(w,w1))
        if verbose: print iter,delta
        if delta<eps: return w1
        w = w1
    return None

def fastica(data,ncomp,maxiter=1000,g=g,gp=gp,eps=1e-2):
    """Perform fast-ICA for ncomp components using the given
    g and g' functions."""
    result = []
    for comp in range(ncomp):
        w = fastica1(data,maxiter=maxiter,g=g,gp=gp,eps=eps)
        result.append(w)
        project_perp(data,w)
        if verbose: print comp,amin(data),amax(data)
    return array(result)

class FastICA:
    """Compute an ICA data transformation using the given
    training data.  Also performs PCA and whitening of the
    data prior to invoking FastICA."""
    w = None
    def train(self,data,k=None,pca_k=None,frac=0.99,g=g,gp=gp):
        """Train using the given data set."""
        assert self.w is None
        n,d = data.shape
        ys,mean,evals,evecs = pca(data,k=pca_k,frac=frac)
        assert "ys",ys.shape==(n,len(evals))
        if verbose: print "#pca",len(evals),"#ica",k
        if not k: k = len(evals)
        comps = fastica(ys,ncomp=k)
        self.mean = mean
        self.w = dot(comps,evecs)
    def transform(self,data):
        """Transform data using the PCA and ICA transformations."""
        n,d = data.shape
        assert d==self.w.shape[1]
        assert self.w is not None
        k,d = self.w.shape
        return dot(data-self.mean.reshape(1,d),self.w.T)
    def reconstruct(self,data):
        """Perform the inverse transformation given the components."""
        k,d = self.w.shape
        return dot(data,self.w)+self.mean.reshape(1,d)
    def save(self,stream):
        """Save the transformer."""
        self.mean.dump(stream)
        self.w.dump(stream)
    def load(self,stream):
        """Load the transformer."""
        self.mean = load(stream)
        self.w = load(stream)

import unittest
from test_transformer import *

class TestFastICA(TestBatchTransformer):
    params = {"k":3,"pca_k":5}
    factory = FastICA

if __name__ == "__main__":
    unittest.main()
