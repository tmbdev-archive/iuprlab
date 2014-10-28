__all__ = ["KMeans", "SlowKMeans", "kmeans", "soft_kmeans", "fast_kmeans"]

import sys,os,random,math
import numpy,pylab,scipy
from numpy import *

verbose = 0
nops_dist = 0
CHECK = 0

def rchoose(k,n):
    assert k<=n
    return random.permutation(range(n))[:k]
def rowwise(f,data,samples=None):
    assert data.ndim==2
    if samples is None: samples = range(len(data))
    return array([f(data[i]) for i in samples])
def argmindist(x,data):
    dists = [distsq(x,v) for v in data]
    return argmin(dists)
def argmindist2(x,data):
    dists = [distsq(x,v) for v in data]
    i = argmin(dists)
    return i,dists[i]
def dist(u,v):
    return linalg.norm(u-v)
def distsq(u,v):
    d = u-v
    return dot(d,d)
def pairdistances(u,v):
    n,m = u.shape
    l,m1 = v.shape
    assert m==m1
    result = zeros((n,l))
    for i in range(n):
        for j in range(l):
            d = dist(u[i],v[j])
            result[i,j] = d
    return result

# regular k-means algorithm

def kmeans(data,k,maxiter=100):
    """Regular k-means algorithm.  Computes k means from data."""
    global nops_dist, verbose, CHECK
    means = data[rchoose(k,len(data))]
    oldmins = None
    for i in range(maxiter):
        if verbose: sys.stderr.write("[kmeans iter %d]\n"%i)
        mins = array([argmindist(x,means) for x in data],'i')
        nops_dist += len(data) * len(means)
        if alltrue(mins==oldmins): break
        for i in range(k):
            where = mins==i
            if sum(where)<1: continue
            means[i] = average(data[where],axis=0)
        oldmins = mins
    return means

def incremental_kmeans(data,k,maxiter=None,rate_offset=1.0,rate_pow=0.5):
    """k-means, but update centers after each sample."""
    global nops_dist, verbose, CHECK
    if not maxiter: maxiter = 100*len(data)
    assert k>2 and k<1000000
    assert rate_offset>0.0
    assert rate_pow>=0.01 and rate_pow<=4.0
    n = len(data)
    assert n>k
    means = data[rchoose(k,n)]
    count = 100
    for i in xrange(maxiter):
        j = random.randint(n)
        m = argmindist(data[j],means)
        # l = 1.0/(rate_offset+math.pow(count,rate_pow))
        l = 1.0/(rate_offset+math.pow(i,rate_pow))
        means[m] = (1-l)*means[m]+l*data[j]
    return means

def auto_kmeans(data,k,maxiter=None,runlength=10000):
    """Incremental k-means with simple stopping rule."""
    global nops_dist, verbose, CHECK
    if not maxiter: maxiter = 50*len(data)
    n = len(data)
    means = data[rchoose(k,n)]
    count = 100
    err = 0.0
    le = 0.5/runlength
    best = 1e30
    run = 0
    for i in xrange(maxiter):
        j = random.randint(n)
        m,d = argmindist2(data[j],means)
        l = 1.0/math.sqrt(count)
        means[m] = (1-l)*means[m]+l*data[j]
        err = (1-le)*err+le*d
        if err<best:
            best = err
            run = 0
        if run>runlength: break
        run += 1
    return means

def soft_kmeans(data,k,maxiter=1000,beta=1.0):
    """Like kmeans, but with a non-sharp cutoff.  Basically mixture
    learning with an exponential."""
    global r,means
    n,d = data.shape
    means = data[rchoose(k,n),:]
    last = zeros((k,n))
    for i in range(maxiter):
        dists = pairdistances(means,data)
        if (abs(dists-last)<1e-5).all(): break
        last = dists
        r = exp(-beta * dists)
        r /= sum(r,axis=0).reshape(1,n)
        means = dot(r,data) / sum(r,axis=1).reshape(k,1)
    return means


def fast_kmeans(data,k,maxiter=100):
    """ An unpublished fast k-means algorithm that uses bounds on the changes
    of distances between datapoints and means to avoid re-evaluation
    of points that haven't moved between clusters."""
    global nops_dist, verbose, CHECK
    n,d = data.shape
    # initial assignment
    means = data[rchoose(k,len(data))]
    dists = pairdistances(means,data)
    nops_dist += n*k
    cluster = argmin(dists,axis=0)
    # recompute the means and counts
    means = zeros((k,d))
    counts = zeros(k)
    for i in range(k):
        matching = sum(i==cluster)
        if matching==0:
            means[i] = data[random.randint(0,len(data)-1)]
        else:
            means[i] = average(data[i==cluster,:],axis=0)
        counts[i] = matching
    # recompute the distances
    dists = pairdistances(means,data)
    nops_dist += n*k
    errs = zeros((k,n))
    nchanged = n
    for iter in range(maxiter):
        if verbose: sys.stderr.write("[fkmeans %d %d]\n"%(iter,nchanged))
        assert sum(counts)==n
        # update distances where the minimum has become ambiguous
        for i in range(n):
            lo = dists[:,i]-errs[:,i]
            hi = dists[:,i]+errs[:,i]
            js = argsort(lo)
            for ji in range(len(js)-1):
                j = js[ji]
                j1 = js[ji+1]
                if lo[j1]>hi[j]: break
                dists[j,i] = dist(means[j],data[i])
                errs[j,i] = 0
                nops_dist += 1
        if CHECK:
            actual_dists = pairdistances(means,data)
            assert (argmin(actual_dists,axis=0)==argmin(dists,axis=0)).all()
        # find the new cluster assignments
        ncluster = argmin(dists,axis=0)
        changed = compress(cluster!=ncluster,range(n))
        nchanged = len(changed)
        if nchanged==0: break
        if CHECK:
            actual_dists = pairdistances(means,data)
            assert (abs(actual_dists-dists)<=errs).all()
        # move vectors between classes
        oldmeans = means.copy()
        for i in changed:
            oc = cluster[i]
            means[oc] = (means[oc]*counts[oc] - data[i])/(counts[oc]-1)
            counts[oc] -= 1
            nc = ncluster[i]
            means[nc] = (means[nc]*counts[nc] + data[i])/(counts[nc]+1)
            counts[nc] += 1
            dists[:,i] = [dist(means[l],data[i]) for l in range(k)]
            errs[:,i] = 0
        # now, update the error estimates by how much the means have moved
        shifts = [dist(means[i],oldmeans[i]) for i in range(k)]
        for i in range(n): errs[:,i] += shifts
        if CHECK:
            actual_dists = pairdistances(means,data)
            assert (abs(actual_dists-dists)<=errs).all()
        cluster = ncluster
    if verbose: print counts
    return means

class SlowKMeans:
    """k-means using the standard k-means algorithm.  If beta is given,
    uses a soft k-means algorithm."""
    def __init__(self):
        self.means = None
    def train(self,data,k=None,maxiter=None,beta=None):
        """Train a KMeans quantizer."""
        assert self.means is None
        n,d = data.shape
        if k is None: k = max(2,int(math.sqrt(d)))
        if maxiter is None: maxiter = 10*n
        if beta is None: 
            self.means = kmeans(data,k,maxiter=maxiter)
        else:
            self.means = soft_kmeans(data,k,maxiter=maxiter,beta=beta)
    def quantize(self,data):
        """Quantize the data."""
        assert self.means is not None
        return rowwise(lambda x:argmindist(x,self.means),data)
    def prototype(self,i):
        """Get the prototype for index i."""
        return self.means[i]
    def save(self,stream):
        """Save the quantizer."""
        self.means.dump(stream)
    def load(self,stream):
        """Load the quantizer."""
        self.means = load(stream)

class KMeans:
    """k-means using the fast k-means algorithm."""
    def __init__(self):
        self.means = None
    def train(self,data,k=None,maxiter=None):
        """Train a KMeans quantizer."""
        assert self.means is None
        n,d = data.shape
        if k is None: k = max(2,int(math.sqrt(d)))
        if maxiter is None: maxiter = 4*n
        self.means = fast_kmeans(data,k,maxiter=maxiter)
    def quantize(self,data):
        """Quantize the data."""
        assert self.means is not None
        return rowwise(lambda x:argmindist(x,self.means),data)
    def prototype(self,i):
        """Get the prototype for index i."""
        return self.means[i]
    def save(self,stream):
        """Save the quantizer."""
        self.means.dump(stream)
    def load(self,stream):
        """Load the quantizer."""
        self.means = load(stream)

class IKMeans:
    """k-means using the incremental k-means algorithm."""
    def __init__(self):
        self.means = None
    def train(self,data,k=None,maxiter=None,rate_offset=1.0,rate_pow=0.5):
        """Train a KMeans quantizer."""
        assert self.means is None
        n,d = data.shape
        if k is None: k = max(2,int(math.sqrt(d)))
        if maxiter is None: maxiter = 4*n
        self.means = incremental_kmeans(data,k,maxiter=maxiter,
                                        rate_offset=rate_offset,
                                        rate_pow=0.5)
    def quantize(self,data):
        """Quantize the data."""
        assert self.means is not None
        return rowwise(lambda x:argmindist(x,self.means),data)
    def prototype(self,i):
        """Get the prototype for index i."""
        return self.means[i]
    def save(self,stream):
        """Save the quantizer."""
        self.means.dump(stream)
    def load(self,stream):
        """Load the quantizer."""
        self.means = load(stream)


import unittest
from test_quantizer import *

class TestSlowKMeansQuantizer(TestBatchQuantizer):
    factory = SlowKMeans

class TestKMeansQuantizer(TestBatchQuantizer):
    factory = KMeans

class TestIKMeansQuantizer(TestBatchQuantizer):
    factory = IKMeans

if __name__ == "__main__":
    unittest.main()
