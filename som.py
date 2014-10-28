__all__ = [
    "SOM",
    "som", "som_theta", "som_frequency",
    "som_representatives", "som_posteriors", "som_classes",
    "verbose",
]

from scipy import mgrid,linalg,ndimage
import sys,os,random,math
import numpy,pylab,scipy
from numpy import *

verbose = 0

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
def dist(u,v):
    return linalg.norm(u-v)
def distsq(x,y):
    d = x-y
    return dot(d.ravel(),d.ravel())

def som_theta(dist,iter,ngrid):
    """Compute a SOM theta value used for updating.
    (This is the default; you can define your own.)"""
    so = 100.0*ngrid
    to = 100.0*ngrid
    sigma = 10.0 * so/(so+iter)
    t = to/(to+iter) * exp(-dist/2/sigma)
    if t<1e-3: return 0
    return t

def som(data,shape=None,niter=10000000,threshold=0.03,theta=som_theta,torus=0):
    assert not isnan(data).any()
    """Compute a 2D self-organizing map for the data,
    with the given shape and the maximum number of iterations.
    The theta value used for updating is computed by the theta
    function passed as an argument."""
    if shape is None:
        k = max(3,floor(data.shape[0]**(1.0/3.0)))
        shape = (k,k)
    assert shape[0]>=3 and shape[1]>=3
    w,h = shape
    n,m = data.shape
    total = w*h
    items = rchoose(total,n)
    # print items
    grid = data[items,:].copy()
    for iter in range(niter):
        neighbor_update = theta(1.0,iter,total)
        if neighbor_update<threshold: break
        best = argmindist(data[iter%n],grid.reshape(w*h,m))
        x,y = best/h,best%h
        if verbose and iter%100==0:
            print iter,x,y,theta(1,iter,total)
        if theta(1,iter,total)<1e-2: break
        for index in range(w*h):
            u,v = index/h,index%h
            dx = u-x
            dy = v-y
            if torus:
                if abs(dx)>w/2: dx = abs(dx)-w
                if abs(dy)>h/2: dy = abs(dy)-h
            d = math.hypot(dx,dy)
            t = theta(d,iter,total)
            if t<1e-8: continue
            diff = data[iter%n,:]-grid[index,:]
            grid[index,:] += t * diff
    grid.shape = (w,h,m)
    return grid

def som_frequency(grid,data):
    """Given a SOM grid and some data, compute a grid of counts
    showing how frequently each vector is used."""
    counts = zeros(grid.shape[:2])
    w,h,m = grid.shape
    for i in range(len(data)):
        best = argmindist(data[i],grid.reshape(w*h,m))
        x,y = best/h,best%h
        counts[x,y] += 1
    return counts

def som_representatives(grid,data):
    """Given a SOM grid and some data, select the closest representative
    for each grid vector from the data."""
    w,h,m = grid.shape
    representatives = zeros((w,h,m))
    for index in range(w*h):
        x,y = index/w,index%w
        best = argmindist(grid[x,y],data)
        representatives[x,y] = data[best]
    return representatives

def som_posteriors(grid,data,classes):
    """Given a SOM grid, data, and classes, compute the posterior probabilities
    for each grid vector."""
    assert amin(classes)>=0
    k = amax(classes)+1
    counts = zeros(list(grid.shape[:2])+[k])
    w,h,m = grid.shape
    for i in range(len(data)):
        best = argmindist(data[i],grid.reshape(w*h,m))
        x,y = best/h,best%h
        counts[x,y,classes[i]] += 1
    return counts

def som_classes(grid,data,classes):
    """Given a SOM grid, data, and classes, compute the most probable class
    for each grid point (this just computes the argmax for the som_posteriors.)"""
    w,h,m = grid.shape
    posteriors = som_posteriors(grid,data,classes)
    cls = zeros(grid.shape[:2])
    for i in range(w):
        for j in range(h):
            cls[i,j] = argmax(posteriors[i,j])
    return cls

def som_show(grid,shape=None):
    pylab.clf()
    w,h,m = grid.shape
    if not shape:
        shape = (int(math.sqrt(m)),int(math.sqrt(m)))
    for i in range(w*h):
        pylab.subplot(w,h,i+1)
        pylab.imshow(grid[i/w,i%w].reshape(shape))

def plot_fun2d(f,x0=-1,y0=-1,x1=1,y1=1):
    xs,ys = mgrid[x0:x1:100j,y0:y1:100j]
    coords = c_[xs.ravel(),ys.ravel()]
    zs = reshape(f(coords),xs.shape)
    pylab.imshow(rot90(zs),extent=[x0,x1,y0,y1])

def plot_density2d(data):
    assert(data.shape[1]==2)
    x0 = min(data[:,0])
    y0 = min(data[:,1])
    x1 = max(data[:,0])
    y1 = max(data[:,1])
    kde = KernelDensity(data)
    pylab.clf()
    plot_fun2d(kde.eval,x0,y0,x1,y1)
    # scatter(data)

class SOM:
    """Transform or quantize data using a 2D SOM.  The output values
    are always 2D.  Transformation is carried out by first
    finding the closest matching SOM vector, then interpolating
    based on the activation of the central vector and its neighbors
    by the input vector."""
    grid = None
    def train(self,data,shape=None,niter=100000):
        """Train with the given input data."""
        assert self.grid is None
        self.grid = som(data,shape,niter=niter)
    def transform1(self,x):
        w,h,d = self.grid.shape
        best = argmindist(x,self.grid.reshape(w*h,d))
        x,y = best/h,best%h
        weights = []
        coords = []
        for i in range(max(x-1,0),min(x+1,w-1)):
            for j in range(max(y-1,0),min(y+1,h-1)):
                coords.append((i,j))
                weights.append(dist(self.grid[i,j],x))
        weights = array(weights)
        weights = 1.0/maximum(weights,1e-6)
        weights /= sum(weights)
        coords = array(coords,dtype='f')
        location = average(coords*weights.reshape(len(weights),1),axis=0)
        return location
    def transform(self,data):
        """Transform the input data using the computed SOM map."""
        return rowwise(self.transform1,data)
    def quantize(self,data):
        w,h,d = self.grid.shape
        n,d1 = data.shape
        assert d==d1
        return array([argmindist(data[i],self.grid.reshape(w*h,d))
                      for i in range(len(data))])
    def prototype(self,i):
        w,h,d = self.grid.shape
        return self.grid.reshape(w*h,d)[i]
    def save(self,stream):
        self.grid.dump(stream)
    def load(self,stream):
        self.grid = load(stream)

import unittest
from test_quantizer import *
from test_transformer import *

from som import SOM

class TestSOMQuantizer(TestBatchQuantizer):
    params = {"niter":100,"shape":(3,3)}
    factory = SOM

class TestSOMTransformer(TestBatchTransformer):
    params = {"niter":100,"shape":(3,3)}
    factory = SOM

if __name__ == "__main__":
    unittest.main()

