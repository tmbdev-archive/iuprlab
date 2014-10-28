from pylab import *
from scipy import mgrid,linalg,ndimage
from matplotlib.collections import LineCollection
import sys,os,random,math
import numpy,pylab,scipy
from numpy import *

verbose = 0


def rchoose(k,n):
    "Choose k distinct value from range(n)."
    assert k<=n
    return random.permutation(range(n))[:k]
def rowwise(f,data,samples=None):
    """Apply f to the rows of data, optionally selecting
    rows with samples array."""
    assert data.ndim==2
    if samples is None: samples = range(len(data))
    return array([f(data[i]) for i in samples])
def argmindist(x,data):
    "Find the row in data with the minimum distance from x."
    dists = [distsq(x,v) for v in data]
    return argmin(dists)
def dist(u,v):
    "Euclidean distance between u and v."
    return linalg.norm(u-v)
def distsq(x,y):
    "Squared Euclidean distance between u and v."
    d = x-y
    return dot(d.ravel(),d.ravel())

def symset(a,i,j,v):
    "Set entries i,j and j,i in a to v."
    a[i,j] = v
    a[j,i] = v

def random_selections(data,n=999999999):
    """Iterator through random permutations of the given data.
    Each "epoch" contains all the samples exactly once."""
    count = 0
    while 1:
        selection = random.permutation(len(data))
        for index in selection:
            if count>=n: return
            yield data[index]
            count += 1

def plot_gng(units,conn,data=None):
    """Plot the status of the GNG."""
    ion()
    clf()
    if data is not None:
        scatter(x=list(data[:,0]),
                y=list(data[:,1]),
                c='g',marker='+')
    scatter(x=[u[0] for u in units if u is not None],
            y=[u[1] for u in units if u is not None],
            c='b')
#     indexes = [tuple(x) for x in argwhere(conn) if x[0]<x[1]]
#     segments = []
#     for index in indexes:
#         u = units[index[0]]
#         v = units[index[1]]
#         segments.append(concatenate((u,v)))
#     lc = LineCollection(segments)
    draw()

def logrange(start,end,n):
    """Step through a range logarithmically."""
    return exp(arange(n)*(log(end)-log(start))/n+log(start))

def gng(data,
        nmax=100,               # maximum number of units
        amax=100,               # maximum age
        nfreq=500,              # number of iters before adding new unit
        maxunchanged=10000,     # stop if no structural changes after this # iters
        maxiter=1000000,        # total maximum number of iterations
        eps1 = 0.1,             # learning rate for closest unit
        eps2 = 0.01,            # learning rate for next closest unit
        alpha = 0.7,            # error decay after splitting
        beta = 0.99,            # general error decay
        verbose = verbose):           
    """Compute a 2D self-organizing map for the data, 
    with the given shape and the maximum number of iterations.
    The theta value used for updating is computed by the theta
    function passed as an argument.
        nmax=100,               # maximum number of units
        amax=100,               # maximum age
        nfreq=500,              # number of iters before adding new unit
        maxunchanged=10000,     # stop if no structural changes after this # iters
        maxiter=1000000,        # total maximum number of iterations
        eps1 = 0.1,             # learning rate for closest unit
        eps2 = 0.01,            # learning rate for next closest unit
        alpha = 0.7,            # error decay after splitting
        beta = 0.99,            # general error decay
        verbose = 0
    """
    units = [None]*nmax
    units[0] = data[0]
    units[1] = data[1]
    errs = zeros(nmax)
    age = zeros((nmax,nmax))
    age[:,:] = amax
    unchanged = 0               # number of iterations without structural change
    count = 0
    for x in random_selections(data):
        assert not isnan(x).any()
        dists = [distsq(x,v) for v in units if v is not None]
        indexes = argsort(dists)
        i1 = indexes[0]
        d1 = dists[i1]
        i2 = indexes[1]
        d2 = dists[i2]
        age[i1,i2] = 0; age[i2,i1] = 0
        errs[i1] += d1
        units[i1] += eps1 * (x - units[i1])
        units[i2] += eps2 * (x - units[i2])
        age[i1,:] += 1; age[:,i1] += 1
        for unit in [i1,i2]:
            if units[unit] is None: continue
            if (age[unit,:]<amax).any(): continue
            if verbose: print "deleting",unit
            units[unit] = None
            errs[unit] = 0
            unchanged = 0
        if (count+1)%nfreq==0 and units.count(None)>0:
            worst = argmax(errs)
            if verbose: print "splitting",worst
            neighbors = (age[worst,:]<amax) # bool vector of neighbors
            neighbor = argmax(neighbors*errs) # 0 for non-neighbors
            interpolated = 0.5*(units[worst]+units[neighbor])
            unit = units.index(None) # find first empty slot
            units[unit] = interpolated
            # delete old connections
            age[worst,neighbor] = amax; age[neighbor,worst] = amax
            # add new connections
            age[worst,unit] = 0; age[unit,worst] = 0 
            age[neighbor,unit] = 0; age[unit,neighbor] = 0
            # decrease error
            errs[worst] *= alpha
            errs[neighbor] *= alpha
            errs[unit] = (errs[worst]+errs[neighbor])/2
            unchanged = 0
        errs[:] *= beta
        if verbose and count%1000==0:
            average_err = sum(errs) / len(errs)
            if len(units[0])==2: plot_gng(units,age,data)
            nunits = len(units)-units.count(None)
            nconn = sum(age<amax)/2
            total_err = sum(errs)
            print count,"#units",nunits,"#conn",nconn,"err",total_err,"err/unit",total_err/nunits
        unchanged += 1
        count += 1
        if unchanged>maxunchanged: break
    active = (age<amax).any(axis=1)
    units = array([units[i] for i in range(len(active)) if active[i]])
    errs = errs[active]
    age = age[active,:]
    age = age[:,active]
    return (units,(age<amax),errs)

def show_grid(data,ncols=10):
    """Show the data on a grid."""
    print data.shape
    pylab.clf()
    for i in range(data.shape[0]):
        pylab.subplot((data.shape[0]+ncols-1)/ncols,ncols,i+1)
        pylab.imshow(data[i,:,:])

def show_classes(units,data,classes):
    """Show which class is associated with which unit."""
    units = units.reshape(units.shape[0],prod(units.shape[1:]))
    data = data.reshape(data.shape[0],prod(data.shape[1:]))
    nclasses = amax(classes)+1
    counts = zeros((len(units),nclasses))
    for i in range(len(data)):
        v = data[i]
        unit = argmindist(v,units)
        counts[unit,classes[i]] += 1
    for unit in range(len(units)):
        print unit,":",
        rank = argsort(-counts[unit,:])
        for i in rank:
            if counts[unit,i]==0: break
            if i>32:
                print "%s->%d"%(chr(i),counts[unit,i]),
            else:
                print "%d->%d"%(i,counts[unit,i]),
        print

import pickle

def test1():
    """Simple test case involving two squares."""
    data = concatenate((random.uniform(size=(100,2))-array([2,0]),
                        random.uniform(size=(100,2))))
    gng(data,nmax=20,nfreq=1000,amax=200,verbose=1)

def test2():
    """Test case on character shapes extracted from alice."""
    global images,classes
    (images,classes) = pickle.load(open("alice.pickle"))
    (n,w,h) = images.shape
    images.shape = (n,w*h)
    images = 1.0*images
    for i in range(images.shape[0]):
        images[i,:] *= 1.0/max(1e-6,amax(images[i,:]))
    global units,conns,errs
    units,conns,errs = gng(images,
                           nmax=50,
                           nfreq=10000,
                           amax=1000,
                           maxunchanged=60000,
                           eps1=0.1,eps2=0.03,
                           verbose=1)
    units = array(units)
    units.shape = (units.shape[0],w,h)

def test3():
    """Test case on mnist data."""
    global images,classes
    (images,classes) = pickle.load(open("mnist.pickle"))
    (n,w,h) = images.shape
    images.shape = (n,w*h)
    images = 1.0*images
    for i in range(images.shape[0]):
        images[i,:] *= 1.0/max(1e-6,amax(images[i,:]))
    global units,conns,errs
    units,conns,errs = gng(images,
                           nmax=50,
                           maxiter=100000,
                           nfreq=30000,
                           amax=1000,
                           maxunchanged=60000,
                           eps1=0.1,eps2=0.03,
                           verbose=1)
    units = array(units)
    units.shape = (units.shape[0],w,h)

class GrowingNeuralGas:
    """Quantization using the "growing neural gas" algorithm."""
    def __init__(self):
        self.means = None
    def train(self,data,k=None,maxiter=None,**keywords):
        """Train a GNG quantizer."""
        assert self.means is None
        n,d = data.shape
        if k is None:
            k = max(2,int(math.sqrt(d)))
        if maxiter is None:
            maxiter = 4*n
        self.means,_,_ = gng(data,nmax=k,maxiter=maxiter,**keywords)
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

class TestGrowingNeuralGasQuantizer(TestBatchQuantizer):
    factory = GrowingNeuralGas

if __name__ == "__main__":
    unittest.main()
        
