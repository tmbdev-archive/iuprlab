# -*- Python -*-

import random as pyrandom
from scipy import ndimage
from pylab import *
from numpy import *
from kmeans import *

if not vars().has_key("images"):
    images = load("mnist-train-images.dump")/255.0
    classes = load("mnist-train-labels.dump")
    images_t = load("mnist-test-images.dump")/255.0
    classes_t = load("mnist-test-labels.dump")

def distsq(u,v):
    d = u.ravel()-v.ravel()
    return dot(d,d)

import accel
argmindist = accel.argmindist


def skewed(image,s):
    im = ndimage.affine_transform(image,
                                  [[1,0],[s,1]],
                                  [0,-(s/2)*images[0].shape[1]])
    return im
def skews(image):
    for s in linspace(-0.3,0.3,10):
        yield (skewed(image,s),(s,))

def translated(image,dx,dy):
    return ndimage.translation(image,[dx,dy])
def translations(image):
    for dx in linspace(-2.0,2,5):
        for dy in linspace(-2.0,2,5):
            yield (translated(image,dx,dy),(dx,dy))
            

class TPQuant:
    def __init__(self,variants=skews,k=10,maxiter=100000):
        self.variants = skews
        self.means = None
        self.k = k
        self.maxiter = maxiter
        self.rate_offset = 1.0
        self.rate_pow = 0.5
    def lookup(self,base):
        best_v = None
        best_m = None
        best_p = None
        best_d = 1e38
        for v,p in self.variants(base):
            m = argmindist(v,self.means)
            d = distsq(v,self.means[m])
            # print m,p,d
            if d>=best_d: continue
            best_d = d
            best_m = m
            best_v = v
            best_p = p
        return (best_m,best_v,best_p)
    def train(self,data):
        k = self.k
        n = len(data)
        self.means = data[pyrandom.sample(xrange(n),k)]
        count = 100
        for i in xrange(self.maxiter):
            j = random.randint(n)
            m,v,p = self.lookup(data[j])
            l = 1.0/(self.rate_offset+math.pow(i,self.rate_pow))
            self.means[m] = (1-l)*self.means[m]+l*v
            if i%1000==0: print i,l
    def counts(data,centers,variants=skews):
        counts = zeros((len(centers),))
        for i in range(len(data)):
            m,v,p = vbest(data[i])
            counts[m] += 1
        return array(counts,int)
    def histograms(data,classes,centers,variants=skews):
        counts = zeros((len(centers),maximum(classes)+1))
        for i in range(len(data)):
            m,v,p = vbest(data[i])
            counts[m,classes[i]] += 1
        return array(counts,int)

class TPSom:
    def __init__(self,variants=skews,r=10,maxiter=100000):
        self.variants = skews
        self.grid = None
        self.r = r
        self.k = r*r
        self.maxiter = 100000
        self.threshold = 0.01
        self.rate_offset = 1.0
        self.rate_pow = 0.5
    def lookup(self,base):
        best_v = None
        best_m = None
        best_p = None
        best_d = 1e38
        for v,p in self.variants(base):
            m = argmindist(v,self.grid)
            d = distsq(v,self.grid[m])
            # print m,p,d
            if d>=best_d: continue
            best_d = d
            best_m = m
            best_v = v
            best_p = p
        return (best_m,best_v,best_p)
    def theta(self,dist,iter):
        """Compute a SOM theta value used for updating.
        (This is the default; you can define your own.)"""
        ngrid = self.k
        so = 100.0*ngrid
        to = 100.0*ngrid
        sigma = 10.0 * so/(so+iter)
        t = to/(to+iter) * exp(-dist/2/sigma)
        if t<1e-3: return 0
        return t
    def train(self,data):
        k = self.k
        r = self.r
        n = len(data)
        self.grid = data[pyrandom.sample(xrange(n),r)]
        for iter in xrange(self.maxiter):
            v = data[iter%n]
            neighbor_update = self.theta(1.0,iter)
            if neighbor_update<self.threshold: break
            best = argmindist(v,grid)
            x,y = best/h,best%h
            if iter%100==0:
                print iter,x,y,theta(1,iter,total)
            if theta(1,iter,total)<1e-2: break
            for index in range(w*h):
                u,v = index/r,index%r
                dx = u-x
                dy = v-y
                if torus:
                    if abs(dx)>w/2: dx = abs(dx)-w
                    if abs(dy)>h/2: dy = abs(dy)-h
                d = math.hypot(dx,dy)
                t = theta(d,iter,total)
                if t<1e-8: continue
                diff = v-grid[index,:]
                grid[index,:] += t * diff
        grid.shape = (w,h,m)
        return grid
    def counts(data,centers,variants=skews):
        counts = zeros((len(centers),))
        for i in range(len(data)):
            m,v,p = vbest(data[i])
            counts[m] += 1
        return array(counts,int)
    def histograms(data,classes,centers,variants=skews):
        counts = zeros((len(centers),maximum(classes)+1))
        for i in range(len(data)):
            m,v,p = vbest(data[i])
            counts[m,classes[i]] += 1
        return array(counts,int)

def showgrid(data,w=10,h=10):
    for i in range(min(w*h,len(data))):
        subplot(h,w,i+1)
        imshow(data[i])

q = TPSom(maxiter=10000)
q.train(images)
