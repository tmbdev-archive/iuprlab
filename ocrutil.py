from __future__ import with_statement

import glob,os,sys,re,gzip
import cStringIO
from math import atan2
from types import ListType
from PIL import Image
import numpy
from numpy import linalg
from numpy import array,amax,amin,arange,where,dot,diag,\
    pi,cos,sin,zeros,concatenate
import scipy
from scipy import ndimage
import pylab
from pylab import imshow,show,gray
from imageutil import pad_image,crop_tight,extend_image,crop_image,\
    read_rgb32,channels_to_rgb32,read_pil

################################################################
# MNIST data reader
################################################################                    

def read_int(stream):
    "Read a 32 bit integer from the stream."
    result = 0
    for i in range(4):
        result = (result<<8) + ord(stream.read(1))
    return result

def mnist_read(file,nmax=9999999,verbose=1):
    "Read an MNIST data file and return the contents as a NumPy array."
    if re.search('\.gz$',file):
        stream = gzip.GzipFile(file,"rb")
    else:
        stream = open(file,"rb")
    magic = read_int(stream)
    rank = (magic & 255) - 1
    n = read_int(stream)
    n = min(n,nmax)
    if rank==0:
        data = stream.read(n)
        result = numpy.fromstring(data,"u1",n)
        if verbose:
            sys.stderr.write("read %d scalars from %s\n"%(n,file))
        return list(result)
    elif rank==2:
        w = read_int(stream)
        h = read_int(stream)
        result = []
        for i in range(n):
            data = stream.read(w*h)
            img = numpy.fromstring(data,"u1",w*h)
            img.shape = (w,h)
            result.append(img)
        if verbose:
            sys.stderr.write("read %d %dx%d images from %s\n"%(n,w,h,file))
        return result

################################################################
# isolated character shape/size normalization
################################################################

def bbox1(a):
    indexes = arange(len(a)).compress(a!=0)
    lo = min(indexes)
    hi = max(indexes)
    return (lo,hi+1)
    
def bbox(image,eps=0.1):
    image = where(image<eps,0.0,1.0)
    hor = amax(image,axis=1)
    x0,x1 = bbox1(hor)
    vert = amax(image,axis=0)
    y0,y1 = bbox1(vert)
    return (x0,y0,x1,y1)

def normalize_range(image):
    return (image - image.min()) * 1.0 / (image.max() - image.min())


def normalize_center(img,shape=(30,30),normalize=1):
    if type(img)==ListType:
        return [normalize_center(image,shape=shape) for image in img]
    # rescale into 0-1 range
    image = normalize_range(img)
    # prepare x/y coordinate arrays (we rely on automatic replication for
    # element-wise operations)
    xs = arange(image.shape[0])
    xs.shape = (len(xs),1)
    ys = arange(image.shape[1])
    ys.shape = (1,len(ys))
    # compute image centroid
    total = image.sum()
    cx = (xs*image).sum()/total
    cy = (ys*image).sum()/total
    # perform the actual transformation
    affine = diag([1,1])
    ccenter = array((cx,cy))
    ocenter = array((shape[0]/2,shape[1]/2))
    offset = ccenter - dot(affine,ocenter)
    if normalize:
        img = image
    return ndimage.affine_transform(img,affine, offset=offset, output_shape=shape)

def compute_stats(image):
    # prepare x/y coordinate arrays (we rely on automatic replication for
    # element-wise operations)
    xs = arange(image.shape[0])
    xs.shape = (len(xs),1)
    ys = arange(image.shape[1])
    ys.shape = (1,len(ys))
    # compute image centroid
    total = image.sum()
    cx = (xs*image).sum()/total
    cy = (ys*image).sum()/total
    # compute recentered coordinates (make sure they're float)
    nxs = xs - cx + 0.0
    nys = ys - cy + 0.0
    # compute eigenvectors for rotation
    cxx = (nxs*nxs*image).sum()/total
    cxy = (image*nxs*nys).sum()/total
    cyy = (nys*nys*image).sum()/total
    return (cx,cy,cxx,cxy,cyy)

def atan2right(y,x):
    alpha = atan2(y,x)
    while alpha<-pi/2:
        alpha += pi
    while alpha>pi/2:
        alpha -= pi
    return alpha

def atan2upper(y,x):
    alpha = atan2(y,x)
    while alpha<0:
        alpha += pi
    while alpha>=pi:
        alpha -= pi
    return alpha

def normalize_rotation(img,shape=(30,30),normalize=1):
    if type(img)==ListType:
        return [normalize_rotation(image,shape=shape) for image in img]
    # rescale into 0-1 range
    image = normalize_range(img)
    # compute image statistics
    cx,cy,cxx,cxy,cyy = compute_stats(image)
    # compute eigenvectors
    mat = array([[cxx,cxy],[cxy,cyy]])
    v,d = linalg.eig(mat)
    alpha0 = atan2upper(d[1,0],d[0,0]) - pi/2
    alpha1 = atan2upper(d[1,1],d[1,0]) - pi/2
    if abs(alpha0)<abs(alpha1):
        alpha = -alpha0
    else:
        alpha = -alpha1
    # perform the actual transformation
    affine = array([[cos(alpha),-sin(alpha)],[sin(alpha),cos(alpha)]])
    ccenter = array((cx,cy))
    ocenter = array((shape[0]/2,shape[1]/2))
    offset = ccenter - dot(affine,ocenter)
    if normalize: img = image
    return ndimage.affine_transform(img,affine,
                                    offset=offset,
                                    output_shape=shape)

def normalize_skew(img,shape=(30,30),normalize=1):
    if type(img)==ListType:
        return [normalize_skew(image,shape=shape) for image in img]
    # rescale into 0-1 range
    image = normalize_range(img)
    # compute image statistics
    cx,cy,cxx,cxy,cyy = compute_stats(image)
    # note: we're using cxy/cxx because the coordinates are backwards (row,column)
    # rather than (x,y)
    alpha = cxy/cxx
    # alpha = -1 corrects a 45deg right slanted line (yup, indexes are backwards)
    affine = array([[1,0],[alpha,1]]) 
    ccenter = array((cx,cy))
    ocenter = array((shape[0]/2,shape[1]/2))
    offset = ccenter - dot(affine,ocenter)
    if normalize: img = image
    return ndimage.affine_transform(img,affine,
                                    offset=offset,
                                    output_shape=shape)

def fit_into(image,shape=(15,15),eps=0.2):
    x0,y0,x1,y1 = bbox(image,eps=eps)
    scale = 1.0/min(shape[0]*1.0/(x1-x0),shape[1]*1.0/(y1-y0))
    offset = array([(x1+x0)/2-scale*shape[0]/2,(y1+y0)/2-scale*shape[1]/2])
    affine = scale*array([[1,0],[0,1]]) 
    return ndimage.affine_transform(image,affine,
                                    offset=offset,
                                    output_shape=shape)

def fit_into_0(image,shape=(15,15),aa=1.0):
    w,h = image.shape
    scale = 1.0/min(shape[0]*1.0/w,shape[1]*1.0/h)
    image = scipy.ndimage.gaussian_filter(image,aa*scale)
    offset = array([(w-scale*shape[0])/2,(h-scale*shape[1])/2])
    affine = scale*array([[1,0],[0,1]]) 
    return scipy.ndimage.affine_transform(image,affine,
                                          offset=offset,
                                          output_shape=shape)

def fit_into_1(image,shape=(15,15),eps=0.2,aa=1.0,pad=2.0):
    image = crop_tight(image)
    w,h = image.shape
    scale = 1.0/min(shape[0]*1.0/w,shape[1]*1.0/h)
    image = pad_image(image,pad/scale)
    w,h = image.shape
    scale = 1.0/min(shape[0]*1.0/w,shape[1]*1.0/h)
    image = scipy.ndimage.gaussian_filter(image,aa*scale)
    offset = array([(w-scale*shape[0])/2,(h-scale*shape[1])/2])
    affine = scale*array([[1,0],[0,1]]) 
    return scipy.ndimage.affine_transform(image,affine,
                                          offset=offset,
                                          output_shape=shape)

def skew_correct(image,shape=(15,15),eps=0.2):
    image = normalize_skew(image,shape=image.shape)
    image = fit_into(image,shape=shape,eps=eps)
    return image

def box_correction(image,shape=(50,50),eps=0.2):
    x0,y0,x1,y1 = bbox(image,eps=eps)
    scale = 1.0/min(shape[0]*1.0/(x1-x0),shape[1]*1.0/(y1-y0))
    offset = array([(x1+x0)/2-scale*shape[0]/2,(y1+y0)/2-scale*shape[1]/2])
    affine = scale*array([[1,0],[0,1]])
    return offset,affine

def skew_correction(image):
    # rescale into 0-1 range
    image = normalize_range(image)
    # compute image statistics
    cx,cy,cxx,cxy,cyy = compute_stats(image)
    # note: we're using cxy/cxx because the coordinates are backwards (row,column)
    # rather than (x,y)
    alpha = cxy/cxx
    # alpha = -1 corrects a 45deg right slanted line (yup, indexes are backwards)
    affine = array([[1,0],[alpha,1]])
    center = array(image.shape)/2.0
    offset = center - dot(affine,center)
    return (offset,affine)

################################################################
### feature extraction
################################################################

def find_blobs(image,sigma):
    """Find the interior regions of an image (like the interior
    of the letter "o"."""
    w,h = image.shape
    smoothed = scipy.ndimage.gaussian_filter(image,sigma)
    threshold = (amax(smoothed)+amin(smoothed))/2
    labels,n = scipy.ndimage.label(smoothed<threshold)
    global slices
    slices = scipy.ndimage.find_objects(labels)
    for i in range(len(slices)):
        sx = slices[i][0]
        sy = slices[i][1]
        if sx.stop-sx.start>=w/2 and sy.stop-sy.start>=h/2:
            labels[labels==i+1] = 0
    blobs = 1.0*(labels!=0)
    return blobs

def find_deriv(image,sigma,direction):
    """Compute the derivative map of the image."""
    w,h = image.shape
    image = array(image,dtype='float')
    pad = int(10)
    image = extend_image(image,pad)
    if direction==0:
        dx = scipy.ndimage.gaussian_filter(image,sigma,order=(1,0))
        dx = crop_image(dx,pad)
        dx = abs(dx)
        s = amax(dx,axis=None)
        if s>0: dx /= s
        return dx
    if direction==1:
        dy = scipy.ndimage.gaussian_filter(image,sigma,order=(0,1))
        dy = crop_image(dy,pad)
        dy = abs(dy)
        s = amax(dy,axis=None)
        if s>0: dy /= s
        return dy
    raise "unknown direction"

def feature_maps(image,sigma):
    return [find_deriv(image,sigma,0),
            find_deriv(image,sigma,1),
            find_blobs(image,sigma=1.0)]

def show_maps(maps):
    pylab.clf()
    count = 1
    for fms,c in maps:
        for fm in fms:
            pylab.subplot(len(maps),len(fms),count)
            pylab.imshow(fm)
            count += 1

################################################################
# OCRopus-related file and image formats
################################################################

class OcrLine:
    """Given a basename or PNG text line file, provide access to the
    transcription, cseg, rseg, image, ctext (characters corresponding
    to labels), and cimages (character images)."""
    def __init__(self,base,id=None,old_csegs=0):
        self.old_csegs = old_csegs
        self.text = None
        base = re.sub(r'\.[^/]*$','',base)
        self.base = base
        self.id = None
        if id is None:
            match = re.search(r'.*/(Volume_[0-9][0-9][0-9][0-9]/.*)',base)
            if match: self.id = match.group(1)
    def path(self,ext):
        return self.base+"."+ext
    def image32(self,ext):
        return read_rgb32(self.path(ext))
    def get_transcription(self):
        if not self.text:
            with open(self.path("txt")) as stream:
                self.text = stream.read()
            self.text = re.sub(r'[\r\n]*$','',self.text)
        return self.text
    def get_cseg(self):
        return white_to_black(read_rgb32(self.path("cseg.png")))
    def get_rseg(self):
        return white_to_black(read_rgb32(self.path("rseg.png")))
    def get_image(self):
        return 1.0*(self.cseg!=0)
    def get_ctext(self):
        if self.old_csegs:
            return re.sub(r'\s','',self.transcription)
        else:
            return self.transcription
    def get_cimages(self):
        cimages = cseg_to_chars(self.cseg)
        assert len(cimages)==len(self.ctext)
        return cimages
    transcription = property(get_transcription)
    image = property(get_image)
    cseg = property(get_cseg)
    rseg = property(get_rseg)
    ctext = property(get_ctext)
    cimages = property(get_cimages)

def white_to_black(image):
    image[image==16777215] = 0
    return image

def read_cseg(file):
    image = read_pil(file)
    image = channels_to_rgb32(image)
    white_to_black(image)
    return image

def extract_cseg(cseg,i):
    """Extract the image with label i from the cseg image."""
    labels = scipy.ndimage.find_objects(cseg)
    slice = labels[i]
    print slice
    if slice:
        subimage = cseg[slice[0],slice[1],:]
        return subimage
    else:
        return None
    
def cseg_to_chars(cseg):
    """Extract all the individual characters from the cseg image."""
    if cseg.ndim==3: cseg = channels_to_rgb32(cseg)
    cseg = white_to_black(cseg)
    cseg = array(cseg,'uint8')
    labels = scipy.ndimage.find_objects(cseg)
    n = len(labels)
    result = [None]*n
    for i in range(n):
        label = i+1
        slice = labels[i]
        subimage = cseg[slice[0],slice[1]]
        assert (subimage==label).any()
        subimage[subimage!=label] = 0
        subimage[subimage==label] = 255
        subimage = array(subimage,dtype='uint8')
        result[i] = subimage
    return result

def array_in(a,l):
    """Given a list of integers, return an array that
    contains non-zero entries where the value of the original
    array a is contained in the list."""
    result = zeros(a.shape,dtype='b')
    for i in l: result += (a==i)
    return result
            
def extract_feature_maps(cseg,str,line=None,regex='[a-zA-Z0-9]',
                         sigma=2.0,cpad=1,pad=10,show=0):
    if cseg.ndim==3: cseg = channels_to_rgb32(cseg)
    if line==None: line = array(cseg!=0,'float')
    cseg = white_to_black(cseg)
    cseg = array(cseg,'uint8')
    cseg = pad_image(cseg,pad,value=0)
    line = extend_image(line,pad)
    maps = feature_maps(line,sigma)
    if show:
        pylab.clf()
        pylab.subplot(len(maps)+1,1,1)
        pylab.imshow(line)
        for i in range(len(maps)):
            pylab.subplot(len(maps)+2,1,i+2)
            pylab.imshow(maps[i])
        pylab.show()
    labels = scipy.ndimage.find_objects(cseg)
    n = min(len(str),len(labels))
    result = []
    for i in range(n):
        if labels[i]==None: continue
        label = i+1
        if not re.match(regex,str[i]): continue
        s = labels[i]
        hs = slice(s[0].start-cpad,s[0].stop+cpad)
        vs = slice(s[1].start-cpad,s[1].stop+cpad)
        assert (cseg[hs,vs]==label).any()
        fms = [fm[hs,vs] for fm in maps]
        result.append((fms,str[i]))
    return result
    
def char_features(cseg,components,line=None,
                  sigma=2.0,cpad=None,rpad=0,pad=10,show=0):
    """Extract feature maps corresponding to components.
    cseg: color segmentation of the input.
    components: list of lists of component numbers.
    line: optional grayscale image (if it isn't given, then
          the binary image corresponding to the segmentation is used).
    sigma: smoothing applied during feature extraction
    cpad: amount of padding applied to each pixel bitmask before extraction
    rpad: amount of extra padding around the component's bounding box
    pad: amount of padding around the entire feature map"""
    if cpad is None: cpad = int(2 * sigma)
    # prepare the segmentation image
    if cseg.ndim==3: cseg = channels_to_rgb32(cseg)
    cseg = white_to_black(cseg)
    cseg = array(cseg,'uint8')
    # if no grayscale line image is given, use the binarized segmentation
    if line==None: line = array(cseg!=0,'float')
    # pad everything
    line = extend_image(line,pad)
    cseg = pad_image(cseg,pad,value=0)
    # extract the feature maps
    maps = feature_maps(line,sigma)
    # optionally, display for debugging
    if show:
        pylab.clf()
        pylab.subplot(len(maps)+1,1,1)
        pylab.imshow(line)
        for i in range(len(maps)):
            pylab.subplot(len(maps)+2,1,i+2)
            pylab.imshow(maps[i])
        pylab.show()
    # compute bounding rectangles to speed up extraction (maybe later)
    #   labels = scipy.ndimage.find_objects(cseg)
    # extract the feature maps for each collection of components
    result = []
    for component in components:
        mask = array_in(cseg,component)
        if not mask.any():
            result.append(None)
            continue
        # pad the bitmap mask
        if cpad>0:
            mask = scipy.ndimage.binary_dilation(mask,iterations=cpad)
        # compute the bounding box for the character features
        x0,y0,x1,y1 = bbox(mask)
        hs = slice(x0-rpad,x1+rpad)
        vs = slice(y0-rpad,y1+rpad)
        fms = []
        submask = mask[hs,vs]
        for fm in maps:
            # extract and mask the features from each feature map
            sub = where(submask,fm[hs,vs],0.0)
            fms.append(sub)
        result.append(fms)
    # the result is a list of lists of feature maps
    assert len(result)==len(components)
    return result

def maps_to_vector(fms,shape=(15,15),aa=1.0):
    result = []
    for fm in fms:
        fit = fit_into(fm,shape,aa)
        result.append(fit.ravel())
    return concatenate(result)

def extract_feature_vectors(cseg,str,shape=(15,15),sigma=1.0,cpad=1):
    maps = extract_feature_maps(cseg,str,sigma=sigma,cpad=cpad)
    # show_maps(maps)
    return [(maps_to_vector(fms,aa=0.5,shape=shape),c) for fms,c in maps]

################################################################
# random stuff
################################################################

def main_extract_nist():
    test_labels = array(mnist_read("mnist/t10k-labels-idx1-ubyte.gz"))
    test_labels.dump("mnist-test-labels.dump")
    test_images = mnist_read("mnist/t10k-images-idx3-ubyte.gz")
    array(test_images).dump("mnist-test-images.dump")
    test_images = [skew_correct(image) for image in test_images]
    array(test_images).dump("mnist-test-images-deskewed.dump")
    train_labels = array(mnist_read("mnist/train-labels-idx1-ubyte.gz"))
    train_labels.dump("mnist-train-labels.dump")
    train_images = mnist_read("mnist/train-images-idx3-ubyte.gz")
    array(train_images).dump("mnist-train-images.dump")
    train_images = [skew_correct(image) for image in train_images]
    array(train_images).dump("mnist-train-images-deskewed.dump")

