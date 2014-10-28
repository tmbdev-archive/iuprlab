from __future__ import with_statement

__all__ = """pil2numpy numpy2pil channels_to_rgb32 rgb32_to_channels
read_pil write_pil pil2string read_rgb32
bbox1 bbox join_channels
pad_image extend_image crop_image smart_pad_image crop_tight""".split()

import re
import numpy,scipy,scipy.ndimage
import cStringIO
from numpy import arange,where,amax,zeros
from PIL import Image

def base(filename):
    """Returns the last /-free portion of a string."""
    return re.sub(r'\.[^.]*$','',filename)

### image format conversions

def pil2numpy(image):
    """Convert a PIL image into a NumPy array."""
    modes = { "RGB" : numpy.uint8, "L"   : numpy.uint8, 
              "F"   : numpy.float32 }
    result = numpy.asarray(image,dtype=modes[image.mode])
    result = result.copy() # make it read/write
    return result

def numpy2pil(image,mode=None):
    """Converta NumPy array into a PIL image."""
    # result = Image.fromstring("RGB",(w,h),data,"raw",modes[0],
    #   input.widthStep,orientation)
    result = Image.fromarray(image,mode=mode)
    return result

def channels_to_rgb32(image):
    """Convert a NumPy array of shape [:,:,3] and dtype uint8
    into an intarray of shape [:,:] with 32 bit RGB values for
    each pixel."""
    assert image.ndim==3
    assert image.dtype=='uint8'
    image = numpy.array(image,'i')
    result = numpy.zeros(image.shape[0:2],'i')
    result += 1*image[:,:,2]
    result += 256*image[:,:,1]
    result += 65536*image[:,:,0]
    return result

def rgb32_to_channels(image):
    """Convert a NumPy array of shape [:,:] with RGB values
    packed into 32 bit integers to a NumPy uint8 array of
    shape [:,:,3]."""
    assert image.ndim==2
    assert image.dtype=='int32' or image.dtype=='uint32'
    result = numpy.zeros((image.shape[0],image.shape[1],3),'uint8')
    result[:,:,0] = image/65536
    result[:,:,1] = image/256
    result[:,:,2] = image
    return result

### image I/O

def write_pil(filename,image,type="png"):
    """Write a NumPy array as a PNG image using PIL."""
    image = numpy2pil(image)
    image.save(filename,type)

def read_pil(filename):
    """Read a NumPy array using PIL."""
    image = Image.open(filename)
    return pil2numpy(image)

def pil2string(image,format):
    """Convert a PIL image into a string."""
    output = cStringIO.StringIO()
    image.save(output,format)
    value = output.getvalue()
    output.close()
    return value

def read_rgb32(filename):
    """Read an image with PIL, then convert it to a uint32
    NumPy array with packed RGB values."""
    image = Image.open(filename)
    return channels_to_rgb32(pil2numpy(image))

### padding and extending images

def bbox1(a):
    """Compute 1D "bounding box" """
    indexes = arange(len(a)).compress(a!=0)
    lo = min(indexes)
    hi = max(indexes)
    return (lo,hi+1)
    
def bbox(image,eps=0.1):
    """Compute bounding box of non-zero pixels in image."""
    image = where(image<eps,0.0,1.0)
    hor = amax(image,axis=1)
    x0,x1 = bbox1(hor)
    vert = amax(image,axis=0)
    y0,y1 = bbox1(vert)
    return (x0,y0,x1,y1)

def join_channels(l):
    """Joint a list of rank 2 images into a single rank 3 image
    with the list index becoming the last index.  Useful in 
    constructs like 
    join_channels([process(image[:,:,i]) for i in range(3)])"""
    result = zeros(list(l[0].shape)+[len(l)],l[0].dtype)
    for i in range(len(l)):
        result[:,:,i] = l[i][:,:]
    return result

def pad_image(image,d,value=0):
    """Pad an image on all sides by d pixels."""
    if image.ndim==3:
        return join_channels([pad_image(image[:,:,i],d,value=value)
                              for i in range(image.shape[2])])
    w,h = image.shape
    result = zeros((w+2*d,h+2*d),image.dtype)
    if value!=0:
        result[:,:] = value
    result[d:w+d,d:h+d] = image
    return result

def extend_image(image,d):
    """Extend an image on all sides by d pixels."""
    if image.ndim==3:
        return join_channels([extend_image(image[:,:,i],d)
                              for i in range(image.shape[2])])
    w,h = image.shape
    result = zeros((w+2*d,h+2*d),image.dtype)
    result[d:w+d,d:h+d] = image
    for i in range(d):
        result[d:d+w,i] = image[:,0]
        result[d:d+w,h+2*d-i-1] = image[:,h-1]
        result[i,d:d+h] = image[0,:]
        result[w+2*d-i-1,d:d+h] = image[w-1,:]
    result[0:d,0:d] = image[0,0]
    result[w+d:w+2*d,0:d] = image[w-1,0]
    result[w+d:w+2*d,h+d:h+2*d] = image[w-1,h-1]
    result[0:d,h+d:h+2*d] = image[0,h-1]
    return result

def crop_image(image,d):
    """Crop an image on all sides by d pixels."""
    if image.ndim==3:
        w,h,c = image.shape
        return join_channels([image[d:w-d,d:h-d,i] for i in range(c)])
    else:
        w,h = image.shape
        return image[d:w-d,d:h-d].copy()

def crop_tight(image,eps=0.2):
    x0,y0,x1,y1 = bbox(image,eps=eps)
    return image[x0:x1,y0:y1]

def smart_pad_image(image,d,rounds=10):
    """Like extend image, but smoothes out the border a bit."""
    if image.ndim==3:
        return join_channels([smart_pad_image(image[:,:,i],d,rounds=rounds)
                              for i in range(image.ndim)])
    w,h = image.shape
    result = extend_image(image,d)
    for count in range(rounds):
        result[d:w+d,d:h+d] = image
        r = max(rounds - count,1)
        result = scipy.ndimage.gaussian_filter(result,r)
    result[d:w+d,d:h+d] = image
    return result

