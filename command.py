import pylab,numpy

__all__ = """
rscatter imgray imshows
""".split()

################################################################
# useful commands for importing into the toplevel
################################################################

def rscatter(data):
    pylab.scatter(data[:,0],data[:,1])
    pylab.show()

def imgray(image):
    pylab.clf()
    pylab.imshow(1.0*image,cmap=pylab.cm.gray,
                 interpolation="nearest",origin="upper")
    pylab.show()

def imshows(images):
    pylab.clf()
    n = len(images)
    if n>36: n=36
    r = floor(0.9+sqrt(n))
    index = 0
    for i in range(n):
        image = images[i]
        pylab.subplot(r,r,i+1)
        pylab.imshow(1.0*image,cmap=pylab.cm.gray,
                     interpolation="nearest",origin="upper")
    pylab.show()

