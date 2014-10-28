# just run this with

import cherrypy

import glob,os,sys,string,re,math,random
import numpy,scipy,scipy.ndimage
import cStringIO
from PIL import Image
from imageutil import *
from ocrutil import *


_html = ""
def clear():
    global _html
    _html = ""
def html():
    global _html
    return _html
def out(*args):
    global _html
    for arg in args:
        _html += arg
    _html += "\n"

def fclean(file):
    file = re.sub(r'/[.][.]*','/',file)
    file = re.sub(r'[.][.]*/','/',file)
    file = re.sub(r'[^/_0-9a-zA-Z.-]','',file)
    file = re.sub(r'^/*','',file)
    return file

def read_file(file):
    stream = open(file,"r")
    data = stream.read()
    stream.close()
    return data

class Display:
    # TODO replace this with CherryPy's static file server
    @cherrypy.expose
    def send(self,file=None,type="image/jpeg"):
        file = fclean(file)
        stream = open(file,"r")
        data = stream.read()
        stream.close()
        cherrypy.response.headers['Content-Type'] = type
        return data

    @cherrypy.expose
    def recolor(self,file=None):
        file = fclean(file)
        image = read_rgb32(file)
        image = numpy.array(image,'uint8')
        image = numpy2pil(image,mode='L')
        palette = []
        palette.extend((0,0,0))
        for i in range(1,255):
            r = int(abs(math.sin(i))*220+30)
            g = int(abs(math.sin(i*1.3))*220+30)
            b = int(abs(math.sin(i/1.7))*220+30)
            palette.extend((r,g,b))
        palette.extend((0,0,0))
        assert len(palette)==768
        image.putpalette(palette)
        cherrypy.response.headers['Content-Type'] = "image/png"
        return pil2string(image,"png")

    @cherrypy.expose
    def extract(self,file=None,index=None):
        file = fclean(file)
        index = int(index)
        image = read_pil(file)
        subimage = extract_cseg(image,index)
        if subimage!=None:
            subimage = numpy2pil(subimage)
        else:
            subimage = Image.new("RGB",(11,11),(255,0,0))
        cherrypy.response.headers['Content-Type'] = "image/png"
        return pil2string(subimage,"png")


    @cherrypy.expose
    def index(self):
        clear()
        out("<h1>Index</h1>")
        volumes = glob.glob("Volume*")
        volumes.sort()
        for vol in volumes:
            out("<h2>%s</h2>"%vol)
            pages = glob.glob("%s/????"%vol)
            pages.sort()
            for page in pages:
                match = re.search(r'/(\d+)$',page)
                pageno = int(match.group(1))
                out("<a href='dir?dir=%s'>%d</a> "%(page,pageno))
        return html()

    @cherrypy.expose
    def dir(self,dir=None):
        dir = fclean(dir)
        clear()
        out("<h1>Text Lines [%s]</h1>"%dir)
        for file in glob.glob("%s/[0-9][0-9][0-9][0-9].png"%dir):
            base = re.sub(r'.png','',file)
            # out("<img src='send?file=%s.jpg'><br>"%base)
            out("<a href='line?file=%s.png'>"%base)
            out("<img src='recolor?file=%s.cseg.png'><br>"%base)
            out("</a>")
            out("<p>")
        return html()

    @cherrypy.expose
    def line(self,file=None):
        file = fclean(file)
        clear()
        base = re.sub(r'.png','',file)
        transcription = open("%s.txt"%base).read()
        transcription = re.sub(r'[ ]','',transcription)
        n = numpy.amax(white_to_black(read_pil("%s.cseg.png"%base)))
        l = list(range(15,26))
        out("<h2>%s</h2>"%file)
        out("%d components<p>"%n)
        out("<table border=1>")
        if os.path.exists(base+".jpg"):
            out("<img src='send?file=%s.jpg'><p>"%base)
        if os.path.exists(base+".png"):
            out("<img src='send?file=%s'><p>"%file)
        if os.path.exists(base+".rseg.png"):
            out("<img src='recolor?file=%s.rseg.png'><p>"%base)
        if os.path.exists(base+".cseg.png"):
            out("<img src='recolor?file=%s.cseg.png'><p>"%base)
        if os.path.exists(base+".cut_dgb.png"):
            out("<img src='send?file=%s.png.cut_dbg.png'><p>"%base)
        out("<tr>")
        for i in range(min(n,200)):
            out("<td>")
            out("<img src='extract?file=%s.cseg.png&index=%d'>"%(base,i))
            out("</td>")
        out("</tr>")
        out("<tr>")
        for i in range(min(n,200)):
            out("<td>")
            try:
                out("<font size=6>%s</font>"%transcription[i])
            except:
                out("<font size=6>ERR</font>")
            out("</td>")
        out("</tr>")
        out("<tr>")
        if os.path.exists(base+".costs"):
            stream = open(base+".costs")
            for line in stream.readlines():
                key,index,cost = line.split()
                cost = float(cost)
                out("<td>%s</td>"%("|"*int(math.floor(cost))))
        out("</tr>")
        out("<tr>")
        if os.path.exists(base+".costs"):
            stream = open(base+".costs")
            for line in stream.readlines():
                key,index,cost = line.split()
                cost = float(cost)
                out("<td>%.2f</td>"%cost)
        out("</tr>")
        out("</table>")
        if os.path.exists(base+".align.log"):
            out("<pre>")
            out(read_file(base+".align.log"))
            out("</pre>")
        return html()

config = {
    "server.socket_port": 9999,
    "server.thread_pool": 10,
    "server.environment": "development",
    "server.showTracebacks": True,
}
# cherrypy.config.update(config)

if __name__ == "__main__":
    root = "/home/tmb/data/g1000w"
    if len(sys.argv)>1: root = sys.argv[1]
    sys.argv[0] = os.path.abspath(sys.argv[0])
    cherrypy.engine.reload_files += [ os.path.abspath(sys.argv[0]) ]

    cherrypy.config["server.socket_port"] = 9999
    cherrypy.config["server.showTracebacks"] = True
    os.chdir(root)
    cherrypy.quickstart(Display())

