from libpydetector import YoloDetector
import os, io, numpy, time, ctypes, array
import numpy as np
from skimage.transform import resize
from ctypes import cdll, c_char_p
from numpy.ctypeslib import ndpointer
import csv

# AnnInferenceLib = ctypes.cdll.LoadLibrary('/home/rajy/work/yolov2/build/libannmodule.so')
# inf_fun = AnnInferenceLib.annRunInference
# inf_fun.restype = ctypes.c_int
# inf_fun.argtypes = [ctypes.c_void_p,
#                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
#                ctypes.c_size_t,
#                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
#                ctypes.c_size_t]

class AnnAPI:
    def __init__(self,library):
        self.lib = ctypes.cdll.LoadLibrary(library)
        self.annQueryInference = self.lib.annQueryInference
        self.annQueryInference.restype = ctypes.c_char_p
        self.annQueryInference.argtypes = []
        self.annCreateInference = self.lib.annCreateInference
        self.annCreateInference.restype = ctypes.c_void_p
        self.annCreateInference.argtypes = [ctypes.c_char_p]
        self.annReleaseInference = self.lib.annReleaseInference
        self.annReleaseInference.restype = ctypes.c_int
        self.annReleaseInference.argtypes = [ctypes.c_void_p]
        self.annCopyToInferenceInput = self.lib.annCopyToInferenceInput
        self.annCopyToInferenceInput.restype = ctypes.c_int
        self.annCopyToInferenceInput.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t, ctypes.c_bool]
        self.annCopyFromInferenceOutput = self.lib.annCopyFromInferenceOutput
        self.annCopyFromInferenceOutput.restype = ctypes.c_int
        self.annCopyFromInferenceOutput.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t]
        self.annRunInference = self.lib.annRunInference
        self.annRunInference.restype = ctypes.c_int
        self.annRunInference.argtypes = [ctypes.c_void_p, ctypes.c_int]
        print('OK: AnnAPI found "' + self.annQueryInference().decode("utf-8") + '" as configuration in ' + library)


class BBox(object):
    def __init__(self, bbox, xscale, yscale, offx, offy):
        self.left = int(bbox.left / xscale)-offx
        self.top = int(bbox.top / yscale)-offy
        self.right = int(bbox.right / xscale)-offx
        self.bottom = int(bbox.bottom / yscale)-offy
        self.x = bbox.x;
        self.y = bbox.y;
        self.width = bbox.width;
        self.height = bbox.height;
        self.confidence = bbox.confidence
        self.objType = bbox.objType
        self.name = bbox.name

class AnnieObjectWrapper():
    def __init__(self, annpythonlib, weightsfile):
        select = 1
        self.detector = YoloDetector(select)
        self.api = AnnAPI(annpythonlib)
        input_info,output_info = self.api.annQueryInference().decode("utf-8").split(';')
        input,name,ni,ci,hi,wi = input_info.split(',')
        self.hdl = self.api.annCreateInference(weightsfile.encode('utf-8'))
        self.dim = (int(hi),int(wi))
        self.blockwd = 12
        self.wh = self.blockwd*self.blockwd
        self.targetBlockwd = 13
        self.classes = 20
        self.threshold = 0.18
        self.nms = 0.4

    def __del__(self):
        self.api.annReleaseInference(self.hdl)


    def PrepareImage(self, img, dim):
        imgw = img.shape[1]
        imgh = img.shape[0]
        imgb = np.empty((dim[0], dim[1], 3))
        imgb.fill(0.5)

        if imgh/imgw > dim[1]/dim[0]:
            neww = int(imgw * dim[1] / imgh)
            newh = dim[1]
        else:
            newh = int(imgh * dim[0] / imgw)
            neww = dim[0]
        offx = int((dim[0] - neww)/2)
        offy = int((dim[1] - newh)/2)

        imgb[offy:offy+newh,offx:offx+neww,:] = resize(img.copy()/255.0,(newh,neww),1)
        #print('INFO:: newW:%d newH:%d offx:%d offy: %d elem0:%.5f elem1:%.5f elem2:%.5f' % (neww, newh, offx, offy, imgb[offy,offx+1,0], imgb[offy,offx+1,1], imgb[offy,offx+1,2]))
        im = imgb[:,:,(2,1,0)]
        return im, int(offx*imgw/neww), int(offy*imgh/newh), neww/dim[0], newh/dim[1]

    def Reshape(self, out, dim):
        shape = out.shape
        out = np.transpose(out.reshape(self.wh, int(shape[0]/self.wh)))  
        out = out.reshape(shape)
        return out

    def runInference(self,img, out):
        imgw = img.shape[1]
        imgh = img.shape[0]
        #convert image to tensor format (RGB in seperate planes)
        img_r = img[:,:,0]
        img_g = img[:,:,1]
        img_b = img[:,:,2]
        img_t = np.concatenate((img_r, img_g, img_b), 0)
        status = self.api.annCopyToInferenceInput(self.hdl, np.ascontiguousarray(img_t, dtype=np.float32), (img.shape[0]*img.shape[1]*3*4), 0)
        #print('INFO: annCopyToInferenceInput status %d'  %(status))
        status = self.api.annRunInference(self.hdl, 1)
        #print('INFO: annRunInference status %d ' %(status))
        status = self.api.annCopyFromInferenceOutput(self.hdl, np.ascontiguousarray(out, dtype=np.float32), out.nbytes)
        #print('INFO: annCopyFromInferenceOutput status %d' %(status))
        return out

    def Detect(self, img):
        imgw = img.shape[1]
        imgh = img.shape[0]
        im,offx,offy,xscale,yscale = self.PrepareImage(img, self.dim)
        out_buf = bytearray(12*12*125*4)
        out = np.frombuffer(out_buf, dtype=numpy.float32)
        output = self.runInference(im, out)
        internalresults = self.detector.Detect(output.astype(np.float32), int(output.shape[0]/self.wh), self.blockwd, self.blockwd, self.classes, imgw, imgh, self.threshold, self.nms, self.targetBlockwd)
        pyresults = [BBox(x,xscale,yscale, offx, offy) for x in internalresults]
        return pyresults
