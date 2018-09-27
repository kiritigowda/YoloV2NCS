from __future__ import division

import sys,os,time,csv,getopt,cv2,argparse
import numpy as np
from datetime import datetime
import csv

from ObjectWrapper import *
from Visualize import *

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest='image', type=str,
                        default='./images/dog.jpg', help='An image path.')
    parser.add_argument('--video', dest='video', type=str,
                        default='./videos/car.avi', help='A video path.')
    parser.add_argument('--imagefolder', dest='imagefolder', type=str,
                        default='./', help='A directory with images.')
    parser.add_argument('--capture', dest='capmode', type=int,
                        default=0, help='captute input from camera')
    parser.add_argument('--annpythonlib', dest='pyhtonlib', type=str,
                        default='./libannpython.so', help='pythonlib')
    parser.add_argument('--weights', dest='weightsfile', type=str,
                        default='./weights.bin', help='A directory with images.')    
    parser.add_argument('--resultsfolder', dest='resultfolder', type=str,
                        default='./', help='A directory with images.')
    args = parser.parse_args()

    outputdir = args.resultfolder
    weightsfile = args.weightsfile
    annpythonlib = args.pyhtonlib
    videoFile = args.video
    detector = AnnieObjectWrapper(annpythonlib, weightsfile)
    
    if sys.argv[1] == '--image':
        # image preprocess
        imagefile = args.image
        img = cv2.imread(imagefile)
        #data = np.asarray(img, dtype=np.float32)
        #f = open('inannie_file.f32', 'wb')
        #np.save(f, img)
        #f.close()
        start = datetime.now()

        results = detector.Detect(img)

        end = datetime.now()
        elapsedTime = end-start

        print ('total time is " milliseconds', elapsedTime.total_seconds()*1000)

        imdraw = Visualize(img, results)
        cv2.imshow('Demo',imdraw)
        cv2.imwrite('test.jpg',imdraw)
        exit()
    elif sys.argv[1] == '--imagefolder':
        imagedir  = args.imagefolder
        count = 0
        start = datetime.now()
        dictnames = ['input_image_name', 'x', 'y', 'width', 'height', 'confidence', 'class', 'class_name']
        csvFile = open('yolo_out.csv', 'w')
        with csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=dictnames)
            writer.writeheader()
            for image in sorted(os.listdir(imagedir)):
                print('Processing Image ' + image)
                img = cv2.imread(imagedir + image)
                results = detector.Detect(img)
                for i in range(len(results)):
                    #writer.writerow({'input_image_name': 'output_' + str(count) + '.jpg', 'x': results[i].bbox.x, 'y': results[i].bbox.y, 'width': results[i].bbox.w, \
                    #'height': results[i].bbox.h, 'confidence': "{:.5f}".format(results[i].confidence), 'class': results[i].objType, 'class_name':results[i].name})
                    writer.writerow({'input_image_name': image, 'x': "{:.5f}".format(results[i].x), 'y': "{:.5f}".format(results[i].y), \
                    'width': "{:.5f}".format(results[i].width), 'height': "{:.5f}".format(results[i].height), 'confidence': "{:.5f}".format(results[i].confidence), 'class': results[i].objType, 'class_name':results[i].name})
                imdraw = Visualize(img, results)

                #cv2.imshow('Demo',imdraw)
                cv2.imwrite(outputdir + 'yolo-output_' + str(count) + '.jpg',imdraw)
                count += 1
        end = datetime.now()
        elapsedTime = end-start
        print ('total time is " milliseconds', elapsedTime.total_seconds()*1000)
        exit()
    elif sys.argv[1] == '--video':
        # video preprocess
        print ('Video File')
        capmode = args.capmode    
        cap = cv2.VideoCapture(videoFile)
        assert cap.isOpened(), 'Cannot capture source'    
        frames = 0
        start = time.time()
        while cap.isOpened(): 
            ret, frame = cap.read()
            if ret:
                #frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (416, 416))           
                results = detector.Detect(frame)
                imdraw = Visualize(frame, results)
                cv2.imshow('AMD YoloV2 Video File', imdraw)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                frames += 1
                if (frames % 16 == 0):
                    print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            else:
                break
        exit()
    elif sys.argv[1] == '--capture':
        print ('Capturing Live')
        capmode = args.capmode    
        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), 'Cannot capture source'    
        frames = 0
        start = time.time()
        while cap.isOpened(): 
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)                
                results = detector.Detect(frame)
                imdraw = Visualize(frame, results)
                cv2.imshow('AMD YoloV2 Live', imdraw)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                frames += 1
                if (frames % 16 == 0):
                    print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            else:
                break
        exit()
