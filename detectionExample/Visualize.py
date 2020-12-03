from __future__ import division
import os, cv2
import csv
import shutil
from distutils.dir_util import copy_tree
import subprocess
import numpy as np

colornum = 20
colors =    [
            (160,82,45),        # aeroplane
            (128,0,0),          # bicycle
            (47,79,79),         # bird
            (155,140,145),      # boat
            (140,155,255),      # bottle
            (255,105,180),      # bus
            (255,0,0),          # car
            (75,0,130),         # cat
            (255,140,0),        # chair
            (250,128,114),      # cow
            (153,50,204),       # diningtable
            (130,230,150),      # dog
            (0,220,255),        # horse
            (0,191,255),        # motorbike
            (0,0,255),          # person
            (0,255,255),        # potted plant
            (107,142,35),       # sheep
            (0,128,0),          # sofa
            (124,252,0),        # train
            (199,21,133)        # tvmonitor
            ];

CLASSES =   [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "dining table",
            "dog",
            "horse",
            "motorbike",
            "person",
            "potted plant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor"
            ]


def Visualize(img, results):
	img_cp = img.copy()
	detectedNum = len(results)
	if detectedNum > 0:
            for i in range(detectedNum):
                
                clr = colors[results[i].objType % colornum]
                txt = results[i].name

                left = results[i].left
                top = results[i].top
                right = results[i].right
                bottom = results[i].bottom
                confidence = results[i].confidence
                confidence = confidence * 100
                confidence = format(confidence,'.2f') 
                txt = txt+' '+str(confidence)+'%'
                cv2.rectangle(img_cp, (left,top), (right,bottom), clr, thickness=3)
                size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)
                width = size[0][0]
                height = size[0][1]

                cv2.rectangle(img_cp, (left,(bottom-5) - (height + 2)),((left + width),(bottom-5)),clr,-1)
                cv2.putText(img_cp,txt,(left,(bottom-5)),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(20,20,20),1)
	return img_cp




def VisualizeCamera(img, results):
    img_cp = img.copy()
    #print "type = ", type(img_cp)
    crop_imgs = []
    count = 0
    detectedNum = len(results)
  
    if detectedNum > 0:
            for i in range(detectedNum):
                
                clr = colors[results[i].objType % colornum]
                txt = results[i].name

                left = results[i].left
                top = results[i].top
                right = results[i].right
                bottom = results[i].bottom
                confidence = results[i].confidence
                confidence = confidence * 100
                confidence = format(confidence,'.2f') 
                txt = txt+' '+str(confidence)+'%'

                #print left, " ", top , " ", right , " ", bottom
 
                h = bottom - top
                w = right - left
                x1 = left - 5
                x2 = left + h+ 5
                y1 = top -5
                y2 = top+w+5
                crop_img = img_cp[y1:y2, x1:x2]

                #print img_cp.size
                #print crop_img.size
                if crop_img.size > 57600:
                    cv2.rectangle(img_cp, (x1,y1), (x2,y2), clr, thickness=1)
                    size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)
                    width = size[0][0]
                    height = size[0][1]

                    cv2.rectangle(img_cp, (x1,(y2-5) - (height + 2)),((x1 + width),(y2-5)),clr,-1)
                    cv2.putText(img_cp,txt,(x1,(y2-5)),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(20,20,20),1)
                    """
                    path = os.path.join(outputdir ,  'yolo-output_'+ str(count) + '.jpg')
                    #cv2.imshow('AMD YoloV2 Live', crop_img)
                    cv2.imwrite(path,crop_img)
                    count += 1
                    """
                    crop_imgs.append(crop_img)
    
    return crop_imgs

def VisualizeBox(image, img, results, anniedir):
    img_cp = img.copy()
    #print "type img_cp= ", type(img_cp)
    crop_imgs = []
    count = 0
    detectedNum = len(results)
  
    if detectedNum > 0:
            for i in range(detectedNum):
                
                clr = colors[results[i].objType % colornum]
                txt = results[i].name

                left = results[i].left
                top = results[i].top
                right = results[i].right
                bottom = results[i].bottom
                confidence = results[i].confidence
                confidence = confidence * 100
                confidence = format(confidence,'.2f') 
                txt = txt+' '+str(confidence)+'%'

                orig_image = cv2.imread(anniedir + image)
                        #print left, " ", top , " ", right , " ", bottom
                #print "type orig_image= ", type(orig_image)    
                x1 = int((left/416) * 1024)
                x2 = int((right/416) * 1024)
                y1 = int((top/416) * 1024)
                y2 = int((bottom/416) * 1024)
                """
                h = bottom - top
                w = right - left
                x1 = left - 5
                x2 = left + h+ 5
                y1 = top -5
                y2 = top+w+5
                """
                crop_img = orig_image[y1:y2, x1:x2]

                #print img_cp.size
                #print crop_img.size
                
                cv2.rectangle(orig_image, (x1,y1), (x2,y2), clr, thickness=3)
                size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)
                width = size[0][0]
                height = size[0][1]

                cv2.rectangle(orig_image, (x1,(y2-5) - (height + 2)),((x1 + width),(y2-5)),clr,-1)
                cv2.putText(orig_image,txt,(x1,(y2-5)),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(20,20,20),1)
                """
                path = os.path.join(outputdir ,  'yolo-output_'+ str(count) + '.jpg')
                #cv2.imshow('AMD YoloV2 Live', crop_img)
                cv2.imwrite(path,crop_img)
                count += 1
                """
                crop_imgs.append(crop_img)
    
    return crop_imgs

