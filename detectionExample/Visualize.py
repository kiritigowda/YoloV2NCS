from __future__ import division
import os, cv2
import csv
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
                confidence = int(confidence * 100)
                #confidence = format(confidence,'.2f')
                txt = txt+' '+str(confidence)+'%'
                cv2.rectangle(img_cp, (left,top), (right,bottom), clr, thickness=2)
                size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                width = size[0][0] + 10
                height = size[0][1]

                cv2.rectangle(img_cp, (left, (bottom-5) - (height+5)), ((left + width), (bottom-5)),clr,-1)
                cv2.putText(img_cp,txt,((left + 5),(bottom-5)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
	return img_cp

def LegendImage():
    window_name = "AMD Object Detection - Legend"
    # initialize the legend visualization
    legend = np.zeros(((len(CLASSES) * 25), 200, 3), dtype="uint8")
    legend.fill(255)
    # loop over the class names + colors
    for (i, (className, color)) in enumerate(zip(CLASSES, colors)):
        # draw the class name + color on the legend
        color = [int(c) for c in color]
        cv2.putText(legend, className, (5, (i * 25) + 17),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.rectangle(legend, (125, (i * 25)), (200, (i * 25) + 25),
        tuple(color), -1)
    #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, legend)
