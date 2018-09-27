from __future__ import division
import os, cv2
import csv

colornum = 20
colors = [(160,82,45),(128,0,0),(47,79,79),(255,240,245),(240,255,255),(255,105,180),(255,255,0),(75,0,130),(153,50,204),(230,230,250),(0,0,255),(0,191,255),(0,128,128),(0,255,255),(107,142,35),(0,128,0),(124,252,0),(199,21,133),(255,140,0),(250,128,114)];

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
                cv2.rectangle(img_cp, (left,top), (right,bottom), clr, thickness=1)
                size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)
                width = size[0][0]
                height = size[0][1]

                cv2.rectangle(img_cp, (left,(bottom-5) - (height + 2)),((left + width),(bottom-5)),clr,-1)
                cv2.putText(img_cp,txt,(left,(bottom-5)),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(20,20,20),1)
	return img_cp

