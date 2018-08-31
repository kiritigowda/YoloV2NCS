import os, cv2
import csv

colornum = 12
colors = [(128,128,128),(128,0,0),(192,192,128),(255,69,0),(128,64,128),(60,40,222),(128,128,0),(192,128,128),(64,64,128),(64,0,128),(64,64,0),(0,128,192),(0,0,0)];

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

                cv2.rectangle(img_cp, (left,top), (right,bottom), clr, thickness=3)
                size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1, 2)
                width = size[0][0]
                height = size[0][1]
                cv2.rectangle(img_cp, (left,top - (height + 2)),((left + width),top),clr,-1)
                cv2.putText(img_cp,txt,(left,top),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.1,(20,20,20),2)
	return img_cp

