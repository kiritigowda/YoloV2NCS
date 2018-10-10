"""
Main_v3 + trying cascaded yolo+classification (cropped)
"""
from __future__ import division

import sys,os,time,csv,getopt,cv2,argparse
import numpy as np
from datetime import datetime
import csv
from operator import itemgetter
import shutil
import subprocess

from distutils.dir_util import copy_tree
#from resizeimage import resizeimage

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from ObjectWrapper import *
from Visualize import *
#from work.kiriti-git.annie-capture-demo.AnnieCapture import *

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

class App(QWidget):
 
	def __init__(self):
		super(App, self).__init__()
		self.initUI()

	def initUI(self):

		self.title = 'OBJECT DETECTION'
		self.setWindowTitle(self.title)
		cwd = os.getcwd() 
		self.setWindowIcon(QIcon(cwd + '/detectionExample/icons/amd-logo-150x150.jpg'))
				
		#pixmap = QPixmap(cwd + '/detectionExample/icons/amd-logo.jpg')

		# Create widgets
		self.label  = QLabel('Most Recent Objects Detected', self)		
		self.label.setStyleSheet("font: bold 14pt TIMES NEW ROMAN")
		self.label.setWordWrap(True)
		self.label.setAlignment(Qt.AlignCenter) 
		#self.imageWidget = QLabel()
		#self.imageWidget.setPixmap(pixmap)
		self.tableWidget = QTableWidget()

		self.layout = QGridLayout()
		self.layout.setSpacing(5)
		#self.layout.addWidget(self.imageWidget,1,0)
		self.layout.addWidget(self.label,2,0)
		self.layout.addWidget(self.tableWidget,3,0) 

		self.setLayout(self.layout) 
		
		self.setGeometry(0, 0, 300, 300)

	def createTable(self,data):
		
		self.show()

		

		self.tableWidget.setRowCount(0)
		self.tableWidget.setColumnCount(2)
		self.tableWidget.setHorizontalHeaderLabels(['Name', 'Confidence'])
		for row_number in xrange(0,min(5,len(data))):
			#print data[row_number]
			row_number =  self.tableWidget.rowCount()
			self.tableWidget.insertRow(row_number)
			self.tableWidget.setItem(row_number, 0, QTableWidgetItem(str(data[row_number][1])))
			conf = str(int(round(float(data[row_number][2]), 4)*100)) + "%"
			self.tableWidget.setItem(row_number, 1, QTableWidgetItem(conf))
		self.tableWidget.resizeColumnsToContents()
		#self.tableWidget.resizeRowsToContents()
		
		# Show widget
		self.show()

def image_function():
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


def imagefolder_function(imagedir, outputdir):
	count = 0
	count_cascade = 0
	start = datetime.now()

	cascadeFile = os.getcwd() + '/YoloOutCascadeClassfication'
	if os.path.exists(cascadeFile):
		shutil.rmtree(cascadeFile)
	os.makedirs(cascadeFile)

	dictnames = ['input_image_name', 'x', 'y', 'width', 'height', 'confidence', 'class', 'class_name']
	csvFile = open('yolo_out.csv', 'w')
	with csvFile:
	  
		writer = csv.DictWriter(csvFile, fieldnames=dictnames)
		writer.writeheader()

		for image in sorted(os.listdir(imagedir)):

			print('Processing Image ' + image)
			img = cv2.imread(imagedir + image)
			results = detector.Detect(img) 
			imdraw = Visualize(img, results)
			imdraw_crop = VisualizeBox(img,results,cascadeFile)
			path = os.path.join(outputdir ,  'yolo-output_'+ str(count) + '.jpg')

			#time.sleep(0.5)
			cv2.imshow('AMD YoloV2 Live', imdraw)
			time.sleep(0.5)
			#cv2.waitKey(1)
			for i in range(len(results)):
				#writer.writerow({'input_image_name': 'output_' + str(count) + '.jpg', 'x': results[i].bbox.x, 'y': results[i].bbox.y, 'width': results[i].bbox.w, \
				#'height': results[i].bbox.h, 'confidence': "{:.5f}".format(results[i].confidence), 'class': results[i].objType, 'class_name':results[i].name})
				writer.writerow({'input_image_name': image, 'x': "{:.5f}".format(results[i].x), 'y': "{:.5f}".format(results[i].y), \
				'width': "{:.5f}".format(results[i].width), 'height': "{:.5f}".format(results[i].height), 'confidence': "{:.5f}".format(results[i].confidence), 'class': results[i].objType, 'class_name':results[i].name})
			cv2.imwrite(path,imdraw)

			for img in imdraw_crop:
				if img.size != 0:
					img = cv2.resize(img, dsize=(299,299), interpolation = cv2.INTER_CUBIC)
					path = os.path.join(cascadeFile ,  'yolo-output_'+ str(count_cascade) + '.jpg')
					cv2.imwrite(path,img)
					count_cascade += 1

			count += 1
			newKey = cv2.waitKey(1)
			if newKey & 0xFF == ord('c'):
				camera_function()
			
			if newKey == 32:
				if cv2.waitKey(0) == 32:
					continue

			if newKey & 0xFF == ord('q'):
				exit()

			#if newKey & 0xFF == ord('x'):
		
		end = datetime.now()
		elapsedTime = end-start
		print ('total time is " milliseconds', elapsedTime.total_seconds()*1000)
		currentDirectory = os.getcwd()
		fromDirectory = os.getcwd() + '/YoloOutCascadeClassfication'
		
		os.chdir("../annie-capture-demo/")
		toDirectory = os.getcwd() + '/YoloOutCascadeClassfication'
		if os.path.exists(toDirectory):
			shutil.rmtree(toDirectory)
		os.makedirs(toDirectory)
		copy_tree(fromDirectory, toDirectory)
		subprocess.call(['python', 'AnnieCapture.py', '--imagefolder' , 'YoloOutCascadeClassfication/' , '--pm' ,'1', '--pa', '0'])
		os.chdir(currentDirectory)


def camera_function():
	print ('Capturing Live')

	outputdir = os.getcwd() + '/YoloOutCascadeClassfication'
	if os.path.exists(outputdir):
		shutil.rmtree(outputdir)
	os.makedirs(outputdir)

	capmode = args.capmode    
	cap = cv2.VideoCapture(0)
	assert cap.isOpened(), 'Cannot capture source'    
	frames = 0
	count = 0
	start_cam = time.time()
	
	dictnames = ['class','class_name','confidence', 'x', 'y', 'width', 'height']
	with open('history_file.csv', 'w+') as csvHistoryFile:
		writer = csv.DictWriter(csvHistoryFile, fieldnames=dictnames)
		writer.writeheader()
		while cap.isOpened(): 
			ret, frame = cap.read()
			if ret:
				frame = cv2.flip(frame, 1)                
				results = detector.Detect(frame)
				imdraw = Visualize(frame, results)
				imdraw_crop = VisualizeBox(frame,results,outputdir)
				cv2.imshow('AMD YoloV2 Live', imdraw)

				for i in range(len(results)):
					writer.writerow({'class': results[i].objType, 'class_name':results[i].name ,'confidence': "{:.5f}".format(results[i].confidence), 'x': "{:.5f}".format(results[i].x), 'y': "{:.5f}".format(results[i].y), \
					'width': "{:.5f}".format(results[i].width), 'height': "{:.5f}".format(results[i].height)})

				frames += 1
				if (frames % 16 == 0):
					print("FPS of the video is {:5.2f}".format( frames / (time.time() - start_cam)))

					for img in imdraw_crop:
						if img.size != 0:
							img = cv2.resize(img, dsize=(299,299), interpolation = cv2.INTER_CUBIC)
							path = os.path.join(outputdir ,  'yolo-output_'+ str(count) + '.jpg')
							cv2.imwrite(path,img)
							count += 1
					
					csvHistoryFile.seek(0)
					resultCSV = csv.reader(csvHistoryFile)
					next(resultCSV,None) # skip header
					resultDataBase = [r for r in resultCSV]
					numElements = len(resultDataBase)

					resultDataBase = [row for row in resultDataBase if row and len(row) == 7]
					
					if numElements > 0:	
						
						resultDataBase = sorted(resultDataBase, key=itemgetter(2), reverse = True)
						
						for i in xrange(len(resultDataBase) - 1, 0 , -1):
							if resultDataBase[i][0] == resultDataBase[i-1][0]:
								del resultDataBase[i]
						
						qt_tryme.createTable(resultDataBase)

						csvHistoryFile.seek(0)
						writer = csv.DictWriter(csvHistoryFile, fieldnames=dictnames)
						writer.writeheader()
			  
				key = cv2.waitKey(1)
				if key & 0xFF == ord('q'):
					break

				if key & 0xFF == ord('1'):
					cap.release()
					imagedir  = os.getcwd() + '/images_1/'
					outputdir = os.getcwd() + '/outputFolder_1/'
					if not os.path.exists(outputdir):
						os.makedirs(outputdir)
					imagefolder_function(imagedir, outputdir)
			   
				if key & 0xFF == ord('2'):
					cap.release()
					imagedir  = os.getcwd() + '/images_2/'
					outputdir = os.getcwd() + '/outputFolder_2/'
					if not os.path.exists(outputdir):
						os.makedirs(outputdir)
					imagefolder_function(imagedir, outputdir)

				if key == 32:
					if cv2.waitKey(0) == 32:
						continue

				if key & 0xFF == ord('f'):
					cap.release()
					imagedir  = os.getcwd() + '/images/'
					outputdir = os.getcwd() + '/outputFolder_3/'
					if not os.path.exists(outputdir):
						os.makedirs(outputdir)
					flag = True
					while (flag):
						imagefolder_function(imagedir, outputdir)
			   
				if key & 0xFF == ord('x'):
					currentDirectory = os.getcwd()
					fromDirectory = os.getcwd() + '/YoloOutCascadeClassfication'
					
					os.chdir("../annie-capture-demo/")
					toDirectory = os.getcwd() + '/YoloOutCascadeClassfication'
					if os.path.exists(toDirectory):
						shutil.rmtree(toDirectory)
					os.makedirs(toDirectory)
					copy_tree(fromDirectory, toDirectory)
					subprocess.call(['python', 'AnnieCapture.py', '--imagefolder' , 'YoloOutCascadeClassfication/' , '--pm' ,'1', '--pa', '0'])
					os.chdir(currentDirectory)
				
			else:
				break
		exit()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image', dest='image', type=str,
						default='./images/dog.jpg', help='An image path.')
	parser.add_argument('--imagefolder', dest='imagefolder', type=str,
						default='./images/' + 'images', help='A directory with images.')
	parser.add_argument('--video', dest='video', type=str,
						default='./test1.mp4.avi', help='A video path.')
	parser.add_argument('--capture', dest='capmode', type=int,
						default=0, help='capture input from camera')
	parser.add_argument('--annpythonlib', dest='pythonlib', type=str,
						default='./libannpython.so', help='pythonlib')
	parser.add_argument('--weights', dest='weightsfile', type=str,
						default='./weights.bin', help='A directory with images.')    
	parser.add_argument('--resultsfolder', dest='resultfolder', type=str,
						default=os.getcwd() + '/outputFolder', help='A directory with images.')
	parser.add_argument('--imageWidth', dest='imageWidth', type=int,
						default=416, help='YoloV2 image width')
	parser.add_argument('--imageHeight', dest='imageHeight', type=int,
						default=416, help='YoloV2 image height')
	args = parser.parse_args()

	 
   

	weightsfile = args.weightsfile
	annpythonlib = args.pythonlib
	detector = AnnieObjectWrapper(annpythonlib, weightsfile)
	videoFile = args.video
	imageWidth = args.imageWidth
	imageHeight = args.imageHeight

	app = QApplication(sys.argv)
	qt_tryme = App()
	
	if sys.argv[1] == '--image':
		image_function()
	elif sys.argv[1] == '--imagefolder':  
		outputdir = args.resultfolder
		if not os.path.exists(outputdir):
			os.makedirs(outputdir);     
		imagedir  = args.imagefolder
		imagefolder_function(imagedir, outputdir)
		exit()
	elif sys.argv[1] == '--capture':
		camera_function()
		exit()
	elif sys.argv[1] == '--video':
		# video preprocess
		print ('Video File')
		window_name = "AMD YoloV2 Video File"
		capmode = args.capmode    
		cap = cv2.VideoCapture(videoFile)
		assert cap.isOpened(), 'Cannot capture source'    
		frames = 0
		start = time.time()
		while cap.isOpened(): 
			ret, frame = cap.read()
			if ret:
				#frame = cv2.flip(frame, 1)
				frame = cv2.resize(frame, (imageWidth, imageHeight))           
				results = detector.Detect(frame)
				imdraw = Visualize(frame, results)
				#cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
				#cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
				cv2.imshow(window_name, imdraw)
				key = cv2.waitKey(1)
				if key & 0xFF == ord('q'):
					break
				frames += 1
				if (frames % 16 == 0):
					print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
			else:
				break