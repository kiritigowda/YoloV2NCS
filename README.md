# YOLOv2 for Annie (AMD NN Inference Engine)

*This project shows how to run tiny yolov2 (20 classes) with AMD's NN inference engine(Annie):*
+ A python convertor from yolo to caffe
+ A c/c++ implementation and python wrapper for region layer of yolov2
+ A sample for running yolov2 with Annie

---

### Preliminaries
Please install amdovx modules and modelcompiler from https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-modules.git.

### Step 1. Compile Python Wrapper
```make```

### Step 2. Convert Caffe to Annie python lib as shown below using NNIR ModelCompiler (amdovx-modules/utils/model_compiler/)
### First convert caffe to NNIR format and compile NNIR to deployment python lib using the following steps

```
% python caffe2nnir.py ./models/caffemodels/yoloV2Tiny20.caffemodel <nnirOutputFolder> --input-dims 1,3,416,416
% python nnir2openvx.py [OPTIONS] <nnirInputFolder> <outputFolder> (details are in ModelCompiler page of amdovx_modules git repository)
```
There will be a file libannpython.so (under build) and weights.bin

### Step 3. Run tests
```	
python ./detectionExample/Main.py --image ./data/dog.jpg --annpythonlib <libannpython.so> --weights <weights.bin>
python ./detectionExample/Main.py --capture 0 --annpythonlib <libannpython.so> --weights <weights.bin> (live Capture)
python ./detectionExample/Main.py --video <video location> --annpythonlib <libannpython.so> --weights <weights.bin>
```
This runs inference and detections and results will be like this: 
![](/data/yolo_dog.jpg)

## Yolo Cascaded with Classification
### 1. Resize to 416X416 for Yolo and 1024X1024 for classification

Use the following link to resize the images. 

https://github.com/kiritigowda/help/tree/master/classificationLabelGenerator

### 2. Run step 1 and step 2 as above.

### 3. Run tests
```
python -W ignore detectionExample/Main.py --imagefolder <images1_416X416/> --cascade <images1_1024X1024/> --weights <weights.bin> --annpythonlib <libannpython.so>
```
(or)
```
python ./detectionExample/Main.py --capture 0 --annpythonlib <libannpython.so> --weights <weights.bin> (live Capture)
(and use key press for mode change)
```
# Run Other YoloV2 models
### Convert Yolo to Caffe 
```
Install caffe and config the python environment path.
sh ./models/convertyo.sh
```
Tips:

Please ignore the error message similar as "Region layer is not supported".

The converted caffe models should end with "prototxt" and "caffemodel".

### Update parameters

Please update parameters (biases, object names, etc) in ./src/CRegionLayer.cpp, and parameters (dim, blockwd, targetBlockwd, classe, etc) in ./detectionExample/ObjectWrapper.py.

Please read ./src/CRegionLayer.cpp and ./detectionExample/ObjectWrapper.py for details.

## Key Press Options when in capture mode:
Press these different keys to switch between modes (uses openCV)
1. **Keys '1' through 'n'** - Runs through a folder corresponding to number once and goes back to live mode (currently supports keys 1,2 folders)

2. **Key 'f'** - Runs through a folder until asked to change

3. **Key 'q'** - Quits from the program

4. **Key 'Space Bar'** - pauses the capture until space bar pressed again

5. **Key 'x'** - Runs cascaded classification on bounding boxes obtained in each frame.

## Key Press Options when in imagefolder mode:
Press these different keys to switch between modes (uses openCV)
1. **Key 'c'** - Switches to camera capture mode until asked to change
 
2. **Key 'q'** - Quits from the program
 
3. **Key 'Space Bar'** - pauses the capture until space bar pressed again

5. **Key 'x'** - Runs cascaded classification on bounding boxes obtained in each image.

# References
+ [caffe](https://github.com/BVLC/caffe)
+ [yolo](https://github.com/pjreddie/darknet)
+ [caffe-yolo](https://github.com/xingwangsfu/caffe-yolo)

# Contributors
+ [ichigoi7e](https://github.com/ichigoi7e)
+ [nathiyaa](https://github.com/nathiyaa)
+ [cpagravel](https://github.com/cpagravel)

---

# License
Research Only

# Author
duangenquan@gmail.com
rrawther@amd.com
