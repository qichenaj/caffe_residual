import dicom
import pylab
import os
import numpy as np
import cv2
import csv


path = "/home/qichen/caffe_residual/examples/dream/data/source_data/pilot_images/"
save_path = "/home/qichen/caffe_residual/examples/dream/data/training_image/"
files = []  # create an empty list
for dirName, subdirList, fileList in os.walk(path):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            files.append(os.path.join(dirName,filename))


cnt = 0
for f in os.listdir(path):
#for f in files:
	cnt = cnt + 1
	print f, cnt
	img_prefix = f.split(".", 1)[0]
	ds=dicom.read_file(path + f)
	#print ds.dir("pat")
	#pixel_bytes = ds.PixelData
	pix = ds.pixel_array
	#print pix.shape
	#print np.amax(pix), np.amin(pix)
	#pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)  
	#pylab.show()
	pix = cv2.resize(pix, (256,256), interpolation=cv2.INTER_AREA)
	img = (pix * 255.0) / np.amax(pix)
	img = np.uint8(img)
	
	#print np.amax(img), np.amin(img), img.dtype
	cv2.imwrite(save_path + img_prefix + '.jpg', img)
	#cv2.imshow('Image', img)
	#cv2.waitKey()
