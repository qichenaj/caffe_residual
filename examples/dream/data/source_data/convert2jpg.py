import dicom
import pylab
import os
import numpy as np
import cv2
import csv
import getpass
import random
import shutil  


pc_user_name = getpass.getuser()

source_img_path = "/home/" + pc_user_name  + "/caffe_residual/examples/dream/data/source_data/pilot_images/"
source_tsv_path = "/home/" + pc_user_name  + "/caffe_residual/examples/dream/data/source_data/"
save_training_img_path = "/home/" + pc_user_name  + "/caffe_residual/examples/dream/data/training_image/"
save_testing_img_path = "/home/" + pc_user_name  + "/caffe_residual/examples/dream/data/testing_image/"

source_img_names = os.listdir(source_img_path)
source_img_names = sorted(source_img_names)

if os.path.exists(save_training_img_path):
	shutil.rmtree(save_training_img_path)
	os.mkdir(save_training_img_path)

if os.path.exists(save_testing_img_path):
	shutil.rmtree(save_testing_img_path)
	os.mkdir(save_testing_img_path)

training_txt = open(save_training_img_path + "training.txt", "w")
testing_txt = open(save_testing_img_path + "testing.txt", "w")

testing_img_list = random.sample(source_img_names, 100)  #randomly select testing data 
training_img_list = list(set(source_img_names).difference(set(testing_img_list))) # training data

training_cancer_cnt = 0
testing_cancer_cnt = 0
with open(source_tsv_path + "images_crosswalk_pilot_20160906.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        if ".dcm" in line[5].lower():
        	if line[5] in training_img_list:
        		#print line[5], int(line[6])
	        	training_cancer_cnt = training_cancer_cnt + int(line[6])
	        	img_name = line[5].split('.', 1)[0] + '.jpg'
	        	img_name_rot90 = line[5].split('.', 1)[0] + '_rot90.jpg'
	        	img_name_rot180 = line[5].split('.', 1)[0] + '_rot180.jpg'
	        	img_name_rot270 = line[5].split('.', 1)[0] + '_rot270.jpg'
	        	training_txt.write(img_name + ' ' + line[6]+ '\n')
	        	training_txt.write(img_name_rot90 + ' ' + line[6]+ '\n')
	        	training_txt.write(img_name_rot180 + ' ' + line[6]+ '\n')
	        	training_txt.write(img_name_rot270 + ' ' + line[6]+ '\n')
	        if line[5] in testing_img_list:
        		#print line[5], int(line[6])
	        	testing_cancer_cnt = testing_cancer_cnt + int(line[6])
	        	img_name = line[5].split('.', 1)[0] + '.jpg'
	        	testing_txt.write(img_name + ' ' + line[6]+ '\n')
        		#training_txt.write(save_training_img_path + img_name + ' ' + line[6]+ '\n')

training_txt.close()
testing_txt.close()

print "training cancer cnt: ", training_cancer_cnt
print "testing cancer cnt: ", testing_cancer_cnt
print "total cancer cnt: ", testing_cancer_cnt + training_cancer_cnt

cnt = 0
for f in testing_img_list:
	cnt = cnt + 1
	print f, cnt
	img_prefix = f.split(".", 1)[0]
	ds=dicom.read_file(source_img_path + f)
	pix = ds.pixel_array
	#print pix.shape
	#print np.amax(pix), np.amin(pix)
	#pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)  
	#pylab.show()
	pix = cv2.resize(pix, (256,256), interpolation = cv2.INTER_AREA)
	img = (pix * 255.0) / np.amax(pix)
	img = np.uint8(img)
	#print np.amax(img), np.amin(img), img.dtype
	cv2.imwrite(save_testing_img_path + img_prefix + '.jpg', img)
	#cv2.imshow('Image', img)
	#cv2.waitKey()

cnt = 0
for f in training_img_list:
	cnt = cnt + 1
	print f, cnt
	img_prefix = f.split(".", 1)[0]
	ds=dicom.read_file(source_img_path + f)
	pix = ds.pixel_array

	pix = cv2.resize(pix, (256, 256), interpolation = cv2.INTER_AREA)
	img = (pix * 255.0) / np.amax(pix)
	img = np.uint8(img)
	img_rot90 = np.rot90(img, 1)
	img_rot180 = np.rot90(img, 2)
	img_rot270 = np.rot90(img, 3)
	cv2.imwrite(save_training_img_path + img_prefix + '.jpg', img)
	cv2.imwrite(save_training_img_path + img_prefix + '_rot90.jpg', img_rot90)
	cv2.imwrite(save_training_img_path + img_prefix + '_rot180.jpg', img_rot180)
	cv2.imwrite(save_training_img_path + img_prefix + '_rot270.jpg', img_rot270)
	#cv2.imshow('Image', img)
	#cv2.waitKey()

print "training cancer cnt: ", training_cancer_cnt
print "testing cancer cnt: ", testing_cancer_cnt
print "total cancer cnt: ", testing_cancer_cnt + training_cancer_cnt