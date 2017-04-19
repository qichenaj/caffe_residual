#coding=utf-8
import os
import numpy as np
import SimpleITK
import matplotlib.pyplot as plt
import h5py, sys
import lmdb
from PIL import Image
import cv2
import getpass
import caffe

g_pc_user_name = getpass.getuser()

def show_data(data, title = 'image', padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.figure()
    plt.title(title)
    plt.imshow(data,cmap='gray')
    plt.axis('off')
    plt.show()

def main(): 
    caffe_root = '/home/' + g_pc_user_name +'/caffe_residual/' 
    sys.path.insert(0, caffe_root + 'python')
    os.chdir(caffe_root)
    caffe.set_device(0)
    caffe.set_mode_gpu()

    file_dir = 'examples/dream/'
    img_dir = caffe_root + file_dir + 'data/testing_image/'
    net_file= caffe_root + file_dir + 'net/ResNet-50-deploy-softmax.prototxt'
    #caffe_model=caffe_root + file_dir + 'vgg16ft_single_modal_T2_iter_2500.caffemodel'
    caffe_model=caffe_root + file_dir + 'snapshot/ResNet_iter_6000.caffemodel'
    net = caffe.Net(net_file, caffe_model, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0)) # if using RGB instead if BGR

    testing_img_names = os.listdir(img_dir)
    testing_img_names = sorted(testing_img_names)
    #print testing_img_names

    testing_txt = open(img_dir + "testing.txt", "r")
    label = testing_txt.readlines()
    # print label
    dict = {}
    for line in label:
        line.strip()
        t = line.split(' ', 1)
        dict[t[0]] = int(t[1])
        #print dict[t[0]]

    cnt = 0
    for f in testing_img_names:
        if '.jpg' not in f:
            continue
        img = cv2.imread(img_dir + f)
        #print img.shape
        # img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
        # img[:,:,0] = img[:,:,0] - 135;
        # img[:,:,1] = img[:,:,1] - 149;
        # img[:,:,2] = img[:,:,2] - 182;

        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        out = net.forward()
        #print 'test data shape:', net.blobs['data'].data[...].shape
        #show_data(net.blobs['data'].data[0])

        caffe_result = net.blobs['prob'].data[0]
        prob = caffe_result
        cancer = 0
        
        if (prob[0] > prob[1]):
            cancer = 0
        else:
            cancer = 1
        if (dict[f] == cancer):
            cnt = cnt + 1
            print "correct", str(dict[f]) + '<label------predict>'+ str(cancer), prob
        else:
            print '@@@@@@@@@@wrong:', str(dict[f]) + '<label^^^^^predict> '+ str(cancer), prob
            # cv2.imshow('origin', img)
            # cv2.waitKey(5)
        #print 'softmax prob shape: ', prob.shape
    print float(cnt) / len(testing_img_names)
if __name__ == '__main__':
    main()
