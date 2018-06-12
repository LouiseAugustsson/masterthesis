'''
Demonstration of pedestrian detection. Run by entering: 

python classify.py path/to/image 

in terminal. There exists example images in ../data
'''


#Import
import numpy as np
import matplotlib.pyplot as plt
import caffe
import os
import sys

#Initialize caffe
caffe.set_mode_cpu()
caffe_root = '../../../caffe/'
TRANSFORM = True

#Load weights, model definition and image
model_def = 'models/deploy.prototxt'
model_weights = 'models/weights.caffemodel'
image_path = sys.argv[1]

net = caffe.Net(model_def, model_weights, caffe.TEST)

#Transform image (substract image mean and RGB --> BGR)
if TRANSFORM:
	# load the mean ImageNet image (as distributed with Caffe) for subtraction
	mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
	mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
	# create transformer for the input called 'data'
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
	transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
	transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
	transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
	image = caffe.io.load_image(image_path) #Load image
	transformed_image = transformer.preprocess('data', image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

#Forward network
output = net.forward()


def print_results():
	print ' '
	print '------------------------------------------------------------------------'
	#print '                          Output starts from here                       '
	#print '                          -----------------------                      ' 
	#print '------------------------------------------------------------------------'
	print ' '
	 
	# print ' '
	# print '------------------------------------------------------------------------'
	# print ' '

	if net.blobs['prob'].data[0][0][0][0] > 0.5: 
	  print 'Predicted class: NO PEDESTRIAN'
	else:  
	  print 'Predicted class: PEDESTRIAN'


	print ''

	print 'Class		Probability'
	print '--------------------------------'
	print 'PEDESTRIAN	', net.blobs['prob'].data[0][1][0][0]
	print 'NO PEDESTRIAN	', net.blobs['prob'].data[0][0][0][0]
	print ''

	print '------------------------------------------------------------------------'
	print ' '

print_results()