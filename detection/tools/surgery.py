'''
Helps you to checj output from network etc. 
'''

import caffe
import matplotlib.pyplot as plt
import sys
import numpy as np 
from bb_help import *
from mpl_toolkits.axes_grid1 import make_axes_locatable


caffe.set_mode_cpu() #Or set_mode_gpu() if GPU is availible
np.set_printoptions(threshold=np.nan) 

caffe_root = '../caffe/' #Change if necessary
model_def = '../models/train_val.prototxt'
#net_weights = 'weights.caffemodel'
net_weights = '../models/weights.caffemodel' #This setup is not trained for pedestrian detection, pleas enter your own caffemodel here!
image_path = '../data/I01047.jpg' 

print 'Setting up network...'
net = caffe.Net(model_def, net_weights, caffe.TEST)
#net = caffe.Net(model_def, caffe.TEST)
print 'Network initialisation done'

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


net.blobs['data'].data[...] = transformed_image

print 'Forwarding net...'
output = net.forward()
print 'Forward completed'

params = net.params['heat'][0].data[0,0,:,:]
blobs = net.blobs['heat'].data[0,0,:,:]

annotation_file = '../data/I01047.txt'
labels = np.zeros((2,1,29,39))
lcper, rcper, lcpeo, rcpeo = read_annotations(annotation_file)
label_matrix = np.zeros((29,39))
height =  check_pedestrian_height(annotation_file)
for j in range(0,len(lcper)):
	new_left, new_right = find_fitted_bounding_boxes(lcper[j], rcper[j])
	label_matrix = find_label_matrix(new_left, new_right, label_matrix)
labels[0,0,:,:] = label_matrix

net_loss = caffe.Net('loss.prototxt', caffe.TEST)
net_loss.blobs['prediction'].data[...] = net.blobs['heat'].data[...]
net_loss.blobs['label'].data[...] = labels

loss = net_loss.forward()

print 'loss', loss['loss']


count1 = 0

for i in range(0,29):
	for j in range(0,39):
		if label_matrix[i,j] == 1:
			count1 = count1+1 

ax = plt.subplot(111)
plt.axis('off')
im = ax.imshow(blobs)
divider = make_axes_locatable(ax)
plt.show()

ax = plt.subplot(111)
plt.axis('off')
im = ax.imshow(label_matrix)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()