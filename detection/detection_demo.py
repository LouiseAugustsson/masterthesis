import caffe
import matplotlib.pyplot as plt
import sys
import numpy as np 
from bb_help import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

caffe.set_mode_cpu()
np.set_printoptions(threshold=np.nan)

caffe_root = '../../caffe/'
model_def = 'models/train_val.prototxt'
#net_weights = 'weights.caffemodel'
net_weights = 'models/train_iter_42300.caffemodel'
image_path = sys.argv[1]

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

ax = plt.subplot(111)
plt.axis('off')
im = ax.imshow(blobs)
plt.show()
