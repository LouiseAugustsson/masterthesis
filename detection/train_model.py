'''
Change Caffe path if necessary!

'''

import caffe
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

np.set_printoptions(threshold=np.nan)

caffe.set_mode_gpu() #Or set_mode_cpu() if GPU not availible

caffe_root = '../../caffe/'
solver_file = 'models/solver.prototxt'
weights_file = 'models/weights.caffemodel'

solver = caffe.get_solver(solver_file)
solver.net.copy_from(weights_file)
solver.solve()


# net = caffe.Net('models/train_val.prototxt', caffe.TRAIN)
# for k in range(0,100):
# 	print k
# 	net.forward()
	
# img = net.blobs['image'].data[...]
# label = net.blobs['labels'].data[...]
# print k 

# img_plot0 = img[0, 0, :, :]
# img_plot1 = img[0, 1, :, :]
# img_plot2 = img[0, 2, :, :]
# label_plot = label[0,0,:,:]

# f,(ax1, ax2) = plt.subplots(1,2)
# ax1.matshow(img_plot1)
# ax2.matshow(label_plot)
# plt.show()