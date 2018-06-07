#Create customised python layer

import caffe
import numpy as np
import math

# class heat_map_layer(caffe.Layer):


class heat_map_layer(caffe.Layer):
    """A layer that just multiplies by ten"""

    def setup(self, bottom, top):
        self.blobs.add_blob(1)
        self.blobs[0].reshape(*bottom[0].data.shape)
        self.blobs[0].data[...] = np.random.normal(loc = 0, scale = 0.1, size = (self.blobs[0].data.shape))
        top[0].reshape(bottom[0].data.shape[0],1,bottom[0].data.shape[2], bottom[0].data.shape[3] )
    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0],1,bottom[0].data.shape[2], bottom[0].data.shape[3] )

    def forward(self, bottom, top):
		batch_size = top[0].data.shape[0]
		nbr_of_channels = bottom[0].data.shape[1]
		input_sizeX = top[0].data.shape[2]
		input_sizeY = top[0].data.shape[3]
		#go through all images in the batch
		for i in range(0,batch_size):
			heat_map = np.zeros((input_sizeX, input_sizeY), dtype = np.float32)
			for j in range(0,input_sizeX):
				for k in range(0,input_sizeY):
					w = 0
					for l in range(0,nbr_of_channels):
						w = w + bottom[0].data[i,l,j,k]*self.blobs[0].data[i,l,j,k]
						heat_map[j,k] = w
			top[0].data[batch_size-1, 0, :, :] = heat_map


    def backward(self, top, propagate_down, bottom):
		for i in range(0,bottom[0].data.shape[0]):
			for j in range(0,bottom[0].data.shape[1]):
				for k in range(0,bottom[0].data.shape[2]):
					for l in range(0, bottom[0].data.shape[3]):
						bottom[0].diff[i,j,k,l] = top[0].diff[i,0,k,l]*self.blobs[0].data[i,j,k,l]
						#if math.isnan(bottom[0].diff[i,j,k,l]):
							#print 'Diff is NaN'
							#print 'top: ', top[0].diff[i,0,k,l], 'blob: ', self.blobs[0].data[i,j,k,l]

						#print bottom[0].diff[i,j,k,l]
