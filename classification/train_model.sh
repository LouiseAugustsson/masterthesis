#Change caffe path if necessary! 
./../caffe/build/tools/caffe train -solver=models/SqueezeNet/SqueezeNet_v1.1/solver.prototxt -weights=models/SqueezeNet/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel 2>&1 | tee solve.log