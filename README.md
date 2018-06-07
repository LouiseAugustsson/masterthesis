# Pedestrian classification and Detection
This code is for pedestrian classification and detection. For more detailed explanaitions see the report, MT_report.pdf.
 
 ## Classification
 The classification folder includes a classification demo and a script for training the network.

 To run the demo simply enter:

 python classification_demo .py path/to/image 

 in the command line. There are xample images in the data folder. 

To run the training a training dataset in lmdb format must be added to the data folder. The dataset is not included in the git repository due to size limitations. Save the training and testing data with labels into folders: INRIA_train_lmdb and INRIA_test_lmdb respectively before running the training. Also ensure if you have an GPU availible or not. Change  solver_mode in solver.prototxt to GPU or CPU accordingly.

 ## Detection 

Before running the detection-training, all data in lmdb format must be added into the data folder. Add the images into folders train_images_lmdb and test_images_lmdb respectively and the labels into train_lanels_lmdb and test_labels_lmdb. Be carefull so that the images and labels come in the same order. bb_help in tools folder includes help functions for creating the labels from images in the Caltech pedestrian dataset. 