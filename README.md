# Pedestrian classification and Detection
This code is for pedestrian classification and detection. For more detailed explanaitions see the report, MT_report.pdf.
 
 ## Classification
 The classification folder includes one classification demo and a script fpr training the network.

 To run the demo simply enter:

 python classification_demo .py path/to/image 

 in the command line. Example images are in the data folder. 

To run the training a training dataset in lmdb format must be added to the data folder. The dataset is not included in the git repository due to size limitations. Save the training and testing data with labels into folders: INRIA_train_lmdb and INRIA_test_lmdb respectively before running the training. Also ensure if you have an GPU availible or not. Change  solver_mode in solver.prototxt to GPU or CPU accordingly.

 ## Detection 
