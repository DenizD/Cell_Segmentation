
Deep Learning Based Cell Segmentation
===

This is an example cell segmentation procedure in histopathological images based on convolutional neural networks.

Dataset
-------------
Unlike traditional deep learning based approaches which use 2-classes ("background" and "cell"), we use 4-classes ("background", "cell center", "cell innerboundary" and "cell outerboundary") as inputs to train the model. In this way, we can delineate nucleus boundaries more accurately. Following figure demonstrates a training patch extraction procedure for 4-class model.

<p align="center">
  <img src="./patchExtraction.png" width="50%" height="50%"/>
</p>


Data Augmentation
-------------
In machine learning, the more data samples we have, the better our model will be. However, we do not have so many data samples in practice since collecting and labeling data are time consuming and costly processes. Therefore, we need data augmentation which is a technique for increasing the number of samples in the dataset in artificial ways. In this project, we augment the number of training images using "mirroring", "random rotation", "filtering" and "color casting". Sample augmented images are shown in the following figure. 

<p align="center">
  <img src="./dataAugmentation.png" width="50%" height="50%"/>
</p>


Results
-------------
In the following figure, sample segmentation results are shown for 2-class and 4-class deep learning model. In this figure, each class label is shown with a different color ("background": white, "cell center": blue, "cell innerboundary": red, and "cell outerboundary": green). Here, notice that 4-class model performs better segmentation especially for overlapping cells. 
<p align="center">
  <img src="./sampleSegmentation.png" width="80%" height="80%"/>
</p>


Prerequisites
-------------
[Caffe](http://caffe.berkeleyvision.org/) deep learning framework and Python are used for the implementation of the project.

Model is trained and tested on Ubuntu 16.04 machine.


How to Run
-------------
1) To generate training and validation image patches, run [extractImagePatches.py](extractImagePatches.py)
2) To create training and validation labels run [createLabels.py](createLabels.py)
3) To train the model, run [train.py](train.py). You can configure "batch size" and "augmentation" parameters in "train_val.prototxt" file
4) To test the model on a sample test image, run [test.py](test.py). This will save "estimatedLabels.jpg" to a file

Note that: You can configure the parameters in [config.json](config.json) file

Contact
-------
E-mail: deniz.mail@gmail.com


References
------------
- http://caffe.berkeleyvision.org/
