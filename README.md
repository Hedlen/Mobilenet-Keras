
**Goal**: This is an implementation of MobileNet using Keras. Thanks for the work and learning from rcmalli [keras-mobilenet](https://github.com/rcmalli/keras-mobilenet)
**Dataset**:[kaggle cat/dog dataset](https://www.kaggle.com/c/dogs-vs-cats)
# Requirements #
    1.Python 3.5
    2.Keras,Tensorflow
    3.OpenCV
# Introduction #
	1.data:
        Subfolders containing two types of samples, which are 0("cat") and 1("dog") subfolders, respectively.
	2.weights:
        Save the resulting model weights in this folder.
	3.logs:
        Save the event file, you can visualize the parameters of the training process and the model in the tensorbord
    4.model:
        Model folder, save network model with MobileNet
# Steps #
    1.splitdata.py
        Split out the training set and verification set
    2.train.py
        You should execute this program based on your actual parameter input

# License #
MIT LICENSE
# Reference #
[MobileNet](https://arxiv.org/pdf/1704.04861.pdf)