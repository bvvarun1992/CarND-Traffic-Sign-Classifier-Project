# **Traffic Sign Recognition**

## **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/explore.JPG "Visualization"
[image2]: ./writeup_images/explore2.JPG  "Grayscaling"
[image3]: ./writeup_images/accuracy.JPG  "Grayscaling"
[image4]:./writeup_images/test.JPG  "Grayscaling"

### Data Set Summary & Exploration

The dataset was explored to get an idea of size of training, validation and test sets.

#### 1. Dataset summary

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset

Here are some examples of images found in the dataset.

![alt text][image1]

To get an understanding of the spread of the data over 43 classes of images, bar graphs were plotted for all three sets of data.

![alt text][image2]

---

### Design and Test a Model Architecture

#### 1. Preprocessing

To simplify the input to neural network, the images were grayscaled to exclude the impact of color pixels. The size of images were converted from 32x32x3 to 32x32x1.
Also, the image pixels were normalized by applying (pixel - 128)/128 in order to have mean zero.

#### 2. Model Architecture

The final model consists of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 RGB image   							|
| Convolution layer 1     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution layer 2    | 1x1 stride, valid padding, outputs 10x10x16  |
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 5x5x16   |
| Flattening					|	outputs 400						|
| Fully connected layer 1				| Outputs 120        						|
|	RELU and dropout		|	0.5 probability, Outputs 120						|
|	Fully connected layer 2			|	 Outputs 84											|
|	RELU and dropout		|	0.5 probability, Outputs 84						|
 |	Fully connected layer 3			|	 Outputs 41										|


#### 3. Training, validating and testing the Model

The LeNet architecture was used to model the neural network. A dropout was added to activation layers between the fully connected layers. The main purpose was to regularize the model and make it robust.
The model was trained to reduce cross entropy using the Adam optimizer. An initial learning rate of 0.001 and 50 EPOCHS were used and the validation set accuracy was just over 0.95. The learning rate and EPOCHS were tuned to obtain accuracy more than 95%. The final hyper parameters that were used are:
* Learning rate = 0.0008
* EPOCHS = 90
* Batch size = 128

Here is how accuracy pf validation set improves over number of EPOCHS.

![alt text][image3]

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of 0.972
* test set accuracy of 0.95

The Model still seems to be slightly overfit. Tuning EPOCHS might help reduce the overfit.

---

### Test a Model on New Images

#### 1. Images from web

Here are five German traffic signs that I found on the web:

![alt text][image4]

The images were rescaled to 32x32x3 to feed as input features to the neural network. The last image with Limit speed 70 Km/h consists shadows of the tree and was thought to be slightly tricky to predict.

#### 2. Model prediction

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 9, No passing      		| 9, No passing   									|
| 24, Road narrows on the right    			| 24, Road narrows on the right										|
| 33, Turn right ahead				| 33, Turn right ahead											|
| 13, Yield	      		| 13, Yield					 				|
| 4, Speed limit (70km/h)		| 4, Speed limit (70km/h)     							|


The model was able to correctly predict all 5 traffic signs, which gives an accuracy of 1. This compares favorably to the accuracy on the test set of 0.95

#### 3. Softmax

To understand how well the model has predicted each image, the top 5 softmaxes of each images are visualized.

  Top five probabilities for 9, No passing:
  * Probability = 1.000, Label = 9.000
  * Probability = 0.000, Label = 16.000
  * Probability = 0.000, Label = 23.000
  * Probability = 0.000, Label = 41.000
  * Probability = 0.000, Label = 10.000

Top five probabilities for 24, Road narrows on the right:
  * Probability = 1.000, Label = 24.000
  *  Probability = 0.018, Label = 27.000
  *  Probability = 0.000, Label = 28.000
  *  Probability = 0.000, Label = 20.000
  *  Probability = 0.000, Label = 26.000

Top five probabilities for 33, Turn right ahead:
  *  Probability = 1.000, Label = 33.000
  *  Probability = 0.000, Label = 14.000
  *  Probability = 0.000, Label = 13.000
  *  Probability = 0.000, Label = 35.000
  *  Probability = 0.000, Label = 1.000

Top five probabilities for 13, Yield:
  *  Probability = 1.000, Label = 13.000
  *  Probability = 0.000, Label = 25.000
  *  Probability = 0.000, Label = 15.000
  *  Probability = 0.000, Label = 35.000
  *  Probability = 0.000, Label = 39.000

Top five probabilities for 4, Speed limit (70km/h):
  *  Probability = 1.000, Label = 4.000
  *  Probability = 0.000, Label = 5.000
  *  Probability = 0.000, Label = 8.000
  *  Probability = 0.000, Label = 1.000
  *  Probability = 0.000, Label = 0.000
