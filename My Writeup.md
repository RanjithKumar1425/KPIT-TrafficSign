# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeupimages/TrainingSetViz.png "Training Set visualization"
[image2]: ./writeupimages/ValidationsetViz.png "Validation Set visualization"
[image3]: ./writeupimages/TestSetviz.png "Test Set visualization"
[image4]: ./testimage/9_no_passing.jpg "no passing Sign 1"
[image5]: ./testimage/11_right-of-way.jpg "right of way Sign 2"
[image6]: ./testimage/12_priority_road.jpg "prioroty road Sign 3"
[image7]: ./testimage/15_no_vehicles.jpg "no vehicles Sign 4"
[image8]: ./testimage/18_general_caution.jpg "general caution Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of Validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

I used Pythons enumerate to Count the No of Occurence of each class in the Training, Validation and Test set. Used bar chart to visualize the data.

Below is the bar chart for each of the data set.

![alt text][image1]
![alt text][image1]
![alt text][image1]

### Design and Test a Model Architecture

#### 1. Pre Processing

 I used only used Normalization for image preprocessing.
 I used (pixel - 128)/ 128 to normalize the image, so that the data has mean zero and equal variance.
 
 Need to try more Pre Procesing method. For the Assigment i felt Normalization should work well.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Dropout				| Keep Prod = 0.9								|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | 1x1 stride, same padding, output 10x10x16		|
| RELU					|												|
| Dropout				| Keep Prod = 0.9								|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Fully connected		| Input = 400. Output = 120 					|
| RELU					|												|
| Dropout				| Keep Prod = 0.9								|
| Fully connected		| Input = 120. Output = 84  					|
| RELU					|												|
| Dropout				| Keep Prod = 0.9								|
| Fully connected		| Input = 84. Output = 43   					|


 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used 25 epochs for training the model. Initially I used 10, but I found that more epochs were needed in order to achieve a higher accuracy. I increased the epochs to 15 and then finally settled with 25 epochs

I used the same optimizer, tf.train.AdamOptimizer, as in the LeNet lab.

I used rate = 0.001 for the learning rate; I tried using rate = 0.0001 but the results were not as good.

For dropout, I used keep_prob = 0.9 when training, which I found delivered better results than using keep_prob = 0.5 or not using dropout at all.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy = 0.999
* Validation set accuracy = 0.943
* Test set accuracy = 0.938

I first stared with the LeNet Model designed by changing the initial weightes to shape=(5, 5, 3, 6) and changing the Final Fully Connected output to 43. This Model gave a Validation Set Accuracy of 89%.

Then i tried t0 Include the Dropout after the RELU of First and Second COnvolution Layers. 
This Model gave a Validation set Accuracy of 92%

Then i tried to include the Dropout after both the Fully Connected Layers
This Model gave a Validation set Accuracy of 94.3%

I am satisified with the Validation Set Accuracy and try the model on Test Set.
My Model gave a Test Set Accuracy of 93.8%

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

Prediction   Actual Value
----------   ------------
     9             9     
    11            11     
    12            12     
    15            15     
    18            18  


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Below is the Top 5 Softmax probalities for each of my test images.

9.90531e-01   9.46928e-03   1.33668e-08   1.38727e-11   7.82196e-12
1.00000e+00   9.11099e-08   1.08586e-10   5.03623e-13   4.97527e-16
1.00000e+00   6.57461e-12   6.84074e-14   1.43310e-14   1.39076e-14
1.00000e+00   3.80930e-11   4.51352e-13   2.43889e-13   1.23781e-14
9.99999e-01   7.94764e-07   7.27055e-08   8.63628e-12   5.18951e-12

Index of the above 5 softmax probalities.

9 16 41 40  3
11 30 21 27  5
12 10  9  7 42
15 26 13  8  4
18 27 26 25 24

My Model was able to identify each image with a accuracy of 99% in most of the case.



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


