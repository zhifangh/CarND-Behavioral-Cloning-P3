# *Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior

* Build, a convolution neural network in Keras that predicts steering angles from images

* Train and validate the model with a training and validation set

* Test that the model successfully drives around track one without leaving the road

* Summarize the results with a written report

  

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 12 and 48 (model.py lines 131-139) 

The model includes RELU layers to introduce nonlinearity (code line 131、135、139、143、147、151), and the data is normalized in the model using a Keras lambda layer (code line 129). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 133、137、145、149). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 32 、101-102、158-162). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 157).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was inspired by NVIDA model, which was derived from End to End Learning for Self-Driving Cars.

I crop the images to remove the noise from image (model.py line 125) and resize the image data to accelerate the traning step (model.py line 127)，then I normalize the data (model.py line 129).

I desing three convolution layer. Those filter size is 5 * 5, and those depths between 12 and 48.  Each convolution layer is followed by a dropout layer for preventing overfitting.

I follow the three convolutional layers with three fully connected layers, leading to a final output control value which is steering angle of the vehicle.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

I try a few group parameters for BS and EPOCHS, Finally, I set batch size to 50 and EPOCHS to 1000.

#### 2. Final Model Architecture

Nvidia's driverless car team has come up with a network for end-to-end driverless driving, which is the same network they use for actual vehicle training, and that's something we can learn from. The final model architecture (model.py lines 123-153) is similar to that. The structure of network is as follows:  

![](D:\hzf\udacity\project\CarND-Behavioral-Cloning-P3\image\cnn_architecture.png)



#### 3. Creation of the Training Set & Training Process

I try to collect the training data using center lane driving and put them in the directory "./data/IMG". I can't determine the images in this directory are genrated by myself. Because that the date contained in the name of the images is a long time ago.  However, I can use the data set as training data to train network and run the simulator in autopilot mode.


I split the data set into trainning data and validation data and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I set the size of batch to 50 and the number of epochs to 1000. I used an adam optimizer so that manually training the learning rate wasn't necessary.
