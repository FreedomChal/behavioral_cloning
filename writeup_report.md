#**Behavioral Cloning** 

This is a project to clone the behavior of a human driver in a simulator.

Here is a link to my [project code](https://github.com/FreedomChal/behavioral_cloning/blob/master/model.py)

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### Model Architecture and Training Strategy

#### Model

My model is nearly identical to the model I used in my [Traffic Sign Classification](https://github.com/FreedomChal/traffic-sign-classification) project, only with three more convolutional layers. A more detailed description of the model can be found there. 

#### Model parameter tuning

The model has an adam optimizer, so the learning paramaters did not need to be manually tuned. For the batch size, I set it to 64, which seems to work decently for my model. I set the number of epochs to 50, but almost always stopped training before the end of all 50.

#### Appropriate training data

In the simulator, when recording the training data, I recorded two laps around the track, and some extra on the sharp turns. Later, I made a generator that takes in the recorded data, shuffles it, and outputs a formatted list of the normalized features (images), and the targets (steering angle). In the generator, half the images outputted are center images, one quarter left, and one quarter right. For the generator, and a couple other parts of the model, I got some ideas from, and used some helper code from [here](https://github.com/gardenermike/behavioral-cloning). One of the ideas is a tunable method of binning steering angles. The generator tunably returns data with higher steering angles more often. It generates a random number, and only uses the data if the steering angle to a power plus a leak value is greater than the random number.

#### Evolution of the code

Pretty much universally, the main issue my model had was not steering sharply enough, particularlly on tight turns such as right before the bridge and a little after the entrance to the dirt road. I tried a lot of things to deal with this, modifying the binning hyperparamaters, increasing the batch size, adding more layers and increasing the power of layers in the model, but what really made the difference was multiplying the steering angle by two in the generator. The doubled steering angle greatly increased the loss and caused the model to drive in somewhat wobbly manner, but overall it significantly improoved the driving of my model.
