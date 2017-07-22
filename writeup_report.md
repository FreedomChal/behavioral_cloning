# **Behavioral Cloning** 

This is a project to clone the behavior of a human driver in a simulator.

Here is a link to my [project code](https://github.com/FreedomChal/behavioral_cloning/blob/master/model.py)

---

[//]: # (Image References)

[image1]: ./imgenerall1.jpg "Strait area of track"
[image2]: ./imgeneral2.jpg "Turn near dirt track"
[image3]: ./imgeneral3.jpg "Bridge"
[image4]: ./imgsharpturn1.jpg "Sharp turn before bridge"
[image5]: ./imgsharpturn2.jpg "Sharp turn after dirt track entrance"

[video]: <iframe width="560" height="315" src="https://www.youtube.com/embed/QSH3F_UFe_g" frameborder="0" allowfullscreen></iframe>


### Model Architecture and Training Strategy

#### Model

My model is nearly identical to the model I used in my [Traffic Sign Classification](https://github.com/FreedomChal/traffic-sign-classification) project, only with three more convolutional layers. A more detailed description of the model can be found there. 

#### Model parameter tuning

The model has an adam optimizer, so the learning paramaters did not need to be manually tuned. For the batch size, I set it to 64, which seems to work decently for my model. I set the number of epochs to 50, but almost always stopped training before the end of all 50.

#### Appropriate training data

In the simulator, when recording the training data, I recorded two laps around the track, and some extra on the sharp turns.

![alt text][image1]

![alt text][image2]

![alt text][image3]

Later, I made a generator that takes in the recorded data, shuffles it, and outputs a formatted list of the normalized features (images), and the targets (steering angle). In the generator, half the images outputted are center images, one quarter left, and one quarter right. For the generator, and a couple other parts of the model, I got some ideas from, and used some helper code from [here](https://github.com/gardenermike/behavioral-cloning). One of the ideas is a tunable method of binning steering angles. The generator tunably returns data with higher steering angles more often. It generates a random number, and only uses the data if the steering angle to a power plus a leak value is greater than the random number.

#### Evolution of the code

Pretty much universally, the main issue my model had was not steering sharply enough, particularlly on tight turns such as right before the bridge and a little after the entrance to the dirt road.

![alt text][image4]

![alt text][image5]

I tried a lot of things to deal with this, modifying the binning hyperparamaters, increasing the batch size, adding more layers and increasing the power of layers in the model, but what really made the difference was multiplying the steering angle by two in the generator. The doubled steering angle greatly increased the loss and caused the model to drive in somewhat wobbly manner, but overall it significantly improoved the driving of my model.

[![video](https://img.youtube.com/vi/QSH3F_UFe_g/hqdefault.jpg)](https://youtu.be/QSH3F_UFe_g)

Note that the simulator locked up on the second lap so only one and a half laps were recorded.
