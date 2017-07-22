# **Behavioral Cloning** 

This is a project to clone the behavior of a human driver in a simulator.

Here is my [project code](https://github.com/FreedomChal/behavioral_cloning/blob/master/model.py).

---

[//]: # (Image References)

[image1]: ./imgenerall1.jpg "Strait area of track"
[image2]: ./imgeneral2.jpg "Turn near dirt track"
[image3]: ./imgeneral3.jpg "Bridge"
[image4]: ./imgsharpturn1.jpg "Sharp turn before bridge"
[image5]: ./imgsharpturn2.jpg "Sharp turn after dirt track entrance"

### Model Architecture and Training Strategy

#### Model

My model is nearly identical to the model I used in my [Traffic Sign Classification](https://github.com/FreedomChal/traffic-sign-classification) project, only with three more convolutional layers. A more detailed description of the model can be found there. 

#### Model parameter tuning

The model has an adam optimizer, so the learning parameters did not need to be manually tuned. For the batch size, I set it to 64, which seems to work decently for my model. I set the number of epochs to 50. However, I had a checkpointer that saves the model at the end of every epoch. With the checkpointer, I was able to stop the training before it finished all 50 epochs. I often did cancel training early because most of my model's learning tended to stop around epoch 7.

#### Appropriate training data

In the simulator, I recorded two laps around the track, and some extra on the sharp turns to remove my model's bias to go straight.

![alt text][image1]

![alt text][image2]

![alt text][image3]

Later, I made a generator that takes in the recorded data, shuffles it, and outputs batches of normalized features (images), and targets (steering angle). In the generator, half the images output are center images, one quarter left camera images with an adjusted steering angle, and one quarter right camera images with an adjusted steering angle. For the generator, I got some ideas from, and used some helper code from [here](https://github.com/gardenermike/behavioral-cloning). One of the ideas is a tunable method of binning steering angles. The generator tunably returns data with higher steering angles more often. It generates a random number, and only uses the data if the steering angle to a power plus a leak value is greater than the random number.

#### Evolution of the code

Pretty much universally, the main issue my model had was not steering sharply enough, particularly on tight turns, such as the corner right before the bridge, and the corner after the entrance to the dirt road.

![alt text][image4]

![alt text][image5]

I tried a lot of things to deal with the small steering angles output by my model: modifying the binning hyperparameters, increasing the batch size, adding more layers, increasing the size of layers in the model. What really made the difference, though, was multiplying the steering angle by two in the generator. The doubled steering angle caused some odd effects, greatly increasing the loss and causing the model to drive in a somewhat wobbly manner, but overall it significantly improved the driving in the simulator, enough to consistently make it around the track safely.

Here is the link to the video of my model driving in the simulator:

[![video](https://img.youtube.com/vi/QSH3F_UFe_g/hqdefault.jpg)](https://youtu.be/QSH3F_UFe_g)

Note that the simulator locked up on the second lap so only one and a half laps were recorded.
