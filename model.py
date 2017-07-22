import pandas as pd
import tensorflow as tf
import numpy as np
from keras.layers import Input, Flatten, Dense, Activation, Reshape, Dropout, Lambda, Cropping2D
from keras.models import Model, Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import random
import cv2
import csv
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import PIL
from PIL import Image
from keras.layers import concatenate as Concat

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('csv_file', './driving_log.csv', "driving_log csv file (.p)")
flags.DEFINE_string('csv_file_valid', './validation/driving_log.csv', "driving_log csv file for validation (.p)")
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 64, "The batch size.")
batch_size = FLAGS.batch_size
epochs = FLAGS.epochs
features = []
targets = []

data_base_directory = './cardata/'

sampled_rows = []
with open(data_base_directory + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        center, left, right, steering_angle, throttle, _, speed = line
        _ = sampled_rows.append(line)

random.shuffle(sampled_rows)
valid_start_index = int(len(sampled_rows) * 0.9)
train_rows = sampled_rows[0:valid_start_index]
valid_rows = sampled_rows[valid_start_index:-1]
test_rows = valid_rows[0:batch_size]
valid_rows = valid_rows[batch_size:len(valid_rows)]

def load_data(csv_file):
    print("Training file", csv_file)

    with open(csv_file, 'rb') as f:
        csv = pd.read_csv(f).values.tolist()

    return csv

def get_image(data_path):
    image = cv2.imread(data_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.array(image)

def local_image_path(image_path):
    filename = image_path.split('\\')[-1]
    # I recorded the training data on Windows, and backslashes are the escape charactar in Python, which is why I'm using double backslashes    

    return data_base_directory + '/IMG/' + filename

far = 0
far2 = 1
# data order: center image, left image, right image, steering angle, throttle, brake (unused), speed
exp = 3.0
leak = 0.00025
def get_images(samples, batch_size=batch_size):
    _ = features.clear()
    _ = targets.clear()
    while True:
        _ = random.shuffle(samples)
        for line in samples:
            # brake is unused
            center, left, right, steering_angle, throttle, brake, speed = line
            steering_angle = float(steering_angle) * 2

            if np.random.random() > (np.power(np.abs(steering_angle), exp) + leak):
                continue
            did_work = False
            try:
                center_image = get_image(local_image_path(center))
                left_image = get_image(local_image_path(left))
                right_image = get_image(local_image_path(right))
                did_work = True

            except Exception as e:
              print("Error")
            else:
                try:
                    rdm = np.random.random()
                    if rdm < 0.5:
                        _ = features.append(np.array(center_image).astype(np.float32))
                        _ = targets.append(steering_angle)
                        _ = flipped_center = cv2.flip(center_image.astype(np.float32), 1)
                        _ = features.append(np.array(flipped_center).astype(np.float32))
                        _ = targets.append(-steering_angle)
                    elif rdm < 0.75:
                        _ = features.append(np.array(left_image).astype(np.float32))
                        _ = targets.append(steering_angle + 0.25)
                        _ = features.append(np.array(cv2.flip(left_image.astype(np.float32), 1)).astype(np.float32))
                        _ = targets.append(-steering_angle - 0.25)
                    else:
                        _ = features.append(np.array(right_image).astype(np.float32))
                        _ = targets.append(steering_angle - 0.25)
                        _ = features.append(np.array(cv2.flip(right_image.astype(np.float32), 1)).astype(np.float32))
                        _ = targets.append(-steering_angle + 0.25)
                except:
                    unused = None
            if len(features) >= batch_size:
                output = (np.array(features), np.array(targets))
                yield output
                _ = features.clear()
                _ = targets.clear()


    return None


dropout = 1.5
# load data
csv = load_data(FLAGS.csv_file)
batches_per_epoch = int(len(csv) / batch_size)
if (len(csv) % batch_size) > 0: batches_per_epoch += 1
 # define model
inp = Input(shape=(160,320,3))
crop = Cropping2D(cropping=((75, 55),(65, 65)))(inp)
normal = Lambda(lambda x: (x / 256.) - 0.5)(crop)

x11 = Conv2D(8, 7, activation="relu", padding="same")(normal)
xmp1 = MaxPooling2D(pool_size=(1, 2))(x11)
x12 = Conv2D(16, 5, activation="relu", padding="same")(xmp1)
x13 = Conv2D(32, 3, activation="relu", padding="same")(x12)
x1 = Dropout(0.15 * dropout)(x13)

x21 = Conv2D(8, 5, activation="relu", padding="same")(normal)
xmp2 = MaxPooling2D(pool_size=(1, 2))(x21)
x22 = Conv2D(16, 5, activation="relu", padding="same")(xmp2)
x23 = Conv2D(32, 5, activation="relu", padding="same")(x22)
x2 = Dropout(0.15 * dropout)(x23)

x31 = Conv2D(8, 5, activation="relu", padding="same")(normal)
xmp3 = MaxPooling2D(pool_size=(1, 2))(x31)
x32 = Conv2D(16, 3, activation="relu", padding="same")(xmp3)
x33 = Conv2D(32, 1, activation="relu", padding="same")(x32)
x3 = Dropout(0.15 * dropout)(x33)

inception = Concat([x1, x2, x3], axis=3)

x = Conv2D(32, 6, activation="relu", padding="same")(inception)
x = MaxPooling2D(pool_size=(1, 2))(x)
x = Dropout(0.3 * dropout)(x)

x = Conv2D(32, 5, activation="relu", padding="same")(x)
x = Dropout(0.3 * dropout)(x)

x = Conv2D(32, 4, activation="relu", padding="same")(x)
x = MaxPooling2D(pool_size=(1, 2))(x)
x = Dropout(0.3 * dropout)(x)

x = Conv2D(32, 3, activation="relu", padding="same")(x)
x = Dropout(0.3 * dropout)(x)

x = Conv2D(32, 2, activation="relu", padding="same")(x)
x = Dropout(0.3 * dropout)(x)


x = Conv2D(32, 1, activation="relu", padding="same")(x)

x = Flatten()(x)

x = Dense(64, activation="relu")(x)
x = Dropout(0.3 * dropout)(x)

x = Dense(32, activation="relu")(x)
x = Dropout(0.3 * dropout)(x)

x = Dense(16, activation="relu")(x)
x = Dropout(0.3 * dropout)(x)

x = Dense(1)(x)

model = Model(inp, x)

# TODO: train your model here
optimizer = Adam(lr=0.0001, decay=0.01)
_ = model.compile(optimizer, 'mean_squared_error', ['accuracy'])

checkpointer = ModelCheckpoint(filepath='model.h5', verbose=1)
_ = model.fit_generator(get_images(train_rows, batch_size), steps_per_epoch=batches_per_epoch, epochs=epochs,
  validation_data=get_images(valid_rows, batch_size), validation_steps=5,
  callbacks=[checkpointer])

test_x = []
test_y = []

test_size = 10
for i in range(test_size):
   batch_x, batch_y = next(get_images(test_rows, batch_size))

   _ = test_x.extend(batch_x)
   _ = test_y.extend(batch_y)

test_loss = model.evaluate(np.array(test_x), np.array(test_y))[0]
print("Test loss: {}".format(test_loss))

_ = model.save('model.h5')

