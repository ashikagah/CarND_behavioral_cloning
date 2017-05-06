# Added recovery lap data - counter-steering angle 1.0

#import os
import csv
#import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D, ELU
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers.core import Lambda
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.optimizers import Adam

## Load udacity sample data
samples = []
with open('./data_udacity_recovery/driving_log_udacity.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples = samples[1:] # Remove header
num_samples = len(samples) 

## Add left camera data
samples_addleft = samples
for i in range(0, num_samples): 
    r=samples[i] 
    left_name = './data_udacity_recovery/IMG/'+r[1].split('/')[-1]
    left_angle = float(r[3])+0.25 # counter-steering angle
    r[0]=left_name
    r[3]=left_angle
    samples_addleft.append(r)       
samples = samples_addleft

## Add right camera data
samples_addright = samples
for i in range(0, num_samples): 
    r=samples[i] 
    right_name = './data_udacity_recovery/IMG/'+r[2].split('/')[-1]
    right_angle = float(r[3])-0.25 # counter-steering angle
    r[0]=right_name
    r[3]=right_angle
    samples_addright.append(r)       
samples = samples_addright

## Load recovery lap data - left lane
samples_addrecoveryleft = samples
with open('./data_udacity_recovery/driving_log_recovery_left.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        recovery_left_angle = float(r[3])+1. # counter-steering angle
        line[3]=recovery_left_angle
        samples_addrecoveryleft.append(line)       
samples = samples_addrecoveryleft
        
## Load recovery lap data - right lane
samples_addrecoveryright = samples
with open('./data_udacity_recovery/driving_log_recovery_right.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        recovery_right_angle = float(r[3])-1. # counter-steering angle
        line[3]=recovery_right_angle
        samples_addrecoveryright.append(line)       
samples = samples_addrecoveryright

N_orig = len(samples) 
angle_orig = []
for i in range(0, N_orig): 
    r = samples[i]
    angle_orig.append(float(r[3]))
print("Sample size (Original): ",N_orig)  
    
## Cull sample data with low steering angles
samples_cull = []
for i in range(0, N_orig): 
    r = samples[i] 
    if abs(float(r[3]))>.05: 
        samples_cull.append(r)
    elif np.random.randint(10) > 8: # Remove 80% of sample data with low steering angles
        samples_cull.append(r)
samples = samples_cull
N_cull = len(samples) 
angle_cull = []
for i in range(0, N_cull): 
    r = samples[i]
    angle_cull.append(float(r[3]))
print("Sample size (Culled): ",N_cull)  

# Frequency distribution of steering angles 
hist, bins = np.histogram(angle_orig, bins=25, range=(-1,1))
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show() # Original data
hist, bins = np.histogram(angle_cull, bins=25, range=(-1,1))
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show() # Culled data

## Split samples into training (80%) and test sets (20%)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
       
## Define generator
def generator(samples,batch_size): # Split the samples in batch_size batches an repeat the process
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size): # Repeat until all samples are loaded
            batch_samples = samples[offset:offset+batch_size] # Each batch_samples contains batch_size samples
            images = []
            angles = []
            for batch_sample in batch_samples: # Repeat unil all images in the batch are loaded
                # Load center camera data
                name = './data_udacity_recovery/IMG/'+batch_sample[0].split('/')[-1]
                image = mpimg.imread(name) # mpimg and opencv reads img very differently...
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

## Define image resizing function to fit Comma.ai model
def resize_image(image):
    import tensorflow as tf  # This import is required here otherwise the model cannot be loaded in drive.py
    return tf.image.resize_images(image, (40,160))
           
## Compile model using generator
train_generator = generator(train_samples,batch_size=32)
validation_generator = generator(validation_samples,batch_size=32)

## Comma.ai model 
model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)), # Crop 70 pixels from the top and 25 from the bottom
                     input_shape=(160,320,3),data_format="channels_last"))
model.add(Lambda(resize_image))# Resize image
model.add(Lambda(lambda x: (x/127.5) - 1.)) # Normalize signal intensity
model.add(Conv2D(16,(8,8),strides=(4,4),padding="same",activation="elu"))# Conv layer 1
model.add(Conv2D(32,(5,5),strides=(2,2),padding="same",activation="elu"))# Conv layer 2
model.add(Conv2D(64,(5,5),strides=(2,2),padding="same"))# Conv layer 3
model.add(Flatten())
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(512))# Fully connected layer 1
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(50))# Fully connected layer 2
model.add(ELU())
model.add(Dense(1))
model.compile(loss='mse', optimizer=Adam(lr=0.0001),metrics=['accuracy'])
print("Model summary:\n", model.summary())
history_object = model.fit_generator(train_generator,
                    steps_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples),
                    epochs=20,verbose=1)

## print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

## Save model
name = '20170401_model_test'
with open(name + '.json', 'w') as output:
    output.write(model.to_json())

model.save(name + '.h5')
print("Saved model to disk")

# python drive.py 20170401_model_test.h5 run01

# python video.py run01