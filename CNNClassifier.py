#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:50:50 2019
@author: Andres Echeverri 
"""
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import os

plt.style.use('ggplot' )
img_width     = 80
img_height    = 80
img_channels  = 3

"""
img_read: it reads the file's path and returns a label list of each image
Args: 
    dir_img: str 
        directory where the images are located
        
Returns: list[img_path, label]
     it returns a list that has the image path of the images and the label 
     of each of the images-
"""
def img_read ( dir_img ):
    label_list = []
    imgs_path =[]    
    for imgs_names in os.listdir( dir_img ):
        imgs_path.append( os.path.join( dir_img, imgs_names ) )
        if 'car' in imgs_names:
            label_list.append( 1 )     
        else:
            label_list.append( 0 )
    return  [imgs_path, np.array( label_list ).astype( np.float32 )]    

"""
img_scaling: it scales the images to the desired size
Args: 
    imgs_path: str 
        directory where the images are located
        
Returns: np.array
     it returns an array containing all the images 
"""
def img_scaling ( imgs_path ):    
    img_list     = []  
    for images in imgs_path:
       img = cv2.imread( images )
       width, height, channels = img.shape
       if width!= img_width or height!= img_height :
           img = cv2.resize( img, dsize=( img_width, img_height ), 
                            interpolation=cv2.INTER_CUBIC )

       norm_image = cv2.normalize(img, 
                                   None, 
                                   alpha=0, 
                                   beta=1, 
                                   norm_type=cv2.NORM_MINMAX, 
                                   dtype=cv2.CV_32F)   
       img_list.append( norm_image )    
    imgs_array = np.array( img_list )    
    return imgs_array

"""
plot_metrics: it plots all the metrics 
Args: 
    acc: list
        trainning accuracy 
    val_acc: list
        validation accuracy 
    loss: list
        trainning loss
    val_loss: list
        validation loss
     
Returns: None     
"""
def plot_metrics(acc, val_acc, loss, val_loss):
    plt.figure(1, figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

"""
plot_kernels: it plots the kernels used in the CNN 
Args: 
    model: tensorflow.python.keras.engine.training.Model
         trainned model used 
     
Returns: None     
"""
def plot_kernels(model):
    plt.figure(2)
    for layer in model.layers:
    	if 'conv' not in layer.name:
    		continue
    	filters, biases = layer.get_weights()
    	print(layer.name, filters.shape)
    f_min = filters.min()
    f_max = filters.max()
    filters = (filters - f_min)/(f_max-f_min)
    for i in range(filters.shape[3]):
        f = filters[:, :, :, i]
        plt.subplot(filters.shape[3]/4,filters.shape[3]/8,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(f)
    plt.show()
    
"""
plot_convolution: it plots all the metrics 
Args: 
    acc: list
        trainning accuracy 
    val_acc: list
        validation accuracy 
    loss: list
        trainning loss
    val_loss: list
        validation loss
     
Returns: None     
"""
def plot_convolution(model, img):
    img = np.expand_dims(img, axis=0)
    plt.figure(3)
    print(img.shape)
    new_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[1].output)
    feature_maps = new_model.predict(img)
    for i in range(feature_maps.shape[3]):
        f = feature_maps[0, :, :, i]
        plt.subplot(feature_maps.shape[3]/4,feature_maps.shape[3]/8,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(f)
    plt.show()

"""
plot_predictions: it displays the predicted vehicles with its labels
Args: 
    imgs_path_testing: list
        it contains all the testing images
    y_prediction: list
        it contains all the "number" labels of the testing images
        
Returns: None     
"""      
def plot_predictions(imgs_path_testing, y_prediction): 
    plt.figure(4, figsize=(10,10))
    number_samples = 25
    for i in range(number_samples):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img_path = imgs_path_testing[i]
        img = mpimg.imread(img_path)
        plt.imshow(img)
        label = y_prediction[i]
        if label == 1:
            img_label = "car"
        else:
            img_label = "motorcycle"
        plt.xlabel(img_label)
    plt.show()
    
#Read from the folder 
cwd = os.getcwd( )
dir_img = cwd + '/img_dataset'
dataset_list  = img_read( dir_img ) 
#Split the images in trainning and testing datasets
imgs_path_trainning, imgs_path_testing, label_array_trainning, label_array_testing = \
train_test_split(dataset_list[0], dataset_list[1], test_size=0.3, shuffle=True)
#Resize the images 
dataset_array_trainning = img_scaling( imgs_path_trainning )
dataset_array_testing = img_scaling( imgs_path_testing )
#Model definition 
input_layer = tf.keras.Input(shape=(img_width, img_height, img_channels), name="input_layer")
conv1 = tf.keras.layers.Conv2D(filters=32,
      						   kernel_size= 2,
      						   strides = 2,
      						   padding="same",
      						   data_format="channels_last",
      						   activation=tf.nn.relu)(input_layer)
pool1 = tf.keras.layers.MaxPool2D(pool_size =(2,2),
								  padding = "same",
								  data_format="channels_last")(conv1)
flatten = keras.layers.Flatten()(pool1)
dense1 = keras.layers.Dense(1200, activation='relu')(flatten)
output = keras.layers.Dense(1, activation = tf.nn.relu,name="output")(dense1)
model = tf.keras.Model(inputs=input_layer, outputs=output)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.000002), 
                                              loss="mean_squared_error", 
                                              metrics=["acc"])

metrics = model.fit(dataset_array_trainning, label_array_trainning,
                    batch_size=10,
                    epochs=15, 
                    validation_data=(dataset_array_testing, label_array_testing))

y_prediction = model.predict(dataset_array_testing)
y_prediction = np.where(y_prediction>=0.5,1,0 )
y_prediction = np.reshape(y_prediction, [360,])
y_prediction =list(y_prediction)

acc = metrics.history['acc']
val_acc = metrics.history['val_acc']
loss = metrics.history['loss']
val_loss = metrics.history['val_loss']

#Generate the plots 
plot_metrics(acc, val_acc, loss, val_loss)
plot_kernels(model)
plot_convolution(model, dataset_array_trainning[10])
plot_predictions(imgs_path_testing, y_prediction)





