
import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf
import keras.backend as K
from keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from keras.initializers import RandomNormal

#Function for reading data from our image directories
def getBatchDataset(imageDir, imageHeight, imageWidth, batch_size):
    return tf.keras.utils.image_dataset_from_directory(
        imageDir,
        labels=None,
        seed=123,
        image_size=(imageHeight, imageWidth),
        batch_size=batch_size)

#Here we define the structure of the generator model
def getInspiredGeneratorModel(imageWidth, imageHeight):
    
    #Input layer
    inputs = layers.Input((imageHeight, imageWidth, 3))
    
    #A few convolutional layers with maxpooling
    #This is the encoder part of the model that interprets the input image
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    
    #Their last convolution layer in their encoder seems to have skip layer with noise introduced to it
    #Noise is often used to avoid the overfiting of a machine learning model
    noise = layers.Input((K.int_shape(conv5)[1], K.int_shape(conv5)[2], K.int_shape(conv5)[3]))
    conv5 = layers.Concatenate()([conv5, noise])
    
    #From my understanding, upsampling is used to increase the weight of data in the minority class
    #Skip layer (connects conv4 layer directly to up6 layer), and more convolutional layers
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv5))
    merge6 = layers.Concatenate()([conv4, up6])
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = layers.Conv2D(256, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.Concatenate()([conv3, up7])
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = layers.Conv2D(128, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.Concatenate()([conv2, up8])
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = layers.Conv2D(64, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.Concatenate()([conv1, up9])
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv9)
    conv10 = layers.Conv2D(3, 1, activation='tanh')(conv9)
    
    #Create and return the final model
    model = Model(inputs=[inputs, noise], outputs=conv10)
    #model = Model(inputs=[inputs], outputs=conv10)
    return model

#Using the test feature image, we save generator output images to view the progress of it's training
def saveProgressImage(generator, testImage, iteration, dirName='./ProgressImages/Progress{}.png'):
    w_noise = np.random.normal(0, 1, (1, 32, 16, 1024))
    generatedImage = generator([firstImage[np.newaxis], w_noise], training=False)[0]
    generatedImage = generatedImage.numpy()
    generatedImage -= np.min(generatedImage)
    generatedImage /= np.max(generatedImage)
    generatedImage *= 256

    img = Image.fromarray(generatedImage.astype(np.uint8))
    img.save(dirName.format(iteration))

#Convert TF batchdataset into a numpy array
def batchDatasetToNumpy(dataset):
    dataNP = []
    for batch in dataset:
        for dat in batch:
            dataNP.append(dat.numpy())
    dataNP = np.array(dataNP)
    dataNP /= np.max(dataNP)
    return dataNP

#Randomly select a portion of feature data along with their corresponding target data
def generateRealSamples(features, targets, n_samples, nXPatches, nYPatches):
    idx = np.random.randint(0, features.shape[0], n_samples)
    feats = features[idx]
    targs = targets[idx]
    desiredDiscriminatorProbs = np.ones((n_samples, nXPatches, nYPatches, 1))
    return feats, desiredDiscriminatorProbs, targs


#Image parameters
batch_size = 4
imageWidth = 256
imageHeight = 512

#Choose image data directory depending whether we want to use all data, or a small subset for debugging.
useAllData = True
if useAllData:
    featuresDir = './ImageTrainingData/Features'
    targetsDir = './ImageTrainingData/Targets'
else:
    featuresDir = './SmallImageTrainingDataset/Features'
    targetsDir = './SmallImageTrainingDataset/Targets'


#Load data and extract a batch of each dataset
featureData = getBatchDataset(featuresDir, imageHeight, imageWidth, batch_size)
targetData = getBatchDataset(targetsDir, imageHeight, imageWidth, batch_size)
featureNP = batchDatasetToNumpy(featureData)
targetNP = batchDatasetToNumpy(targetData)


#U-Net generator model
generator = getInspiredGeneratorModel(imageWidth, imageHeight)
mseLoss = tf.keras.losses.MeanAbsoluteError()
generator.compile(optimizer='adam', loss=mseLoss, metrics=['mean_absolute_error'])

nEpochs = 5 #Number of times to repeat overall training loop
batchSize = 4 #n_batch in original code
batchesPerEpochs = int(len(featureData) / batchSize)
nSteps = nEpochs * batchesPerEpochs

#Save a generated image to track progress before training
firstImage = featureNP[0]
saveProgressImage(generator, firstImage, 0)

#Main training loop
for i in range(nSteps):
    
    #Chose a random selection of images for this batch, and generate required data for training
    feats, realLabels, targs = generateRealSamples(featureNP, targetNP, batchSize, 32, 16)
    #fakeOutput, fakeLabels = generateFakeSamples(generator, feats, nXPatches, nYPatches)
    
    w_noise = np.random.normal(0, 1, (batch_size, 32, 16, 1024))
    losses = generator.train_on_batch([feats, w_noise], y=targs)
    
    print('Losses: {}'.format(losses))
    
    #Save progress images of the generator output
    if i % 1 == 0:
        firstImage = featureNP[0]
        saveProgressImage(generator, firstImage, i)