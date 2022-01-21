
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
    return model

def getInspiredDiscriminatorModel(imageHeight, imageWidth):
    
    #Create inputs of initial and generated image to discrimate, and combine them
    init = RandomNormal(stddev=0.02)
    initialImage = layers.Input(shape=(imageHeight, imageWidth, 3))
    generatedImage = layers.Input((imageHeight, imageWidth, 3))
    combinedImages = layers.Concatenate()([initialImage, generatedImage])
    
    #Main structure of the discriminator neural network model
    d = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(combinedImages)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    x = layers.LeakyReLU(alpha=0.2)(d)
    output = layers.Conv2D(1, (4, 4), padding='same', activation='sigmoid', kernel_initializer=init)(d)
    
    #Create the model and return it
    model = Model([initialImage, generatedImage], output)
    return model

def mount_discriminator_generator(generator, discriminator, image_shape):
    discriminator.trainable = False
    input_gen = layers.Input(shape=image_shape)
    input_noise = layers.Input(shape=(32, 16, 1024))
    gen_out = generator([input_gen, input_noise])
    output_d = discriminator([gen_out, input_gen])
    model = Model(inputs=[input_gen, input_noise], outputs=[output_d, gen_out])
    return model

#Convert TF batchdataset into a numpy array
def batchDatasetToNumpy(dataset):
    dataNP = []
    for batch in dataset:
        for dat in batch:
            dataNP.append(dat.numpy())
    return np.array(dataNP)

#Randomly select a portion of feature data along with their corresponding target data
def generateRealSamples(features, targets, n_samples, nXPatches, nYPatches):
    idx = np.random.randint(0, features.shape[0], n_samples)
    feats = features[idx]
    targs = targets[idx]
    desiredDiscriminatorProbs = np.ones((n_samples, nXPatches, nYPatches, 1))
    return feats, desiredDiscriminatorProbs, targs

#Generate fake images corresponding to the real targets
def generateFakeSamples(generator, dataset, nXPatches, nYPatches):
    w_noise = np.random.normal(0, 1, (dataset.shape[0], 32, 16, 1024))
    fakeOutput = generator.predict([dataset, w_noise])
    desiredDescriminatorProbs = np.zeros((len(fakeOutput), nXPatches, nYPatches, 1))
    return fakeOutput, desiredDescriminatorProbs


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
featureBatch = next(iter(featureData))
targetBatch = next(iter(targetData))

#Adam optimizer is a popular algorithm for training neural network models with
adamOptimizer = tf.keras.optimizers.Adam(1e-4)

#Get and setup generator, discriminator and composite models
generator = getInspiredGeneratorModel(imageWidth, imageHeight)
discriminator = getInspiredDiscriminatorModel(imageHeight, imageWidth)
discriminator.compile(loss='binary_crossentropy', optimizer=adamOptimizer)
compositeModel = mount_discriminator_generator(generator, discriminator, (imageHeight, imageWidth, 3))
compositeModel.compile(loss=['binary_crossentropy', 'mae'], loss_weights=[1, 3], optimizer=adamOptimizer)


nEpochs = 5 #Number of times to repeat overall training loop
batchSize = 4 #n_batch in original code
batchesPerEpochs = int(len(featureData) / batchSize)
nSteps = nEpochs * batchesPerEpochs

avg_loss = 0
avg_dloss = 0

#Get number of patches in X and Y directions generated by our discriminator
batch = next(iter(featureData))
w_noise = np.random.normal(0, 1, (batch_size, 32, 16, 1024))
generatedImageBatch = generator([batch, w_noise], training=False)
decision = discriminator((batch, generatedImageBatch))
nXPatches, nYPatches = decision.shape[1], decision.shape[2]

#Save a generated image to track progress before training
firstImage = featureNP[0]
saveProgressImage(generator, firstImage, 0)

#Main training loop
for i in range(nSteps):
    
    #Chose a random selection of images for this batch, and generate required data for training
    feats, realLabels, targs = generateRealSamples(featureNP, targetNP, batchSize, nXPatches, nYPatches)
    fakeOutput, fakeLabels = generateFakeSamples(generator, feats, nXPatches, nYPatches)
    
    w_noise = np.random.normal(0, 1, (batch_size, 32, 16, 1024))
    losses_composite = compositeModel.train_on_batch([feats, w_noise], [realLabels, targs])
    
    loss_discriminator_fake = discriminator.train_on_batch([fakeOutput, feats], fakeLabels)
    loss_discriminator_real = discriminator.train_on_batch([targs, feats], realLabels)
    
    d_loss = (loss_discriminator_fake + loss_discriminator_real) / 2
    avg_dloss = avg_dloss + (d_loss - avg_dloss) / (i + 1)
    avg_loss = avg_loss + (losses_composite[0] - avg_loss) / (i + 1)
    print('total loss:' + str(avg_loss) + ' d_loss:' + str(avg_dloss))
    
    #Save progress images of the generator output
    if i % 1 == 0:
        firstImage = featureNP[0]
        saveProgressImage(generator, firstImage, i)




