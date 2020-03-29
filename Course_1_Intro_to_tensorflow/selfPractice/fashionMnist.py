import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.fashion_mnist

(trainImgs, trainLabels), (testImgs, testLabels) = mnist.load_data()

#########################################################
# Code to display/write sample image for visualization
#sampleIdx = 50
#import cv2
#sampleImg, sampleLabel = trainImgs[sampleIdx], trainLabels[sampleIdx]
#cv2.imwrite("sample.jpg", sampleImg)
#########################################################

# As neural networks works best for numbers 0-1. We will normalize the values of inside the images.
trainImgs = trainImgs / 255.0
testImgs = testImgs / 255.0

# Define the model to be used.
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss = "sparse_categorical_crossentropy")

model.fit(trainImgs, trainLabels, epochs=15)
import pdb
pdb.set_trace()
testLoss = model.evaluate(testImgs, testLabels)
print "Loss observed in guessing: {}".format(testLoss)

