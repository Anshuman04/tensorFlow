import tensorflow as tf


##########################################################################
# Custom callback to provide condition on when we should stop the training

class customCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("acc", 0) >= 0.9:
            print "As accuracy has already crossed 90%. Stopping training."
            self.model.stop_training = True
callBackObj = customCallBack()
###########################################################################


###########################################################################
# Load the fashion MNIST data
mnist = tf.keras.datasets.fashion_mnist
(train_imgs, train_labels), (test_imgs, test_labels) = mnist.load_data()


# CNNs first layer expects a single tensor with al the information.
# Thus instead of sending images as (60000,28,28), we will send it as (60000,28,28,1)
train_imgs = train_imgs.reshape(60000,28,28,1)
test_imgs = test_imgs.reshape(10000,28,28,1)


# Neural networks works best with normalization.
train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

# Defining the model
model = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=(28,28,1)),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=(13,13,1)),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(128, activation="relu"),
                             tf.keras.layers.Dense(10, activation="softmax")])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.summary()      # Prints the structure of our CNN model.
model.fit(train_imgs, train_labels, epochs=5, callbacks=[callBackObj])
test_loss = model.evaluate(test_imgs, test_labels)
print "Loss observed on test dataset: {}".format(test_loss)




