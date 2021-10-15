# TensorFlow and tf.keras
import tensorflow
from tensorflow import keras as tf
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


"""Understanding the Data"""

#Returns the shape and dimensionality of the model (size, dimx, dimy)
train_images.shape

#Just the label size which should = the size of the model
len(train_labels)

#Label is set 0 -> 9 Mapped to the following
"""
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
"""
train_labels

#Same applies to test images as well
test_images.shape

len(test_labels)

"""Data Preprocessing"""
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#Because the pixels are 255 (for the different color values) 1/255 scales them between 0 -> 1 as essentially attributes for each input of the DNN
train_images = train_images / 255.0

test_images = test_images / 255.0

#Essentially this is just to verify 25 images are correctly formatted
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
"""Model Building (What we talked about in the Workshop)"""
"""This model does the following
    we're flatten the 2D array of 28 x 28 (pixel size for input images) into a 1d Array of 784 (28 * 28) for each input value
    We are creating 2 Hidden Layers of the neural net where one is size 128 and the other size 10
    Our Activation function is 'relu' which I did not cover but it is one of the best for classification. """
model = tf.Sequential([
    tf.layers.Flatten(input_shape=(28, 28)),
    tf.layers.Dense(128, activation='relu'),
    tf.layers.Dense(10)
])
"""After the model is created we compile it using the following
   Optimizer is essentially how the weights are changed which is what Gradient Descent was in the workshop (This isn't Gradient Descent) 
   
   Loss function is how we calculated our residuals in the workshop. Cross entropy is simply how similar classes are to each other
   
   metrics we are defining on accuracy which is just how close we were to being correct"""
model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#This model will run over the train data with there respective train labels 10 times
model.fit(train_images, train_labels, epochs=10)
#Tests the accuracy of the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

"""Playing around to view prediction values"""
probability_model = tf.Sequential([model,
                                         tf.layers.Softmax()])
predictions = probability_model.predict(test_images)

#Model confidence in which value it believes it is
predictions[0]

#Argmax takes the value with the most confidence (Which is 9 in this case)
np.argmax(predictions[0])

#If this is 9 than the prediction and test label matched
test_labels[0]


"""Im not going to get into how these functions work because this is data visualization and my main goal is to
Get you to understand the Neural net part before this I may however touch on this in the workshop :)"""
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()