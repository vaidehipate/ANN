'''Write Python program to implement CNN object detection. Discuss numerous performance
evaluation metrics for evaluating the object detecting algorithms' performance.'''

!pip install tensorflow

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train

X_train.shape

X_test.shape

y_train

import matplotlib.pyplot as plt
plt.imshow(X_train[2])

# First we have to change the values of the Pixel Array in the range of 0 to 1 i.e. we need to
# scale them -> coz if the values are in same range then the calculation of weigts will be faster
# i.e. convergence will be faster

# Dividing every value by the max pixel value i.e 255

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255


'''
Understanding the Model
Sequential -> Building the model layer by layer in a sequential order. Each layer flows into the next one

layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))

In the above code 32 indicates -
In a convolutional layer, a filter (or kernel) is like a small window that slides over the input image. Each filter learns to extract certain features from the input.
So 32 indicates that in total 32 times sliding of the window over each pixel of the image will take place and each complete sliding will give us a feature, so in total their will be 32 different features from the input image
(3,3) indicates -
The size of the filter
Pooling -

Dimensionality Reduction Technique,used to reduce the spatial dimensions (width and height) of the input feature maps while retaining important information.
Max Pooling: This type of pooling operation takes small squares of the input data and outputs the maximum value found within each square. It's like zooming out and highlighting the most important feature in that area.
'''


# Create and compile the model
model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = tf.argmax(y_pred, axis=1)

print("Predictions - ",y_pred)
plt.imshow(X_test[0])

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
