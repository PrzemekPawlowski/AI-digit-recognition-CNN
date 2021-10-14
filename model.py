import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist # 28x28 images of hand-written digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizing the data (changing range from 0-255 to 0-1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Reshape
IMG_SIZE = 28
x_train_copy = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_test_copy = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='sigmoid'))

# training the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_copy, y_train, epochs=5, batch_size=128)

print(model.summary())

# compute loss and accuracy
val_loss, val_acc = model.evaluate(x_test_copy, y_test)
print('Validation loss: {} \n Validation accuracy: {}'.format(val_loss, val_acc))

# Save and load model
model.save('digit_CNN.model')
# load
new_model = tf.keras.models.load_model('digit_CNN.model')
# predictions
predictions = new_model.predict([x_test_copy])
#print(predictions)
print(np.argmax(predictions[0]))


plt.imshow(x_test[0], cmap=plt.cm.binary) # cmp - color map (convert to black and white)
plt.show()

