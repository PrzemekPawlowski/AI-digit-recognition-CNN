This simple model predicts a handwritten number. After 5 epochs, 
the model predicts the number with 98% accuracy, the loss value is approx 0.03.
On the test data, we get an accuracy of 98%, which tells us that the model is not overfitting.

![image](https://user-images.githubusercontent.com/83333798/137344235-4486105a-08cb-4ab2-86cf-76aea23ae69c.png)

The file named "prediction_of_own_number.py" implements a simple way to use the model,
to predict the number from the image.

Some examples
![Bez tytułu](https://user-images.githubusercontent.com/83333798/137346272-3bbdcc07-033a-4b59-8f8a-d2eddcd64120.png)

The model.py file uses a sequential model. In addition to the sequential model, we can use functional models. 
Then our network model will look like this: 

# Functional API model
def my_model():
  inputs = layers.keras.Input(shape=(32, 32, 3))
  x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=layers.regularizers.l2(0.01))(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.keras.activations.relu(x)
  x = layers.MaxPooling2D()(x)

  x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=layers.regularizers.l2(0.01))(x)
  x = layers.BatchNormalization()(x)
  x = layers.keras.activations.relu(x)
  x = layers.MaxPooling2D()(x)

  x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=layers.regularizers.l2(0.01))(x)
  x = layers.BatchNormalization()(x)
  x = layers.keras.activations.relu(x)

  x = layers.Flatten()(x)
  x = layers.Dense(64, activation='relu', kernel_regularizer=layers.regularizers.l2(0.01))(x)
  x = layers.Dropout(0.5)(x)
  outputs = layers.Dense(10)(x)
  model = layers.keras.Model(inputs = inputs, outputs = outputs)
  return model

