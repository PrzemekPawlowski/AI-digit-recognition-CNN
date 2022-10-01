# AI-digit-recognition-CNN
This model predicts a number using a convolutional neural network.
The network was built using the Tensorflow framework.

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [About the application](#about-the-application)

## General info
This simple model predicts a handwritten number. After 5 epochs, the model predicts the number with 98% accuracy, the loss value is approx 0.03.
On the test data, we get an accuracy of 98%, which tells us that the model is not overfitting.

![image](https://user-images.githubusercontent.com/83333798/137344235-4486105a-08cb-4ab2-86cf-76aea23ae69c.png)

## Technologies
Project is created with:
* Python programming language version 3.7
* Framework Tensorflow in version 2.10.0
* PyCharm environment with connection to Anaconda
	
## About the application
To run the application, download the Anaconda platform. It allows an easy and quick way to add packages and open source libraries. From this platform, we can also install the PyCharm environment. Later, the necessary libraries, Tensorflow, Numpy, Matplotlib and cv2 should be installed on the Anaonda platform

The file named "prediction_of_own_number.py" implements a simple way to use the model,
to predict the number from the image.

Some examples
![Bez tytułu](https://user-images.githubusercontent.com/83333798/137346272-3bbdcc07-033a-4b59-8f8a-d2eddcd64120.png)

The model.py file uses a sequential model. In addition to the sequential model, we can use functional models. 
Then our network model will look like this: 

![Functional_API_digit_recognition](https://user-images.githubusercontent.com/83333798/141284204-c5e38570-31be-4fa8-983e-206dadf502dc.png)
