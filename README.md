# Handwritten Digit Recognition Project
This project uses the MNIST dataset to train a convolutional neural network (CNN) model to recognize handwritten digits. The model is trained for 10 epochs and evaluated on the test set. The model achieves an accuracy of 98.5% on the test set.

## Dataset
The MNIST dataset is a collection of handwritten digits. The dataset consists of 60,000 training images and 10,000 test images. Each image is a 28x28 grayscale image of a handwritten digit.

## Model
The model is a CNN model. The model consists of the following layers:

* Two convolutional layers with 32 and 64 filters, respectively
* Two max pooling layers
* A flatten layer
* A dense layer with 128 neurons
* A dense layer with 10 neurons, one for each digit
* Training
 The model is trained for 10 epochs using the Adam optimizer. The loss function is sparse categorical crossentropy. The metrics are accuracy and loss.

## Evaluation
The model is evaluated on the test set. The accuracy of the model on the test set is 98.5%.

## Results
The model is able to recognize handwritten digits with a high degree of accuracy. The model can be used to build applications that can read handwritten text.
