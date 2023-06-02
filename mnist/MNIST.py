import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from keras.datasets import mnist
from keras.models import Sequential, load_model

from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

from PIL import Image


# load the dataset
def main():
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()

    # data pre-processing
    xTrain = xTrain.reshape(60000, 784)
    xTest = xTest.reshape(10000, 784)
    xTrain = xTrain.astype("float32")
    xTest = xTest.astype("float32")
    xTrain /= 255
    xTest /= 255

    # one-hot encoding of data:
    classi = 10
    yTrain = np_utils.to_categorical(yTrain, classi)
    yTest = np_utils.to_categorical(yTest, classi)

    # initialize the model
    model = (
        Sequential()
    )  # used to initialize the linear stack of layers, layers are added sequentially

    # add the first hidden layer
    model.add(
        Dense(512, input_shape=(784,))
    )  # Dense = fully connected layer of 512 neurons
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    # add the second hidden layer
    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.summary()

    # compile the model
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # fit the model
    model.fit(xTrain, yTrain, batch_size=128, epochs=5, verbose=1)

    # evaluate the model
    # evaluating the model against the xTest and yTest datasets of 10000 images

    score = model.evaluate(xTest, yTest)
    print("Test score:", score[0])
    print("Test accuracy:", score[1])
    model.save("MNIST.h5")


def loading_model(name):
    model = load_model(name)
    return model


def preprocess(img):
    # convert in 28x28 and on grayscale
    img = img.resize((28, 28)).convert("L")
    # convert in numpy array and normalize
    img = np.array(img) / 255
    # reshape image data for model
    # img = img.reshape(1, -1)
    img = img.reshape(784, 1).T

    return img


def load_image(path):
    img = Image.open(path)
    img = preprocess(img)
    plt.imshow(img.reshape(28, 28), cmap="gray")
    plt.show()
    return img


def predict(img):
    name = "MNIST.h5"
    model = loading_model(name)
    prediction = model.predict(img.reshape(1, 784))
    digitsPredicted = np.argmax(prediction)

    return digitsPredicted


def testMNIST(model):
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess a single image from the MNIST test set
    ran = random.randint(0, 10000)
    test_image = x_test[ran] / 255.0
    test_image = test_image.reshape(1, -1)

    # Predict the digit
    prediction = model.predict(test_image)

    # Print the predicted and actual digit
    predicted_digit = np.argmax(prediction)
    print("Predicted digit:", predicted_digit)
    print("Actual digit:", y_test[ran])


def testxTrain():
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()

    # data pre-processing
    xTrain = xTrain.reshape(60000, 784)
    xTest = xTest.reshape(10000, 784)
    xTrain = xTrain.astype("float32")
    xTest = xTest.astype("float32")
    xTrain /= 255
    xTest /= 255

    return xTrain, xTest


if __name__ == "__main__":
    # main()
    name = "MNIST.h5"
    path = "otto.png"
    img = load_image(path)
    digit = predict(img)
    model = loading_model(name)
    _, xtest = testxTrain()
    rn = random.randint(0, 10000)
    m = model.predict(xtest[rn].reshape(1, 784))
    print("Acatashiwà", np.argmax(m[0]))
    testMNIST(model)

    print("Predicted digit: {}".format(digit))
