# CIFAR-10 dataset
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.datasets import cifar10

# Load the data set
(trainImages, trainLabels), (testImages, testLabels) = cifar10.load_data()
classes = np.unique(trainLabels)
nClasses = len(classes)


def show():
    plt.figure(figsize=[4, 2])

    # Display the first image in training data
    plt.subplot(121)
    plt.imshow(trainImages[0, :, :], cmap="gray")

    # Display the first image in testing data
    plt.subplot(122)
    plt.imshow(testImages[0, :, :], cmap="gray")


def reshaping():
    nRows, nCols, nDims = trainImages.shape[1:]
    trainData = trainImages.reshape(trainImages.shape[0], nRows, nCols, nDims)
    testData = testImages.reshape(testImages.shape[0], nRows, nCols, nDims)
    inputShape = (nRows, nCols, nDims)

    trainData = trainData.astype("float32")
    testData = testData.astype("float32")

    # normalization
    trainData = trainData / 255
    testData /= 255

    # to_categorical, one-hot encoding
    trainLabelsOneHot = to_categorical(
        trainLabels
    )  # tranLabels is the y in MNIST project
    testLabelsOneHot = to_categorical(testLabels)

    # display the change for category label using one-hot encoding
    ran = np.random.randint(0, trainLabels.shape[0])
    print("Original label {}: {}".format(ran, trainLabels[ran]))
    print(
        "After conversion to categorical ( one-hot): {}".format(trainLabelsOneHot[ran])
    )
    return trainData, testData, trainLabelsOneHot, testLabelsOneHot, inputShape


def createModel():
    trainData, testData, trainLabelsOneHot, testLabelsOneHot, inputShape = reshaping()
    model = Sequential()
    # first two layers with 32 filters
    model.add(
        Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=inputShape)
    )
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # to avoid overfitting

    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(nClasses, activation="softmax"))

    return model


def compile():
    model = createModel()
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    return model


def training():
    batch_size = 256  # model's weights will be updated once for every 256 examples.
    epochs = 50

    trainData, testData, trainLabelsOneHot, testLabelsOneHot, inputShape = reshaping()
    model = compile()
    history = model.fit(
        trainData,
        trainLabelsOneHot,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(testData, testLabelsOneHot),
    )

    # plot the accuracy and loss curves:
    plot = True
    if plot:
        accuracy = history.history["accuracy"]
        validation_accuracy = history.history["val_accuracy"]
        loss = history.history["loss"]
        validation_loss = history.history["val_loss"]
        epochs_range = range(epochs)

        plt.figure()
        plt.plot(epochs_range, accuracy, label="Training Accuracy")
        plt.plot(epochs_range, validation_accuracy, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")
        plt.show()

        plt.figure()
        plt.plot(epochs_range, loss, label="Training loss")
        plt.plot(epochs_range, validation_loss, label="Validation loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")
        plt.show()

    model.evaluate(testData, testLabelsOneHot)
    model.save("cifar10.h5")


if __name__ == "__main__":
    training()
