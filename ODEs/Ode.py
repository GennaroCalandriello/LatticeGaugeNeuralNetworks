import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""Reference: 
   https://towardsdatascience.com/using-neural-networks-to-solve-ordinary-differential-equations-a7806de99cdd
"""
# loss function as the mean square error of the difference between the predicted and actual values. At this
# one should be summed another term relative to initial conditions, but more terms on the loss function would
# (usually) imply unstable training.

# simple ode: u'=2x, u(0)=1

# defining variables
f0 = 1
g = np.finfo(np.float32).eps
inf_s = np.sqrt(np.finfo(np.float32).eps)
learning_rate = 0.01
epochs = 5000
batch_size = 100
display_step = 500

# network parameters
nInputLayer = 1
nHidden1neurons = 32
nHidden2neurons = 32
nOutputLayer = 1

weights = {
    "h1": tf.Variable(tf.random.normal([nInputLayer, nHidden1neurons])),
    "h2": tf.Variable(tf.random.normal([nHidden1neurons, nHidden2neurons])),
    "out": tf.Variable(tf.random.normal([nHidden2neurons, nOutputLayer])),
}

biases = {
    "b1": tf.Variable(tf.random.normal([nHidden1neurons])),
    "b2": tf.Variable(tf.random.normal([nHidden2neurons])),
    "out": tf.Variable(tf.random.normal([nOutputLayer])),
}
# stochastic gradient descent optimizer
optimizer = tf.optimizers.SGD(learning_rate)


# create model
def Perceptron(x):
    x = np.array([[[x]]], dtype="float32")
    layer1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
    layer1 = tf.nn.sigmoid(layer1)
    layer2 = tf.add(tf.matmul(layer1, weights["h2"]), biases["b2"])
    layer2 = tf.nn.sigmoid(layer2)
    output = tf.matmul(layer2, weights["out"]) + biases["out"]

    return output


# universal approx
def g(x):
    return x * Perceptron(x) + f0


# define our ode:
def ode(x):
    return 2 * x


# loss function to approximate the derivatives
def loss():
    s = []  # summation
    for x in np.linspace(-1, 1, 10):
        dNN = (g(x + inf_s) - g(x)) / inf_s
        s.append((dNN - ode(x)) ** 2)
    return tf.sqrt(tf.reduce_mean(tf.abs(s)))


def trainStep():
    with tf.GradientTape() as tape:
        lossFunc = loss()
    trainableVariables = list(weights.values()) + list(biases.values())
    gradients = tape.gradient(lossFunc, trainableVariables)
    optimizer.apply_gradients(zip(gradients, trainableVariables))


def realSolution(x):
    return x**2 + 1


if __name__ == "__main__":
    # Training the Model:
    for i in range(epochs):
        trainStep()
        if i % display_step == 0:
            print("loss: %f " % (loss()))

    X = np.linspace(0, 1, 100)
    result = []
    for i in X:
        result.append(g(i).numpy()[0][0][0])
    S = realSolution(X)
    plt.plot(X, result)
    plt.plot(X, S)
    plt.show()
