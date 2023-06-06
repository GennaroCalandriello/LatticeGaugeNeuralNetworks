import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import njit
import os
import shutil

savePath = "trainingSet"
N = 180  # number of points in the grid
noise = False  # introduce noise in the data
numTrainingSamples = 2000
num_of_Gaussians = 100  # how much Gaussians do you want to sum to obtain multimodality
eps = 0.02  # noise parameter
domain = [100, 100]  # maximum extension for x and y
num_of_sinusoide = 6  # number of sinusoide to add to the data

"""This file contains all the functions for generating the initial set of data
(configuration) as a multimodal distribution constructed from a Gaussian mixture
of a total of 100 different Gaussians, each one with random uniform center and standard deviation
Through the command noise=True, a noise is introduced in the data to better train the 
model not only on a gaussian straightforward multimodality.
The aim is to train a model to find weights to extract directly multimodal distribution
configurations. The model is trained in the file FlowBased.py
"""


def multimodalTest():
    """Define multivariate distribution with 3 gaussians in 3D"""
    # Define the number of points to sample
    num_samples = 5000

    # Define the parameters of the Gaussian distributions
    means = [np.array([-3, -3, -3]), np.array([0, 0, 0]), np.array([3, 3, 3])]
    covs = [np.eye(3), np.eye(3), np.eye(3)]
    weights = [0.3, 0.4, 0.3]  # should sum to 1

    # Sample from the Gaussian distributions
    samples = []
    for mean, cov, weight in zip(means, covs, weights):
        num_samples_from_this_gaussian = int(num_samples * weight)
        samples.append(
            np.random.multivariate_normal(mean, cov, num_samples_from_this_gaussian)
        )
    samples = np.concatenate(samples)

    # Create a 3D plot of the samples
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


# @njit(fastmath=True, cache=True)
def initial_state_(N, p, x, y, L):
    """Construct the initial state"""
    a = 1
    for i in range(1, N):
        for j in range(1, N):
            p[i, j] = (
                1
                / (np.sqrt(2 * np.pi))
                * np.exp(-((x[i] - 0.3 * L) ** 2 + (y[j] - 0.3 * L) ** 2) / (2 * a))
            )  # shift di 0.3*L
    return p


def static_plot(x, y, u):
    """Realize a static 3D plot from a 2D x, y grid and a 2D u(x, y) array"""
    fig = plt.figure(figsize=(11, 8), dpi=100)
    ax = fig.add_subplot(projection="3d")
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, u[:, :], cmap=cm.jet, rstride=1, cstride=1)
    # ax.set_zlim(-1, 1)
    ax.set_title(
        f"Multimodal Gaussian Mixture Distribution generated via Normalizing Flow ML Model",
        fontsize=15,
    )
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_zlabel("P(x, y)", fontsize=12)
    # saving figure
    fig.savefig(f"plot_{np.random.randint(0, 23344)}.png")  # change the path
    plt.show()


def uniformNoise(p):
    """Introduce a noise in the sample extracted from
    an uniform distribution centered in 0 and extended from -eps, eps.
    This is done to produce samples to better train and validate the model"""

    l = int(len(p[0]) / 3)

    for i in range(l):
        for j in range(l):
            jU, iU = np.random.randint(0, l * 3), np.random.randint(0, l * 3)
            p[iU, jU] = p[iU, jU] + np.random.uniform(-eps, eps) * 3

    return p


@njit(fastmath=True, cache=True)
def getRandomChoice():
    """Return -1 or 1 a caso"""
    rand = np.random.randint(0, 2)
    if rand == 0:
        return -1
    else:
        return 1


@njit(fastmath=True, cache=True)
def MultimodalGaussian(N, potential, x, y):
    """THE CORE: generate gaussian multimodal distributions picking randomness for
    the parameters of center and standard deviation"""

    # Set the parameters
    max_x = max(x)
    max_y = max(y)
    Ltilde = 1

    for _ in range(num_of_Gaussians):
        # Define the centers of the Gaussians
        centers = [(np.random.uniform(0, max_x), np.random.uniform(0.0, max_y))]

        for cx, cy in centers:
            # rand = np.random.choice([-1, 1])
            rand = getRandomChoice()
            # random standard deviation
            a = np.random.uniform(0, 2)

            for i in range(N):
                for j in range(N):
                    potential[i, j] += rand * np.exp(
                        -(((x[i] - cx * Ltilde)) ** 2 + ((y[j] - cy * Ltilde)) ** 2)
                        / (2 * a)
                    )

    return potential


def gaussian(noise=True, plot=False):
    """Produce the data, insert the noise and eventually plot it"""

    x = np.linspace(0, domain[0], N)
    y = np.linspace(0, domain[1], N)
    p = MultimodalGaussian(N, np.zeros((N, N)), x, y)

    if noise:
        p = uniformNoise(p)

    p = p / np.linalg.norm(p)

    if plot:
        static_plot(x, y, p)
    return p, x, y


def otherDistributions(noise=False):
    x = np.linspace(0, domain[0], N)
    y = np.linspace(0, domain[1], N)
    p = np.zeros((N, N))

    for sin in range(num_of_sinusoide):
        pnew = np.zeros((N, N))
        a = np.random.uniform(0, 0.3, 2)
        b = np.random.uniform(0.01, 0.04, 2)

        for i in range(N):
            for j in range(N):
                pnew[i, j] = b[0] * np.cos(a[0] * x[i] + a[1] * y[j]) + b[1] * np.sin(
                    a[0] * x[i] + a[1] * y[j]
                )
        p += pnew

    return p, x, y

    pass


def generateTrainingSet2(numData=1):
    if os.path.exists(savePath):
        shutil.rmtree(savePath)
    os.makedirs(savePath)

    for num in range(numData):
        print("Generating sinusoidal number: {}".format(num))
        (
            p,
            x,
            y,
        ) = otherDistributions(noise=noise)
        np.savetxt(f"{savePath}/sinusoidal_{num}.txt", p)

    return x, y


def generateTrainingSet(numData=1):
    """TO GENERATE CALL THIS. Generate a training set of numData multimodal distributions"""

    if os.path.exists(savePath):
        shutil.rmtree(savePath)
    os.makedirs(savePath)

    # numData = 3  # how much multimodal distibutions we want to generate
    for num in range(numData):
        print("Generating multimodal number: {}".format(num))
        p, x, y = gaussian(noise=noise)
        np.savetxt(f"{savePath}/multimodal_{num}.txt", p)

    return x, y


def testAndPlot():
    """Plot the training set"""
    if os.path.exists(savePath):
        _, x, y = gaussian()
        for file in os.listdir("trainingSet"):
            join = os.path.join("trainingSet", file)
            p = np.loadtxt(join)
            static_plot(x, y, p)
    else:
        print("No training set found")


def removeDataset():
    """Remove the training set"""
    if os.path.exists(savePath):
        shutil.rmtree(savePath)
    else:
        print("No training set found")


if __name__ == "__main__":
    # Questo programma non fa un cazz
    numTrainingSamples = 10
    # genera i dati
    _, _ = generateTrainingSet(numTrainingSamples)
    # rimuove i dati
    removeDataset()
    # genera la mamma di Raffaele
    # generateOrnella(numOrnelle=10)
    # rimuove la mamma di Raffaele
    # removeOrnella()
