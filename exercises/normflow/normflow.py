import numpy as np
import distributions.multimodal as multimodal
import torch
import pyro
import pyro.distributions as dist
from pyro.distributions.transforms import *
import torch.optim as optim
from torch.distributions import MultivariateNormal
import os
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

numTrainingData = 10
# training data path
trainingPath = "trainingSet"


def loadAndManageData(generation=False):
    if generation:
        (
            x,
            y,
        ) = multimodal.generateTrainingSet(numTrainingData)
        print("Generated training and validation data")

    training_data = []
    for files in os.listdir(trainingPath):
        data = np.loadtxt(os.path.join(trainingPath, files))
        training_data.append(data)

    training_data = np.concatenate(
        training_data, axis=0
    )  # concatenate along first dimension

    # split the data into training and validation set
    np.random.shuffle(training_data)
    split_idx = int(0.8 * len(training_data))  # 80% training data, 20% validation data
    training_set = training_data[:split_idx]
    validation_set = training_data[split_idx:]

    # convert data in PyTorch tensors
    training_set = torch.tensor(training_set, dtype=torch.float32)
    validation_set = torch.tensor(validation_set, dtype=torch.float32)

    return training_set, validation_set


def training():
    # define the base distribution (prior)
    N = 200
    n = N * N
    base_dist = dist.Normal(torch.zeros(N), torch.ones(N))

    # instantiate the model
    # flow_transform = pyro.distributions.transforms.planar(2)
    flow_transform = Planar(n)
    pyro.module(
        "my_transform", flow_transform
    )  # inform Pyro that this module should be learned
    flow_dist = dist.TransformedDistribution(base_dist, [flow_transform])

    # Training
    training_set, validation_set = loadAndManageData()
    optimizer = optim.Adam(flow_transform.parameters())
    num_epochs = 20

    for epochs in range(num_epochs):
        optimizer.zero_grad()
        z, log_jacobians = flow_transform(training_set)
        log_prob = base_dist.log_prob(z) + log_jacobians
        loss = -torch.mean(log_prob)
        loss.backward()
        optimizer.step()

        # validation
        if epochs % (num_epochs / 10) == 0:
            with torch.no_grad():
                z_val, log_jacobians_val = flow_transform(validation_set)
                log_prob_val = base_dist.log_prob(z_val) + log_jacobians_val
                val_loss = -torch.mean(log_prob_val)

            print(f"Epoch {epochs} - Loss: {loss.item()} - Val Loss: {val_loss.item()}")

    # Save the model:
    modelPath = "model"
    if os.path.exists(modelPath):
        shutil.rmtree(modelPath)
    os.makedirs(modelPath)
    torch.save(flow_transform.state_dict(), os.path.join(modelPath, "model.pt"))


if __name__ == "__main__":
    # produce training data of multimodal gaussians for the normalizing flow
    # distr.generateTrainingSet(numTrainingData)
    training()
