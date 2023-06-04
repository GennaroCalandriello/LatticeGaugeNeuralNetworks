import torch
from torch import nn, optim
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
from module.multimodal import *

os.environ[
    "KMP_DUPLICATE_LIB_OK"
] = "True"  # non so cos'è, ma se non lo metto mi caccia cutrecchie

trainPercentage = 0.8  # percentage of the data used for training
validationPercentage = (
    1.0 - trainPercentage
)  # percentage of the data used for validation

trainingPath = r"trainingSet/"

dim = int(N**2)  # dimension of the data, N is in the multimodal.py file
flow_length = 16  # !!!!!number of transformations in the flow!!!!! IMPORTANT


class PlanarFlow(nn.Module):

    """define the planar flow
    z is the input in the planar flow function f(z)=z+uh(w^T*z+b)
    where u, w, b are the parameters
    to be learned, h is a non-linearity function, as a tanh."""

    def __init__(self, dim):
        torchdim = 1
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(torchdim, dim))
        self.bias = nn.Parameter(torch.Tensor(torchdim))
        self.scale = nn.Parameter(torch.Tensor(torchdim, dim))
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(-0.01, 0.01)
        self.scale.data.uniform_(-0.01, 0.01)
        self.bias.data.uniform_(-0.01, 0.01)

    def forward(self, z):
        activation = F.linear(z, self.weight, self.bias)
        return z + self.scale * self.tanh(activation)


class NormalizingFlow(nn.Module):
    """Defining the normalizing flow as a sequence of invertible transformations
    in this case, planar flows"""

    def __init__(self, dim, flow_length):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(flow_length)])

    def forward(self, z):
        for flow in self.flows:
            z = flow(z)
        return z


class PlanarFlowLogDetJacobian(nn.Module):
    """A helper class to compute the determinant of the gradient of
    the planar flow transformation."""

    def __init__(self, affine):
        super().__init__()

        self.weight = affine.weight
        self.bias = affine.bias
        self.scale = affine.scale
        self.tanh = affine.tanh

    def forward(self, z):
        activation = F.linear(z, self.weight, self.bias)
        psi = (1 - self.tanh(activation) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())

        return torch.log(det_grad.abs() + 1e-7)


def Training(
    model, base_dist, optimizer, dataloader, val_dataloader, epochs, batch_size
):
    """Training and validating the model. The function splits the dataset into training (80%) and validation (20%) sets
    trains the model. Every 10 epochs, the model is validated on the validation set."""

    for epoch in range(epochs):
        for batch in dataloader:
            z0 = batch[0]
            zn = model(z0)
            log_pz = base_dist.log_prob(zn)
            log_pz0 = base_dist.log_prob(z0)
            loss = (log_pz0 - log_pz).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % 10 == 0:
                val_loss = 0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        z0 = val_batch[0]
                        zn = model(z0)
                        log_pz = base_dist.log_prob(zn)
                        log_qz0 = base_dist.log_prob(z0)
                        val_loss += (log_qz0 - log_pz).mean().item()
                print(
                    f"Epoch {epoch}, Validation Loss: {val_loss / len(val_dataloader)}"
                )
    savingModel(model)


def DataStacking():
    """Stacking, managing and reorganazing the data in a single array usable from PyTorch, and normalizing it.
    The function returns the training and validation sets."""

    rawArray = []

    for files in os.listdir(trainingPath):
        data = np.loadtxt(os.path.join(trainingPath, files))
        rawArray.append(data)
    print(rawArray[0])
    rawArray = torch.stack([torch.from_numpy(a) for a in rawArray])
    mean = rawArray.mean(dim=0)
    std = rawArray.std(dim=0)
    rawArray = ((rawArray - mean) / std).float()

    # Adjust the shape of the arrays
    rawArray = rawArray.view(rawArray.shape[0], -1)

    dataset = TensorDataset(rawArray)
    train_size = int(trainPercentage * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=False)

    return dataloader, val_dataloader


def savingModel(model, name="multivariateNormal"):
    """Save the model in a .pth file."""
    torch.save(model.state_dict(), name + ".pth")
    print("Model saved")


def loadingAndGenerating(
    model,
    base_dist,
    name="multivariateNormal",
):
    """Load the model and generate a sample from the target distribution."""
    model.load_state_dict(torch.load(name + ".pth"))
    model.eval()

    # Generate an initial sample from the base distribution
    z0 = base_dist.sample((1,))

    # Pass it through the normalizing flow to get a sample from the target distribution
    # This is what we will call a configuration, in the context of Lattice Gauge Theories
    sample = model(z0)
    print("Model loaded and configuration generated")

    return sample


def plotConfiguration(data):
    _, x, y = gaussian()
    data = data.detach().numpy().reshape(N, N)
    static_plot(x, y, data)


def main(generate=True):
    # Generating the initial training configurations set

    if generate:
        _, _ = generateTrainingSet(numTrainingSamples)

    # Generate the initial distribution to which appy the transformations of the flow
    base_dist = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    model = NormalizingFlow(dim, flow_length)
    optimizer = optim.Adam(model.parameters())

    dataloader, val_dataloader = DataStacking()

    epochs = 1000
    batch_size = 100
    Training(
        model, base_dist, optimizer, dataloader, val_dataloader, epochs, batch_size
    )

    s = loadingAndGenerating(model, base_dist)
    plotConfiguration(s)


if __name__ == "__main__":
    generate = 1

    if generate == 0:
        generate = False
    else:
        generate = True

    main(generate)
    # if os.path.exists("module/trainingSet/multimodal_0.txt"):
    #     print("Mannaggia cristo idrocarburo in una valle di solventi apolari")
    # else:
    #     print("Cristo frutto della vergogna e del degrado")
    # # print(np.loadtxt("module/trainingSet/multimodal_0.txt"))
