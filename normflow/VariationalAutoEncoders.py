import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import numpy as np

from FlowBased import DataStacking

"""!!!!Still Working on this, not finished yet!!!!"""


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        # input_size should be in the form N*N
        super(VAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # defining the encoder
        self.fc1 = nn.Linear(np.prod(input_size), hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)

        # defining the decoder
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, np.prod(input_size))

    def encode(self, x):
        """When ReLU (Rectified Linear Unit) is used as an activation function in the layers
        of the encoder, it helps to introduce non-linearity into the model.
        Autoencoders and other neural networks often use ReLU or a variant
        (like Leaky ReLU, Parametric ReLU) because they help to mitigate the
        vanishing gradients problem. This problem occurs during the training
        process where the gradients of the loss function become very small,
        slowing down the learning process or causing it to stop entirely.
        ReLU helps mitigate this issue because its derivative is always 1 for
        positive inputs, ensuring the gradient does not vanish during the
        backpropagation process."""
        x = x.view(-1, np.prod(self.input_size))  # flatten the input array
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # standard deviation
        eps = torch.randn_like(std)  # random noise
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # sigmoid to get values between 0 and 1
        return torch.sigmoid(self.fc4(h3)).view(-1, *self.input_size)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


def trainingVAE():
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 10

    # load the dataset
    dataloader, val_dataloader = DataStacking()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    for epoch in range(num_epochs):
        for i, (x, _) in enumerate(dataloader):
            # forward pass
            x = x.to(device).view(-1, 784)
            x_reconst, mu, logvar = model(x)

            # compute reconstruction loss and kl divergence
            # for KL divergence, see Appendix B in VAE paper
            reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction="sum")
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # backprop and optimize
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(
                    "Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}".format(
                        epoch + 1,
                        num_epochs,
                        i + 1,
                        len(dataloader),
                        reconst_loss.item(),
                        kl_div.item(),
                    )
                )


if __name__ == "__main__":
    trainingVAE()
