A Generative Adversarial Network (GAN) is a type of neural network architecture used for generative modeling — learning to generate new data with the same statistics as the training data. A GAN consists of two parts:

Architecture of GAN:
Generator (G):

Takes random noise as input and generates fake data (e.g., fake images).

Tries to fool the Discriminator into believing the generated data is real.

Discriminator (D):

Takes real data and fake data as input and tries to distinguish between them.

Outputs a probability (0 to 1) representing the likelihood that the input is real.

These two networks are trained simultaneously in a minimax game:

Generator tries to minimize the discriminator's ability to detect fake data.

Discriminator tries to maximize its classification accuracy.

Working Process:
Sample random noise and generate fake data using the Generator.

Feed real data and fake data to the Discriminator.

Calculate the Discriminator loss using binary cross-entropy:

Real data → target: 1

Fake data → target: 0

Update Discriminator weights to improve real/fake classification.

Sample noise, generate fake data again.

Pass fake data to the Discriminator.

Calculate Generator loss (how well it fools the Discriminator).

Update Generator weights to improve realism of generated data.


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
batch_size = 64
lr = 0.0002
epochs = 10
z_dim = 100

# Data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])
mnist = datasets.MNIST(root='.', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, z):
        return self.net(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Initialize models
G = Generator(z_dim)
D = Discriminator()

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    for real_imgs, _ in dataloader:
        real_imgs = real_imgs.view(-1, 784)
        batch_size = real_imgs.size(0)

        # Labels
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # ======================
        # Train Discriminator
        # ======================
        z = torch.randn(batch_size, z_dim)
        fake_imgs = G(z)

        real_loss = criterion(D(real_imgs), real_labels)
        fake_loss = criterion(D(fake_imgs.detach()), fake_labels)
        d_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # ======================
        # Train Generator
        # ======================
        z = torch.randn(batch_size, z_dim)
        fake_imgs = G(z)
        g_loss = criterion(D(fake_imgs), real_labels)  # try to fool D

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}]  D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}")

```
output:
![Screenshot 2025-05-24 105232](https://github.com/user-attachments/assets/17edd6e6-e54d-4501-be4e-12a5e66cb80a)

