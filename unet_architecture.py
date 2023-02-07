
from pickletools import optimize
from typing_extensions import Self
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import time
import math
import cv2
import numpy as np
import tensorflow as tf
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

class Discriminator(nn.Module):
    def __init__(self, in_features, ListLoss):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.List_Loss=['binary_crossEntropy', 'wasserstein']
        self.Loss = self.List_Loss[0]

    def forward(self, x):
        return self.disc(x)

# Change the Generator to a UNET


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)

##loading dataset
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

## implementing LOSS Functions :
# mean squared error
def mean_squared_error( y_true, y_pred):
	sum_square_error = 0.0
	for i in range(len( y_true)):
		sum_square_error += ( y_true[i] - y_pred[i])**2.0
	mean_square_error = 1.0 / len( y_true) * sum_square_error
	return mean_square_error


# binary cross entropy
from math import log, sqrt
def binary_cross_entropy(y_true,y_pred):
	sum_score = 0.0
	for i in range(len(y_true)):
		sum_score += y_true[i] * log(1e-15 + y_pred[i])
	mean_sum_score = 1.0 / len(y_true) * sum_score
	return -mean_sum_score

# categorical cross entropy
def categorical_cross_entropy(y_true, y_pred):
	sum_score = 0.0
	for i in range(len(y_true)):
		for j in range(len(y_true[i])):
			sum_score += y_true[i][j] * log(1e-15 + y_pred[i][j])
	mean_sum_score = 1.0 / len(y_true) * sum_score
	return -mean_sum_score


# wasserstein loss function
def wasserstein( y_true, y_pred):
    return  torch.mean(torch.sum(torch.sqrt((y_true - y_pred))))

# L2 loss function
def L2( y_true, y_pred):
    return  torch.sum(torch.pow((y_true - y_pred)))

 # L1 loss function
def L1( y_true, y_pred):
    return  torch.sum(torch.abs(y_true - y_pred))
    
# perceptual wasserstein function 
def perceptual_wasserstein( y_true, y_pred):
    return  torch.sum(torch.pow((y_true* y_pred)))


##Metrics :
#PSNR

def psnr(self, original, generated):
    mse = np.mean((original -generated) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))




#The Randomization of LOSS functions
def loss_randomization(self,total_epoch, d , denominateur):
        
        '''
        This function does a random choice of 2 loss functions from the list of loss functions we have.
        This randomization is calculated through the use of a % of the total_epoch by using a denominator
        '''


        optimizer = Adam(0.0002, 0.5)

        if(d == total_epoch/denominateur ):
            self.Loss = self.List_Loss[1]

        elif (d == total_epoch/denominateur):
            self.Loss = self.List_Loss[1]

        
def main(self) :
    self.data = [[binary_cross_entropy, wasserstein ] ,[binary_cross_entropy,perceptual_wasserstein],
    [binary_cross_entropy,L1],[binary_cross_entropy,L2]
     ]



criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

start= time.time()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Real Images", img_grid_real, global_step=step
                )
                step += 1


stop=time.time()
print("time execution is : ", stop-start)