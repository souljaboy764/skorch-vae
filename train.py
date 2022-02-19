import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.distributions.kl import kl_divergence

import numpy as np

from models import *

torch.manual_seed(42) # Hopefully the answer to life universe and everything can be an answer to our training
batch_size = 64
lr = 1e-3
hidden_dim = 100
latent_dim = 10
activation = nn.LeakyReLU(0.5)
num_samples = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = datasets.MNIST(root = 'data', train = True, transform = ToTensor(), download = True)
test_dataset = datasets.MNIST(root = 'data', train = False, transform = ToTensor())
train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iterator = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

N,H,W = train_dataset.data.size()

model = VAE(H*W, hidden_dim, latent_dim, activation, num_samples).to(device)

# optimizer
optimizer = optim.RMSprop(model.parameters(), lr=lr)

def run(dataset_iterator):
	total_loss = []
	torch.autograd.set_detect_anomaly(True)

	for i, (x,y) in enumerate(dataset_iterator):
		optimizer.zero_grad()
		x = x.reshape(x.shape[0], -1).to(device)
		x_gen, z_samples, z_dist = model(x)
			
		loss = F.mse_loss(x_gen, x.repeat(model.num_samples,1)) + kl_divergence(z_dist, Normal(0,1)).mean()
				
		total_loss.append(loss)
		if model.training:
			loss.backward()
			optimizer.step()

	return torch.mean(torch.Tensor(total_loss))

for e in range(100):
	model.train()
	train_loss = run(train_iterator)
		
	model.eval()
	test_loss  = run(test_iterator)
	
	print(f'Epoch {e}, Train Loss: {train_loss}, Test Loss: {test_loss}')