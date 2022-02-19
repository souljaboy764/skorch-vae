import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np

from models import *

torch.manual_seed(42) # Hopefully the answer to life universe and everything can be an answer to our training

train_dataset = datasets.MNIST(root = 'data', train = True, transform = ToTensor())
N,H,W = train_dataset.data.size()
data = train_dataset.data.reshape(N, H*W)/255.0

net = VAEWrapper(
	VAE,
	criterion=torch.nn.MSELoss,
	max_epochs=500,
	optimizer=torch.optim.AdamW,
	optimizer__weight_decay=1e-5,
	lr=1e-4,
	batch_size= 64,
	module__input_dim = H*W,
	module__hidden_dim = 100,
	module__z_dim = 10,
	module__activation = nn.LeakyReLU(0.5),
	module__num_samples = 10,
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

net.fit(data)