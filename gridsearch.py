import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from skorch import NeuralNet

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

import numpy as np

from models import *

import traceback

torch.manual_seed(42) # Hopefully the answer to life universe and everything can be an answer to our training

train_dataset = datasets.MNIST(root = 'data', train = True, transform = ToTensor())
N,H,W = train_dataset.data.size()
data = train_dataset.data.reshape(N, H*W)/255.0

class VAELoss():
	def __init__(self, estimator=None, X_test=None):
		# if estimator==None:
		# 	pass
		# else:
		# 	y_pred = estimator.module_(X_test.to(estimator.device))
		# 	return self(y_pred, None)
		pass
	
	def __call__(self, y_pred, y_true, **kwargs):
		if isinstance(y_pred, NeuralNet):
			y_pred = y_pred.module_(y_true.to(y_pred.device))
		x_gen, x, z_samples, z_dist = y_pred  # <- unpack the tuple that was returned by `forward`
		recon_loss = ((x_gen - x)**2).mean([0,2])
		kl_loss = kl_divergence(z_dist, Normal(0, 1)).mean(1)
		return (recon_loss + kl_loss).mean()

net = NeuralNet(
	VAE,
	criterion=VAELoss,
	max_epochs=500,
	optimizer=torch.optim.AdamW,
	optimizer__weight_decay=1e-5,
	lr=1e-4,
	batch_size= 64,
	module__input_dim = H*W,
	module__hidden_dim = 20,
	module__z_dim = 10,
	module__activation = nn.LeakyReLU(0.5),
	module__num_samples = 10,
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
net.set_params(train_split=False, verbose=0)
params = {
    'lr': [1e-5],
    'max_epochs': [2],
	'batch_size': [64],
    # 'module__hidden_dim': [40, 50, 100],
	'module__z_dim': [5,10,20],
	# 'module__activation': [nn.LeakyReLU(0.5), nn.ReLU(), nn.ELU()],
	'module__num_samples': [5, 10, 20],
}
gs = GridSearchCV(net, params, scoring=net.criterion(), refit=False, verbose=2, n_jobs=1, cv=2)
gs.fit(data)
print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))