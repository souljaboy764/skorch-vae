import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from skorch import NeuralNet



class AE(nn.Module):
	"""[Autoencoder Class for learning latent space representations of input data. Code is inspired from https://github.com/graviraja/pytorch-sample-codes/blob/master/simple_vae.py]
	
	Args:
		input_dim (int): [Input Dimension of the Encoder]
		hidden_dim (int): [Number of the states in the hidden layers]
		z_dim (int): [Latent Space Dimension of the Autoencoder]
		activation : [activation function for the hidden layers. Default: LeakyReLU(0.5)]
	"""

	def __init__(self, input_dim, hidden_dim, z_dim, activation=nn.LeakyReLU(0.5)):
		super(AE, self).__init__()
		self._enc = self.Encoder(input_dim, hidden_dim, z_dim,activation)
		self._dec = self.Decoder(z_dim, hidden_dim, input_dim, activation)

	def forward(self, x):
		"""[Forward pass of the Autoencoder. The input `x` is first encoded to the latent space which is then decoded to reconstruct the input]

		Args:
			x (Tensor [batch_size, input_dim]): [Input sample to autoencode]
		
		Returns:
			x_gen (Tensor [batch_size, input_dim]): [Generated output decoded from the encoding vectors of the input]
			z (Tensor [batch_size, z_dim]): [Latent space Encodings of the inputs]
		"""
		# encode 
		z = self._enc(x)
		
		#decode
		x_gen = self._dec(z)
		return x_gen, x, z
		
	class Encoder(nn.Module):
		"""[Encoder Class Q(X) for an Autoencoder]
		
		Args:
			input_dim (int): [Input Dimension of the Encoder]
			hidden_dim (int): [Number of the states in the hidden layers]
			z_dim (int): [Latent Space Dimension of the Autoencoder]
			activation : [activation function for the hidden layers. Default: LeakyReLU(0.5)]
		"""
		def __init__(self, input_dim, hidden_dim, z_dim, activation=nn.LeakyReLU(0.5)):
			super(AE.Encoder, self).__init__()
			self._hidden1 = nn.Linear(input_dim, hidden_dim)
			self._hidden2 = nn.Linear(hidden_dim, hidden_dim)
			self._latent = nn.Linear(hidden_dim, z_dim)
			self._activation = activation
			
		def forward(self, x):
			"""[Forward pass of the Encoder]

			Args:
				x (Tensor [batch_size, input_dim]): [Input samples to encode]

			Returns:
				(Tensor [batch_size, z_dim]): [Latent space Encodings of the inputs]
			"""
			hidden = self._activation(self._hidden1(x))
			hidden = self._activation(self._hidden2(hidden))
			
			return self._latent(hidden)
			
	class Decoder(nn.Module):
		"""[Decoder Class P(z) for an Autoencoder]
		
		Args:
			z_dim (int): [Latent Space Dimension of the Autoencoder]
			hidden_dim (int): [Number of the states in the hidden layers]
			output_dim (int): [Output Dimension of the Decoder]
			activation : [activation function for the hidden layers. Default: LeakyReLU(0.5)]
		"""
		def __init__(self, z_dim, hidden_dim, output_dim, activation=nn.LeakyReLU(0.5)):
			super(AE.Decoder, self).__init__()
			self._hidden1 = nn.Linear(z_dim, hidden_dim)
			self._hidden2 = nn.Linear(hidden_dim, hidden_dim)
			self._output = nn.Linear(hidden_dim, output_dim)
			self._activation = activation

		def forward(self, z):
			"""[Forward pass of the Decoder]

			Args:
				z (Tensor [batch_size, z_dim]): [Encoded Lantent space vectors to decode]

			Returns:
				(Tensor [batch_size, output_dim]): [Generated output of the encoded vectors]
			"""
			hidden = self._activation(self._hidden1(z))
			hidden = self._activation(self._hidden2(hidden))
			
			return self._output(hidden)

class VAE(AE):
	"""[Variational Autoencoder Class for learning latent space representations of input data. Code is inspired from https://github.com/graviraja/pytorch-sample-codes/blob/master/simple_vae.py]
	
	Args:
		input_dim (int): [Input Dimension of the Encoder]
		hidden_dim (int): [Number of the states in the hidden layers]
		z_dim (int): [Latent Space Dimension of the VAE]
		activation : [activation function for the hidden layers. Default: LeakyReLU(0.5)]
		num_samples (int, optional): [The number of samples to draw for Monte Carlo Estimation of the reconstruction loss. Default: 10]
	"""

	def __init__(self, input_dim, hidden_dim, z_dim, activation=nn.LeakyReLU(0.5), num_samples=10):
		super(VAE, self).__init__(input_dim, hidden_dim, z_dim, activation)
		self._enc = self.Encoder(input_dim, hidden_dim, z_dim, activation)
		self.num_samples = num_samples
		
	def forward(self, x):
		"""[Forward pass of the VAE. The input `x` is encoded to obtain a laten space distribution, from which `num_samples` number of samples are drawn and decoded to reconstruct the input.]

		Args:
			x (Tensor [batch_size, input_dim]): [Input sample to the VAE]
		
		Returns:
			x_dists (Tensor [batch_size*num_samples, input_dim]): [Generated output P(X|z) decoded from the Monte Carlo latent space samples.]
			z_samples (Tensor [num_samples, batch_size, z_dim]): [Monte Carlo Samples drawn from the encoded latent space distribution Q(z|X).]
			z_dist (torch.distributions.normal.Normal): [Predicted Normal distribution of the encoded latent space distribution Q(z|X). (batch_shape=batch_size, event_shape=z_dim)]
		"""
		# encode 
		z_dist = self._enc(x)
		
		# Decoding from Monte Carlo Estimation
		z_samples = z_dist.rsample((self.num_samples,))
		x_gen = self._dec(z_samples).view(-1, x.shape[-1])

		return x_gen, x.repeat(self.num_samples,1), z_samples, z_dist

	class Encoder(AE.Encoder):
		"""[Encoder Class Q(z|X) for the VAE]
		
		Args:
			input_dim (int): [Input Dimension of the Encoder]
			hidden_dim (int): [Number of the states in the hidden layers]
			z_dim (int): [Latent Space Dimension of the VAE]
			activation : [activation function for the hidden layers. Default: LeakyReLU(0.5)]
		"""
		def __init__(self, input_dim, hidden_dim, z_dim, activation=nn.LeakyReLU(0.5)):
			super(VAE.Encoder, self).__init__(input_dim, hidden_dim, z_dim, activation)
			self._logstd = nn.Linear(hidden_dim, z_dim)
			
		def forward(self, x):
			"""[Forward pass of the Encoder]

			Args:
				x (Tensor [batch_size, input_dim]): [Input sample to encode]

			Returns:
				(torch.distributions.normal.MultivariateNormal): [Predicted MultivariateNormal distribution of the latent space Q(z|X).]
			"""
			hidden = self._activation(self._hidden1(x))
			hidden = self._activation(self._hidden2(hidden))
			
			z_mean = self._latent(hidden)
			
			return Normal(z_mean, torch.exp(self._logstd(hidden)))

class VAEWrapper(NeuralNet):
	def get_loss(self, y_pred, y_true, *args, **kwargs):
		x_gen, x, z_samples, z_dist = y_pred  # <- unpack the tuple that was returned by `forward`
		recon_loss = super().get_loss(x_gen, x, *args, **kwargs)
		kl_loss = kl_divergence(z_dist, Normal(0, 1)).mean()
		return recon_loss + kl_loss

class AEWrapper(NeuralNet):
	def get_loss(self, y_pred, y_true, *args, **kwargs):
		x_gen, x, z_samples = y_pred  # <- unpack the tuple that was returned by `forward`
		return super().get_loss(x_gen, x, *args, **kwargs)
