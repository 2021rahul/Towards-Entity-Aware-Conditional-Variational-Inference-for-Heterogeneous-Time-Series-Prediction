#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES

# In[1]:


import numpy as np
import torch
from EALSTM import EALSTM


# # LOSS FUNCTIONS

# ## CONTRASTIVE LOSS

# In[2]:


class SimCLRLoss(torch.nn.Module):
	def __init__(self, temperature):
		super(SimCLRLoss, self).__init__()
		self.temperature = temperature
		self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
		self.similarity = torch.nn.CosineSimilarity(dim=2)

	def mask_correlated_samples(self, batch_size):
		N = 2 * batch_size
		mask = torch.ones((N, N), dtype=bool)
		mask = mask.fill_diagonal_(0)

		for i in range(batch_size):
			mask[i, batch_size + i] = 0
			mask[batch_size + i, i] = 0
		return mask

	def forward(self, z):

		z = torch.nn.functional.normalize(z, p=2.0, dim=1)

		N = z.shape[0]
		batch_size = N//2

		sim = self.similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

		sim_i_j = torch.diag(sim, batch_size)
		sim_j_i = torch.diag(sim, -batch_size)

		positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
		mask = self.mask_correlated_samples(batch_size)
		negative_samples = sim[mask].reshape(N, -1)

		#SIMCLR
		labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long()

		logits = torch.cat((positive_samples, negative_samples), dim=1)
		loss = self.criterion(logits, labels)
		loss /= N

		return loss


# ## KL LOSS

# In[3]:


class KLLoss(torch.nn.Module):
	def __init__(self):
		super(KLLoss, self).__init__()

	def forward(self, z, mu, std):
		# 1. define the first two probabilities (in this case Normal for both)
		p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
		q = torch.distributions.Normal(mu, std)

		# 2. get the probabilities from the equation
		log_qzx = q.log_prob(z)
		log_pz = p.log_prob(z)

		# loss
		loss = (log_qzx - log_pz)
		loss = torch.mean(torch.sum(loss, dim=1), dim = 0)
		return loss


# # FORWARD MODEL

# ## LSTM

# In[4]:


class lstm(torch.nn.Module):

	def __init__(self, input_channels, code_dim, output_channels):
		super(lstm,self).__init__()

		# PARAMETERS
		self.input_channels = input_channels
		self.code_dim = code_dim
		self.output_channels = output_channels

		# LAYERS
		self.encoder = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.code_dim, batch_first=True)
		self.out = torch.nn.Linear(in_features=self.code_dim, out_features=self.output_channels)

		# INITIALIZATION
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight)

	def forward(self, x_dynamic):

		# GET SHAPES
		batch, window, _ = x_dynamic.shape

		# OPERATIONS
		x_encoder, _ = self.encoder(x_dynamic)
		out = self.out(x_encoder)
		out = out.view(batch, window, self.output_channels)

		return out


# ## EALSTM

# In[5]:


class ealstm(torch.nn.Module):

	def __init__(self, input_dynamic_channels, input_static_channels, code_dim, output_channels):
		super(ealstm,self).__init__()

		# PARAMETERS
		self.input_dynamic_channels = input_dynamic_channels
		self.input_static_channels = input_static_channels
		self.code_dim = code_dim
		self.output_channels = output_channels

		# LAYERS
		self.encoder = EALSTM(input_size_dyn=self.input_dynamic_channels, input_size_stat=self.input_static_channels, hidden_size=self.code_dim, batch_first=True)
		self.out = torch.nn.Linear(in_features=self.code_dim, out_features=self.output_channels)

		# INITIALIZATION
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight)

	def forward(self, x_dynamic, x_static):

		# GET SHAPES
		batch, window, _ = x_dynamic.shape

		# OPERATIONS
		x_encoder, _ = self.encoder(x_d=x_dynamic, x_s=x_static)
		out = self.out(x_encoder)
		out = out.view(batch, window, self.output_channels)

		return out


# ## CTLSTM

# In[6]:


class ctlstm(torch.nn.Module):

	def __init__(self, input_dynamic_channels, input_static_channels, code_dim, output_channels):
		super(ctlstm,self).__init__()

		# PARAMETERS
		self.input_dynamic_channels = input_dynamic_channels
		self.input_static_channels = input_static_channels
		self.code_dim = code_dim
		self.output_channels = output_channels

		# LAYERS
		self.encoder = torch.nn.LSTM(input_size=self.input_dynamic_channels+self.input_static_channels, hidden_size=self.code_dim, batch_first=True)
		self.out = torch.nn.Linear(in_features=self.code_dim, out_features=self.output_channels)

		# INITIALIZATION
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight)

	def forward(self, x_dynamic, x_static):

		# GET SHAPES
		batch, window, _ = x_dynamic.shape

		# OPERATIONS
		x = torch.cat((x_dynamic, x_static), dim=-1)
		x_encoder, _ = self.encoder(x)
		out = self.out(x_encoder)
		out = out.view(batch, window, self.output_channels)

		return out


# # INVERSE MODEL

# ## AE

# In[7]:


class ae(torch.nn.Module):
	def __init__(self, input_channels, code_dim, output_channels, device):
		super(ae,self).__init__()

		# PARAMETERS
		self.input_channels = input_channels
		self.code_dim = code_dim
		self.output_channels = output_channels
		self.device = device

		# LAYERS
		self.instance_encoder = torch.nn.Sequential(
			torch.nn.Linear(in_features=self.input_channels, out_features=self.code_dim),
			torch.nn.BatchNorm1d(self.code_dim),
			torch.nn.LeakyReLU(0.2)
		)
		self.temporal_encoder = torch.nn.LSTM(input_size=self.code_dim, hidden_size=self.code_dim, bidirectional=True, batch_first=True)	# AE
		torch.nn.BatchNorm1d(self.code_dim)
		self.temporal_decoder = torch.nn.LSTM(input_size=self.code_dim, hidden_size=self.code_dim, batch_first=True)						# AE
		self.instance_decoder = torch.nn.Linear(in_features=self.code_dim, out_features=self.input_channels)								# AE
		self.static_out = torch.nn.Linear(in_features=self.code_dim, out_features=self.output_channels)

		# INITIALIZATION
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight)

	def forward(self, x):

		# GET SHAPES
		batch, window, _ = x.shape

		# OPERATIONS

		x_encoder = self.instance_encoder(x.view(-1, self.input_channels)).view(batch, window, -1)								# ENCODE
		_, x_encoder = self.temporal_encoder(x_encoder)																			# ENCODE
		code_vec = torch.sum(x_encoder[0], dim=0)																				# ENCODE

		static_out = self.static_out(code_vec)																					# STATIC DECODE

		out = torch.zeros(batch, window, self.input_channels).to(self.device)													# DECODE
		input = torch.unsqueeze(torch.zeros_like(code_vec), dim=1)																# DECODE
		h = (torch.unsqueeze(torch.sum(x_encoder[0], dim=0), dim=0), torch.unsqueeze(torch.sum(x_encoder[1], dim=0), dim=0))	# DECODE
		for step in range(window):																								# DECODE
			input, h = self.temporal_decoder(input, h)																			# DECODE
			out[:,step] = self.instance_decoder(input.squeeze())																# DECODE

		return code_vec, static_out, out


# ## VAE

# In[8]:


class vae(torch.nn.Module):
	def __init__(self, input_channels, code_dim, output_channels, device):
		super(vae,self).__init__()

		# PARAMETERS
		self.input_channels = input_channels
		self.code_dim = code_dim
		self.output_channels = output_channels
		self.device = device

		# LAYERS
		self.instance_encoder = torch.nn.Sequential(
			torch.nn.Linear(in_features=self.input_channels, out_features=self.code_dim),
			torch.nn.BatchNorm1d(self.code_dim),
			torch.nn.LeakyReLU(0.2)
		)
		self.temporal_encoder = torch.nn.LSTM(input_size=self.code_dim, hidden_size=self.code_dim, bidirectional=True, batch_first=True)	# VAE
		self.mu = torch.nn.Linear(self.code_dim, self.code_dim)																				# VAE
		self.log_var = torch.nn.Linear(self.code_dim, self.code_dim)																		# VAE
		self.temporal_decoder = torch.nn.LSTM(input_size=self.code_dim, hidden_size=self.code_dim, batch_first=True)						# VAE
		self.instance_decoder = torch.nn.Linear(in_features=self.code_dim, out_features=self.input_channels)								# VAE
		self.static_out = torch.nn.Linear(in_features=self.code_dim, out_features=self.output_channels)

		# INITIALIZATION
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight)

	def forward(self, x):

		# GET SHAPES
		batch, window, _ = x.shape

		# OPERATIONS

		x_encoder = self.instance_encoder(x.view(-1, self.input_channels)).view(batch, window, -1)	# ENCODE
		_, x_encoder = self.temporal_encoder(x_encoder)												# ENCODE
		code_vec = torch.sum(x_encoder[0], dim=0)													# ENCODE

		mu, log_var = self.mu(code_vec), self.log_var(code_vec)										# SAMPLE Z
		std = torch.exp(log_var/2)																	# SAMPLE Z
		z = mu + std * torch.randn_like(std)														# SAMPLE Z

		static_out = self.static_out(z)																# STATIC DECODE

		out = torch.zeros(batch, window, self.input_channels).to(self.device)						# DECODE
		input = torch.unsqueeze(torch.zeros_like(z), dim=1)											# DECODE
		h = (torch.unsqueeze(z, dim=0), torch.unsqueeze(torch.sum(x_encoder[1], dim=0), dim=0))		# DECODE
		for step in range(window):																	# DECODE
			input, h = self.temporal_decoder(input, h)												# DECODE
			out[:,step] = self.instance_decoder(input.squeeze())									# DECODE

		return z, mu, std, static_out, out


# # TEST MODELS

# In[9]:


if __name__ == "__main__":
	batch = 10
	window = 365
	channels = list(range(33))
	static_channels = channels[:27]
	dynamic_channels = channels[27:32]
	output_channels = [channels[-1]]
	data = torch.randn(batch, window, len(static_channels)+len(dynamic_channels))
	data_dynamic = data[:,:,dynamic_channels]
	data_static = data[:,:,static_channels]
	print(data.shape, "DATA")

	code_dim = 128
	device = torch.device("cuda")

	architecture = "lstm"
	model = globals()[architecture](input_channels=len(dynamic_channels), code_dim=code_dim, output_channels=len(output_channels))
	model = model.to(device)
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	out = model(data_dynamic.to(device))
	print(data[:, :, dynamic_channels].shape, out.shape, "#:{}".format(pytorch_total_params), architecture)

	architecture = "ealstm"
	model = globals()[architecture](input_dynamic_channels=len(dynamic_channels), input_static_channels=len(static_channels), code_dim=code_dim, output_channels=len(output_channels))
	model = model.to(device)
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	out = model(data_dynamic.to(device), data_static[:,0].to(device))
	print(data.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)

	architecture = "ctlstm"
	model = globals()[architecture](input_dynamic_channels=len(dynamic_channels), input_static_channels=len(static_channels), code_dim=code_dim, output_channels=len(output_channels))
	model = model.to(device)
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	out = model(data_dynamic.to(device), data_static.to(device))
	print(data.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)

	architecture = "ae"
	model = globals()[architecture](input_channels=len(dynamic_channels), code_dim=code_dim, output_channels=len(static_channels), device=device)
	model = model.to(device)
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	code_vec, static_out, out = model(data[:, :, dynamic_channels].to(device))
	print(data.shape, code_vec.shape, static_out.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)

	architecture = "vae"
	model = globals()[architecture](input_channels=len(dynamic_channels), code_dim=code_dim, output_channels=len(static_channels), device=device)
	model = model.to(device)
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	code_vec, mu, std, static_out, out = model(data[:, :, dynamic_channels].to(device))
	print(data.shape, code_vec.shape, mu.shape, std.shape, static_out.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)


# # COPY MODEL PARAMETERS

# In[10]:


if __name__ == "__main__":

	architecture = "ae"
	model1 = globals()[architecture](input_channels=len(dynamic_channels), code_dim=code_dim, output_channels=len(static_channels), device=device)
	model1 = model1.to(device)

	architecture = "vae"
	model2 = globals()[architecture](input_channels=len(dynamic_channels), code_dim=code_dim, output_channels=len(static_channels), device=device)
	model2 = model2.to(device)

	model2.load_state_dict(model1.state_dict(), strict=False)

