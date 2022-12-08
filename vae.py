#!/usr/bin/env python
# coding: utf-8

### Imports ###

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader
import torch.distributions
import numpy as np
from tqdm import tqdm

### Define encoder class ###

class VariationalEncoder(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_shape, int(input_shape/2))
        self.linear2 = nn.Linear(int(input_shape/2), int(input_shape/3))
        self.linear3 = nn.Linear(int(input_shape/3), int(input_shape/4))
        self.linear4 = nn.Linear(int(input_shape/4), latent_dims) #mu
        self.linear5 = nn.Linear(int(input_shape/4), latent_dims) #logstd
        
        self.N = torch.distributions.Normal(0, 1)
        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        self.kl = 0
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mu =  self.linear4(x)
        logstd = self.linear5(x)
        z = mu + logstd.exp()*self.N.sample(mu.shape)
        self.kl = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
        return z
    
### Define conditional decoder class ###
class Decoder(nn.Module):
    def __init__(self, input_shape, target_shape, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims+target_shape, int(input_shape/4))
        self.linear2 = nn.Linear(int(input_shape/4)+target_shape, int(input_shape/3))
        self.linear3 = nn.Linear(int(input_shape/3)+target_shape, int(input_shape/2))
        self.linear4 = nn.Linear(int(input_shape/2)+target_shape, input_shape)
        
    def forward(self, z, targets):
                                 
        ## targets get concatenated to each layer output in decoder ##
        z = F.relu(self.linear1(torch.cat((z, targets), 1))) 
        z = F.relu(self.linear2(torch.cat((z, targets), 1)))
        z = F.relu(self.linear3(torch.cat((z, targets), 1)))
        z = torch.sigmoid(self.linear4(torch.cat((z, targets), 1)))
        return z
    
### CVAE class ###
class CondVariationalAutoencoder(nn.Module):
    def __init__(self, input_shape, target_shape, latent_dims):
        super(CondVariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(input_shape, latent_dims)
        self.decoder = Decoder(input_shape, target_shape, latent_dims)
    
    def forward(self, x, targets):
        z = self.encoder(x)
        return self.decoder(z, targets)
    
### Function for KL annealing ###
def anneal_schedule(epoch):
    return min(epoch/5,1)

### Training function ###
def train(autoencoder, data, epochs):
    
    recon_list = []
    kl_list = []
    
    opt = torch.optim.Adam(autoencoder.parameters())
    epoch_counter = 0
    for epoch in epochs:
        for i, batch in enumerate(tqdm(data, desc="Epoch Progress")):
            x = batch
            x = x.to(device)
            opt.zero_grad()
            
            inp = x[:,:input_shape]
            tar = x[:,-target_shape:]
            x_hat = autoencoder(inp, tar) #Reconstructed samples (with targets)
            
            mse = F.mse_loss(x_hat[:,:len(inp[0])], inp, reduction='mean')
            kl = anneal_schedule(epoch_counter)*autoencoder.encoder.kl
            loss = mse + kl
            
            loss.backward()
            opt.step()
            
        print("Epoch %s " % (epoch_counter), "MSE Loss: ", float(mse),"KL Loss: ", float(kl))

        recon_list.append(float(mse))
        kl_list.append(float(kl))
        
        PATH = "/ocean/projects/atm200007p/dsmith1/saved_models_5lat/cvae_epoch%s.pt" \
        % (epoch_counter)
        torch.save({
            'epoch': epoch_counter,
            'model_state_dict': autoencoder.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'recon_loss': recon_list,
            'kl_loss': kl_list
        }, PATH)
        
        epoch_counter+=1
    return autoencoder


### Load data ###

path_input = '/ocean/projects/atm200007p/jlin96/CS274E_datasets/train_input.npy'
data_input = torch.Tensor(np.load(path_input))

path_target = '/ocean/projects/atm200007p/jlin96/CS274E_datasets/train_target.npy'
data_target = torch.Tensor(np.load(path_target))

data = torch.cat((data_target, data_input), 1)

train_dataloader = DataLoader(data, batch_size=64, shuffle=True)

latent_dims = 5
input_shape = len(data_target[0])
target_shape = len(data_input[0])
print("Number of data features: ", input_shape)
print("Number of data targets: ", target_shape)
print("")
print("Number of training samples: ", len(data_target))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

cvae = CondVariationalAutoencoder(input_shape, target_shape, latent_dims).to(device)

### Training loop ###

epoch_tot = 20
epochs = tqdm(range(1, epoch_tot + 1), desc="Epochs")

vae = train(cvae, train_dataloader, epochs)
