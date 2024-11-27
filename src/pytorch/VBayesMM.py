#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: tungbioinfo
"""

import os
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime


#------------------------------------------------------------------------------
#****************************** Utils Functions *******************************
#------------------------------------------------------------------------------


def sigmoid(z):
    return 1. / (1 + torch.exp(-z))


def logit(z):
    return torch.log(z/(1.-z))


def gumbel_softmax(logits, U, temperature, hard=False, threshold = 0.5, eps=1e-20):
    
    z = logits + torch.log(U + eps) - torch.log(1 - U + eps)
    y = 1 / (1 + torch.exp(- z / temperature))
    if not hard:
        return y
    
    y_hard = (y > threshold).double()
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

def log_gaussian(x, mu, sigma):
    """
        log pdf of one-dimensional gaussian
    """
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma)
    return float(-0.5 * np.log(2 * np.pi)) - torch.log(sigma) - (x - mu)**2 / (2 * sigma**2)

#------------------------------------------------------------------------------
#******************* Variational Bayesian Neural Network **********************
#************************* with Spike-and-Slab prior **************************
#------------------------------------------------------------------------------

class VBayesMM(nn.Module):
    
    def __init__(self, d1, d2, num_samples = 100, 
                 batch_size=50, latent_dim=3, unorm_type = 2.0, vnorm_type = 0,
                 learning_rate=0.1, beta_1=0.8, beta_2=0.9, 
                 temperature = 0.5, mu01 = 0, mu02 = 1, rho01=0, rho02=1, lambda01=0, lambda02=1, 
                 hard = False, threshold = 0.5, ssprior = "normal",
                 clipnorm=10., device='cpu', save_path=None):
        
        
        super(VBayesMM, self).__init__() 
        self.device = torch.device(device)
        self.p = latent_dim
        if save_path is None:
            basename = "logdir"
            suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            save_path = "_".join([basename, suffix])

        self.d1 = d1
        self.d2 = d2
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.Unorm_type = unorm_type
        self.Vnorm_type = vnorm_type
        
        # Parameter for optimization 
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm
        self.save_path = save_path
        
        # Parameter for Spike-and-slab variational prior
        self.temp = temperature
        self.hard = hard
        self.ssprior = ssprior
        self.mu01 = mu01
        self.mu02 = mu02 
        self.rho01 = rho01
        self.rho02 = rho02
        self.lambda01 = lambda01
        self.lambda02 = lambda02
        self.threshold = threshold
        self.rho_prior = torch.Tensor([np.log(np.exp(1.3) - 1)]).to(self.device)
        
        total = (self.d1 + 1) * self.p
        a = np.log(total) + 0.1*((1+1)*np.log(self.d1) + np.log(np.sqrt(self.num_samples)*self.d1))
        lm = 1/np.exp(a)
        self.phi_prior = torch.tensor(lm)

        # Variational parameters
        self.qUmain_mean = nn.Embedding(self.d1, self.p, norm_type=self.Unorm_type)
        self.qUmain_std = nn.Embedding(self.d1, self.p, norm_type=self.Unorm_type)
        self.qUbias_mean = nn.Embedding(self.d1, 1, norm_type=self.Unorm_type)
        self.qUbias_std = nn.Embedding(self.d1, 1, norm_type=self.Unorm_type)
        
        
        if self.hard == True:
            self.qVmain_mean = nn.Linear(self.p, self.d2, dtype=torch.float64)
            self.qVmain_std = nn.Linear(self.p, self.d2, dtype=torch.float64)
        else:
            self.qVmain_mean = nn.Linear(self.p, self.d2)
            self.qVmain_std = nn.Linear(self.p, self.d2)
            
        # Spike-and-slab variational prior 
        
        if self.ssprior == "normal":
            self.qUmain_mean_w_mu = nn.Parameter(torch.Tensor(self.d1, self.p).normal_(self.mu01,self.mu02))
            self.qUmain_mean_w_rho = nn.Parameter(torch.Tensor(self.d1, self.p).normal_(self.rho01, self.rho02))
            self.qUmain_mean_w_theta = nn.Parameter(logit(torch.Tensor(self.d1, self.p).uniform_(self.lambda01, self.lambda02)))
            self.qUmain_mean_gamma = None
            
            self.qUmain_std_w_mu = nn.Parameter(torch.Tensor(self.d1, self.p).normal_(self.mu01,self.mu02))
            self.qUmain_std_w_rho = nn.Parameter(torch.Tensor(self.d1, self.p).normal_(self.rho01, self.rho02))
            self.qUmain_std_w_theta = nn.Parameter(logit(torch.Tensor(self.d1, self.p).uniform_(self.lambda01, self.lambda02)))
            self.qUmain_std_gamma = None
            
            self.qUbias_mean_w_mu = nn.Parameter(torch.Tensor(self.d1, 1).normal_(self.mu01,self.mu02))
            self.qUbias_mean_w_rho = nn.Parameter(torch.Tensor(self.d1, 1).normal_(self.rho01, self.rho02))
            self.qUbias_mean_w_theta = nn.Parameter(torch.logit(torch.Tensor(self.d1, 1).uniform_(self.lambda01, self.lambda02)))
            self.qUbias_mean_gamma = None
            
            self.qUbias_std_w_mu = nn.Parameter(torch.Tensor(self.d1, 1).normal_(self.mu01,self.mu02))
            self.qUbias_std_w_rho = nn.Parameter(torch.Tensor(self.d1, 1).normal_(self.rho01, self.rho02))
            self.qUbias_std_w_theta = nn.Parameter(torch.logit(torch.Tensor(self.d1, 1).uniform_(self.lambda01, self.lambda02)))
            self.qUbias_std_gamma = None
            
        else:
            self.qUmain_mean_w_mu = nn.Parameter(torch.Tensor(self.d1, self.p).uniform_(self.mu01,self.mu02))
            self.qUmain_mean_w_rho = nn.Parameter(torch.Tensor(self.d1, self.p).uniform_(self.rho01, self.rho02))
            self.qUmain_mean_w_theta = nn.Parameter(logit(torch.Tensor(self.d1, self.p).uniform_(self.lambda01, self.lambda02)))
            self.qUmain_mean_gamma = None
            
            self.qUmain_std_w_mu = nn.Parameter(torch.Tensor(self.d1, self.p).uniform_(self.mu01,self.mu02))
            self.qUmain_std_w_rho = nn.Parameter(torch.Tensor(self.d1, self.p).uniform_(self.rho01, self.rho02))
            self.qUmain_std_w_theta = nn.Parameter(logit(torch.Tensor(self.d1, self.p).uniform_(self.lambda01, self.lambda02)))
            self.qUmain_std_gamma = None
            
            self.qUbias_mean_w_mu = nn.Parameter(torch.Tensor(self.d1, 1).uniform_(self.mu01,self.mu02))
            self.qUbias_mean_w_rho = nn.Parameter(torch.Tensor(self.d1, 1).uniform_(self.rho01, self.rho02))
            self.qUbias_mean_w_theta = nn.Parameter(torch.logit(torch.Tensor(self.d1, 1).uniform_(self.lambda01, self.lambda02)))
            self.qUbias_mean_gamma = None
            
            self.qUbias_std_w_mu = nn.Parameter(torch.Tensor(self.d1, 1).uniform_(self.mu01,self.mu02))
            self.qUbias_std_w_rho = nn.Parameter(torch.Tensor(self.d1, 1).uniform_(self.rho01, self.rho02))
            self.qUbias_std_w_theta = nn.Parameter(torch.logit(torch.Tensor(self.d1, 1).uniform_(self.lambda01, self.lambda02)))
            self.qUbias_std_gamma = None

        # Optimizer setup
        self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)

    def forward(self, trainX, trainY):
        
        """Forward pass for training"""

        # Convert the list of arrays to a single NumPy array first
        numpy_indices = np.array([trainX.row, trainX.col])
        indices = torch.LongTensor(numpy_indices).to(self.device)
        values = torch.FloatTensor(trainX.data).to(self.device)
        shape = torch.Size(trainX.shape)
        X_ph = torch.sparse_coo_tensor(indices, values, shape)

        Y_ph = torch.as_tensor(trainY, dtype=torch.float32, device=self.device)
        
        # Assume X_ph.values contains logits, reshape and take log
        X_ph = X_ph.coalesce()

        logits = torch.log(X_ph.values().clamp(min=1e-6))  # Clamp values to avoid log(0)
        logits = logits.reshape(1, -1).float()

        # Creating a multinomial distribution to sample
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        
        # Double-check that the probabilities sum to 1
        probs = torch.exp(logits)
        
        # Ensure there are no zero probabilities
        epsilon = 1e-8  # Small constant value
        adjusted_probs = (probs + epsilon) / (1 + epsilon * probs.shape[-1])
        
        batch_ids = dist.Multinomial(total_count=self.batch_size, probs=adjusted_probs)
        batch_ids = batch_ids.sample()
        batch_ids = batch_ids.squeeze()

        X_samples = X_ph.indices()[0]
        X_obs = X_ph.indices()[1]

        sample_ids = X_samples[batch_ids.long()]

        Y_batch = Y_ph[sample_ids]
        X_batch = X_obs[batch_ids.long()]
        
        norm = Y_batch.shape[0] / self.batch_size

        # Use a fixed total count for all samples, ensuring it's scalar 
        total_count = Y_ph.sum(dim=1)
        
        # Sample Spike-and-slab variational prior
        
        qUmain_mean_sigma_w = torch.log(1 + torch.exp(self.qUmain_mean_w_rho))
        qUmain_std_sigma_w = torch.log(1 + torch.exp(self.qUmain_std_w_rho))
        qUbias_mean_sigma_w = torch.log(1 + torch.exp(self.qUbias_mean_w_rho))
        qUbias_std_sigma_w = torch.log(1 + torch.exp(self.qUbias_std_w_rho))
        sigma_prior = torch.log(1 + torch.exp(self.rho_prior))
        
        qUmain_mean_u_w = torch.rand(self.qUmain_mean_w_theta.shape)
        qUmain_std_u_w = torch.rand(self.qUmain_std_w_theta.shape)
        qUbias_mean_u_w = torch.rand(self.qUbias_mean_w_theta.shape)
        qUbias_std_u_w = torch.rand(self.qUbias_std_w_theta.shape)
        qUmain_mean_u_w = qUmain_mean_u_w.to(self.device)
        qUmain_std_u_w = qUmain_std_u_w.to(self.device)
        qUbias_mean_u_w = qUbias_mean_u_w.to(self.device)
        qUbias_std_u_w = qUbias_std_u_w.to(self.device)
        self.qUmain_mean_gamma = gumbel_softmax(self.qUmain_mean_w_theta, qUmain_mean_u_w, self.temp, hard=self.hard, threshold = self.threshold)
        self.qUmain_std_gamma = gumbel_softmax(self.qUmain_std_w_theta, qUmain_std_u_w, self.temp, hard=self.hard, threshold = self.threshold)
        self.qUbias_mean_gamma = gumbel_softmax(self.qUbias_mean_w_theta, qUbias_mean_u_w, self.temp, hard=self.hard, threshold = self.threshold)
        self.qUbias_std_gamma = gumbel_softmax(self.qUbias_std_w_theta, qUbias_std_u_w, self.temp, hard=self.hard, threshold = self.threshold)
        
        qUmain_mean_epsilon_w = dist.Normal(0, 1).sample(self.qUmain_mean_w_mu.shape)
        qUmain_std_epsilon_w = dist.Normal(0, 1).sample(self.qUmain_std_w_mu.shape)
        qUbias_mean_epsilon_w = dist.Normal(0, 1).sample(self.qUbias_mean_w_mu.shape)
        qUbias_std_epsilon_w = dist.Normal(0, 1).sample(self.qUbias_std_w_mu.shape)
        qUmain_mean_epsilon_w = qUmain_mean_epsilon_w.to(self.device)
        qUmain_std_epsilon_w = qUmain_std_epsilon_w.to(self.device)
        qUbias_mean_epsilon_w = qUbias_mean_epsilon_w.to(self.device)
        qUbias_std_epsilon_w = qUbias_std_epsilon_w.to(self.device)
        
        
        self.qUmain_mean.weight.data = self.qUmain_mean_gamma * (self.qUmain_mean_w_mu + qUmain_mean_sigma_w * qUmain_mean_epsilon_w)
        self.qUmain_std.weight.data = self.qUmain_std_gamma * (self.qUmain_std_w_mu + qUmain_std_sigma_w * qUmain_std_epsilon_w)
        std = torch.exp(self.qUmain_std(X_batch) * 0.5)
        self.qUmain = self.qUmain_mean(X_batch) +  torch.randn_like(std) * std
        
        self.qUbias_mean.weight.data = self.qUbias_mean_gamma * (self.qUbias_mean_w_mu + qUbias_mean_sigma_w * qUbias_mean_epsilon_w)
        self.qUbias_std.weight.data = self.qUbias_std_gamma * (self.qUbias_std_w_mu + qUbias_std_sigma_w * qUbias_std_epsilon_w)
        std = torch.exp(self.qUbias_std(X_batch) * 0.5)
        self.qUbias = self.qUbias_mean(X_batch) + torch.randn_like(std) * std
        
        self.qU = self.qUmain + self.qUbias
        
        if self.Vnorm_type == 1 or self.Vnorm_type == 2:
            l_norm = torch.norm(self.qVmain_mean.weight.data, p=self.Vnorm_type, dim=1, keepdim=True)
            self.qVmain_mean.weight.data = self.qVmain_mean.weight.data / l_norm
            
            l_norm = torch.norm(self.qVmain_std.weight.data, p=self.Vnorm_type, dim=1, keepdim=True)
            self.qVmain_std.weight.data = self.qVmain_std.weight / l_norm
            
            std = torch.exp(self.qVmain_std(self.qU) * 0.5)
            self.qV = self.qVmain_mean(self.qU) + torch.randn_like(std) * std
        else:
            std = torch.exp(self.qVmain_std(self.qU) * 0.5)
            self.qV = self.qVmain_mean(self.qU) + torch.randn_like(std) * std
        
        # Calculate KL divergences for each variational parameter set
        
        qUmain_mean_w_phi = sigmoid(self.qUmain_mean_w_theta)
        qUmain_std_w_phi = sigmoid(self.qUmain_std_w_theta)
        qUbias_mean_w_phi = sigmoid(self.qUbias_mean_w_theta)
        qUbias_std_w_phi = sigmoid(self.qUbias_std_w_theta)
        
        kl_Umain_mean = qUmain_mean_w_phi * (torch.log(qUmain_mean_w_phi) - torch.log(self.phi_prior)) + \
            (1 - qUmain_mean_w_phi) * (torch.log(1 - qUmain_mean_w_phi) - torch.log(1 - self.phi_prior)) + \
                qUmain_mean_w_phi * (torch.log(sigma_prior) - torch.log(qUmain_mean_sigma_w) + 
                                     0.5 * (qUmain_mean_sigma_w ** 2 + self.qUmain_mean_w_mu ** 2) / sigma_prior ** 2 - 0.5)
        kl_Umain_mean = torch.sum(kl_Umain_mean)
        
        kl_Umain_std = qUmain_std_w_phi * (torch.log(qUmain_std_w_phi) - torch.log(self.phi_prior)) + \
            (1 - qUmain_std_w_phi) * (torch.log(1 - qUmain_std_w_phi) - torch.log(1 - self.phi_prior)) + \
                qUmain_std_w_phi * (torch.log(sigma_prior) - torch.log(qUmain_std_sigma_w) + 
                                    0.5 * (qUmain_std_sigma_w ** 2 + self.qUmain_std_w_mu ** 2) / sigma_prior ** 2 - 0.5)
        kl_Umain_std = torch.sum(kl_Umain_std)
        
        kl_Ubias_mean = qUbias_mean_w_phi * (torch.log(qUbias_mean_w_phi) - torch.log(self.phi_prior)) + \
            (1 - qUbias_mean_w_phi) * (torch.log(1 - qUbias_mean_w_phi) - torch.log(1 - self.phi_prior)) + \
                qUbias_mean_w_phi * (torch.log(sigma_prior) - torch.log(qUbias_mean_sigma_w) + 
                                     0.5 * (qUbias_mean_sigma_w ** 2 + self.qUbias_mean_w_mu ** 2) / sigma_prior ** 2 - 0.5)
        kl_Ubias_mean = torch.sum(kl_Ubias_mean)
        
        kl_Ubias_std = qUbias_std_w_phi * (torch.log(qUbias_std_w_phi) - torch.log(self.phi_prior)) + \
            (1 - qUbias_std_w_phi) * (torch.log(1 - qUbias_std_w_phi) - torch.log(1 - self.phi_prior)) + \
                qUbias_std_w_phi * (torch.log(sigma_prior) - torch.log(qUbias_std_sigma_w) + 
                                    0.5 * (qUbias_std_sigma_w ** 2 + self.qUbias_std_w_mu ** 2) / sigma_prior ** 2 - 0.5)
        kl_Ubias_std = torch.sum(kl_Ubias_std)
        
        kl_U_main = 0.5 * torch.sum(1 + self.qUmain_std.weight - self.qUmain_mean.weight.pow(2) - self.qUmain_std.weight.exp())
        kl_U_bias = 0.5 * torch.sum(1 + self.qUbias_std.weight - self.qUbias_mean.weight.pow(2) - self.qUbias_std.weight.exp())
        kl_V_main = 0.5 * torch.sum(1 + self.qVmain_std.weight - self.qVmain_mean.weight.pow(2) - self.qVmain_std.weight.exp())
        kl_V_bias = 0.5 * torch.sum(1 + self.qVmain_std.bias - self.qVmain_mean.bias.pow(2) - self.qVmain_std.bias.exp())
        
        tc = total_count[sample_ids]
        
        if not (Y_batch.dtype == torch.int64):
            Y_batch = torch.round(Y_batch).long()
        
        log_prob_y = 0
        for i in range(tc.size(0)):  # Assuming tc and dv have compatible first dimensions
            if Y_batch[i].sum() != tc[i]:
                continue
            Y = dist.Multinomial(total_count=int(tc[i].item()), logits=self.qV[i])
            log_prob_y += Y.log_prob(Y_batch[i])

        ELBO = (log_prob_y * norm) + (kl_U_main + kl_U_bias + kl_V_main + kl_V_bias + 
                                      kl_Umain_mean + kl_Umain_std + kl_Ubias_mean + kl_Ubias_std)
        
        log_loss = -ELBO
        return log_loss
    
    def validate(self, testX, testY):
        
        """Validation pass for training"""
        
        numpy_indices = np.array([testX.row, testX.col])
        indices = torch.LongTensor(numpy_indices).to(self.device)
        values = torch.FloatTensor(testX.data).to(self.device)
        shape = torch.Size(testX.shape)
        X_ph = torch.sparse_coo_tensor(indices, values, shape)

        Y_ph = torch.as_tensor(testY, dtype=torch.float32, device=self.device)
        
        # Assume X_ph.values contains logits, reshape and take log
        X_ph = X_ph.coalesce()
        logits = torch.log(X_ph.values().clamp(min=1e-6))  # Clamp values to avoid log(0)
        logits = logits.reshape(1, -1).float()

        # Creating a multinomial distribution to sample
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        
        # Double-check that the probabilities sum to 1
        probs = torch.exp(logits)
        
        # Ensure there are no zero probabilities
        epsilon = 1e-8  # Small constant value
        adjusted_probs = (probs + epsilon) / (1 + epsilon * probs.shape[-1])
        
        batch_ids = dist.Multinomial(total_count=self.batch_size, probs=adjusted_probs)
        batch_ids = batch_ids.sample()
        batch_ids = batch_ids.squeeze()

        X_samples = X_ph.indices()[0]
        X_obs = X_ph.indices()[1]

        sample_ids = X_samples[batch_ids.long()]

        Y_batch = Y_ph[sample_ids]
        X_batch = X_obs[batch_ids.long()]
        
        holdout_count = Y_batch.sum(dim=1)
        holdout_count_reshaped = holdout_count.view(-1, 1)

        std = torch.exp(self.qUmain_std(X_batch) * 0.5)
        self.qUmain = self.qUmain_mean(X_batch) +  torch.randn_like(std) * std
        
        std = torch.exp(self.qUbias_std(X_batch) * 0.5)
        self.qUbias = self.qUbias_mean(X_batch) + torch.randn_like(std) * std
        
        self.qU = self.qUmain + self.qUbias
        
        std = torch.exp(self.qVmain_std(self.qU) * 0.5)
        self.qV = self.qVmain_mean(self.qU) + torch.randn_like(std) * std
        
        softmax_result = F.softmax(self.qV, dim=1)
        
        # Multiply reshaped holdout_count by the softmax result
        pred = holdout_count_reshaped * softmax_result
        
        cv = torch.mean(torch.abs(pred.squeeze() - Y_batch))
        
        SMAPE = torch.mean(
            torch.abs(pred.squeeze() - Y_batch + 1e-6)/(torch.abs(Y_batch)+torch.abs(pred.squeeze()) + 1e-6))
        
        return cv, SMAPE
    

    def fit(self, trainX, trainY, valX, valY, epochs=10):
        """ Fit the model """
        losses = []
        val_losses = []
        val_SMAPE = []
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = self.forward(trainX, trainY)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipnorm)
            self.optimizer.step()
            losses.append(loss.item())
            
            # Evaluate on validation data
            with torch.no_grad():
                self.eval()  # Set the model to evaluation mode
                val_loss, val_smape = self.validate(valX, valY)
                val_losses.append(val_loss.item())
                val_SMAPE.append(val_smape.item())
                self.train()  # Set the model back to training mode

            print(f'Epoch {epoch + 1}, Train_ELBO: {loss.item()}, Validation_SMAPE: {val_smape.item()}')
        
        
        return losses, val_losses, val_SMAPE

