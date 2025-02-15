#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: tungbioinfo
"""

import os
import time
import click
import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from biom import load_table, Table
from biom.util import biom_open
import pandas as pd 
import seaborn as sns
from scipy.spatial.distance import pdist
from skbio.stats.composition import clr, centralize, clr_inv
from scipy.sparse import coo_matrix

import torch
from VBayesMM import VBayesMM


#------------------------------------------------------------------------------
###################### Test performance with real data ########################
#------------------------------------------------------------------------------

microbes = load_table("microbes.biom")
metabolites = load_table("metabolites.biom")
microbes_df = microbes.to_dataframe()
metabolites_df = metabolites.to_dataframe()

microbes_df = microbes_df.astype(pd.SparseDtype("float64",fill_value=0))
metabolites_df = metabolites_df.astype(pd.SparseDtype("float64",fill_value=0))
microbes_df, metabolites_df = microbes_df.align(metabolites_df, axis=0, join='inner')

num_test = 20

sample_ids = set(np.random.choice(microbes_df.index, size=num_test))
sample_ids = np.array([(x in sample_ids) for x in microbes_df.index])

train_microbes_df = microbes_df.loc[~sample_ids]
test_microbes_df = microbes_df.loc[sample_ids]
train_metabolites_df = metabolites_df.loc[~sample_ids]
test_metabolites_df = metabolites_df.loc[sample_ids]

n, d1 = train_microbes_df.shape
n, d2 = train_metabolites_df.shape

train_microbes_coo = coo_matrix(train_microbes_df.values)
test_microbes_coo = coo_matrix(test_microbes_df.values)
trainY_torch = torch.tensor(train_metabolites_df.to_numpy(), dtype=torch.float32)
testY_torch = torch.tensor(test_metabolites_df.to_numpy(), dtype=torch.float32)


# Initialize the VBayesMM model

model = VBayesMM(d1=d1, d2=d2, num_samples=n, batch_size=n) 

losses, val_losses, val_SMAPE = model.fit(train_microbes_coo, trainY_torch, test_microbes_coo, testY_torch, epochs=5000)

plt.plot(losses, "green")
plt.xlabel('Iteration', size=15)
plt.ylabel('ELBO', size=15)

plt.plot(val_losses, "red")

plt.plot(val_SMAPE, "red")
plt.xlabel('Iteration', size=15)
plt.ylabel('SMAPE', size=15)

# Microbiome U and metabolite V matrices plt.plot(val_SMAPE, "red")

u_mean = np.array(model.qUmain_mean.weight.data.detach())
plt.figure(figsize=(6, 6))
sns.histplot(np.ravel(u_mean), bins=50, kde=True, color='red')

u_gamma_mean = np.array(model.qUmain_mean_gamma.detach())
Umain_mean_gamma_mean = np.sort(np.mean(u_gamma_mean, axis=1))[::-1]
plt.figure(figsize=(6, 6))
sns.histplot(Umain_mean_gamma_mean , bins=50, kde=True, color='red', stat="count", alpha=0.5)























































