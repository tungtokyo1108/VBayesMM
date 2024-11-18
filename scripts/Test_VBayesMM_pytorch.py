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

model = VBayesMM(d1=d1, d2=d2, num_samples=n, batch_size=n, device='cpu', 
                 subsample_size = 10, latent_dim=3,
                 unorm_type = 2.0, vnorm_type = 0, 
     #           temperature = 0.5, mu01 = 0, mu02 = 1, rho01=0, rho02=1, lambda01=0, lambda02=1, ssprior = "normal", 
                 temperature = 0.5, mu01 = -0.6, mu02 = 0.6, rho01=-6., rho02=-6., lambda01=0.99, lambda02=0.99, ssprior = "uniform",
                 hard=False, threshold = 0.5)

losses, val_losses = model.fit(train_microbes_coo, trainY_torch, test_microbes_coo, testY_torch, epochs=5000)

plt.plot(losses, "green")

plt.plot(val_losses, "red")





























































