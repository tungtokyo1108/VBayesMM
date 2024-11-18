# -*- coding: utf-8 -*-
"""

@author: TungDang
"""

import os
import time
import click
import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import biom
from biom import load_table, Table
from scipy.stats import entropy, spearmanr
from scipy.sparse import coo_matrix

import tensorflow as tf
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

microbes_df, metabolites_df = microbes_df.align(
    metabolites_df, axis=0, join='inner'
)

num_test = 20

sample_ids = set(np.random.choice(microbes_df.index, size=num_test))
sample_ids = np.array([(x in sample_ids) for x in microbes_df.index])

train_microbes_df = microbes_df.loc[~sample_ids]
test_microbes_df = microbes_df.loc[sample_ids]
train_metabolites_df = metabolites_df.loc[~sample_ids]
test_metabolites_df = metabolites_df.loc[sample_ids]


train_microbes_coo = coo_matrix(train_microbes_df.values)
test_microbes_coo = coo_matrix(test_microbes_df.values)


device_name='/cpu:0'


#### Initialize the VBayesMM model

model = VBayesMM()

config = tf.compat.v1.ConfigProto()

with tf.Graph().as_default(), tf.compat.v1.Session(config=config) as session:
    model(session,
          train_microbes_coo, train_metabolites_df.values,
          test_microbes_coo, test_metabolites_df.values)

    ELBO, MAE, SMAPE = model.fit(epoch=5000)   
    
plt.plot(ELBO, "green")

plt.plot(MAE, "red")

plt.plot(SMAPE, "red")

Umain = model.U
Ubias = model.Ubias
Umain_mean = model.U_mean

plt.figure(figsize=(6, 6))
sns.histplot(np.ravel(Umain), bins=50, kde=True, color='red', stat="count")


Umain_mean_gamma = model.U_mean_gamma
Umain_mean_gamma_df = pd.DataFrame(data=Umain_mean_gamma, index=train_microbes_df.columns)
Umain_mean_gamma_df = Umain_mean_gamma_df.mean(axis=1).sort_values(ascending=False)
Umain_mean_gamma_mean = np.sort(np.mean(Umain_mean_gamma, axis=1))[::-1]

plt.figure(figsize=(6, 6))
sns.histplot(Umain_mean_gamma_mean , bins=50, kde=True, color='red', stat="count", alpha=0.5)



