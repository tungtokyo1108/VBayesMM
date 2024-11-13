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
from biom import load_table, Table
from biom.util import biom_open
from skbio import OrdinationResults
from skbio.stats.composition import clr, centralize, closure
from skbio.stats.composition import clr_inv as softmax
from scipy.stats import entropy, spearmanr
from scipy.sparse import coo_matrix

import tensorflow as tf
from VBayesMM import VBayesMM

#------------------------------------------------------------------------------
###################### Test performance with real data ########################
#------------------------------------------------------------------------------ 

meta_df = pd.read_csv('Meta_OSA_data_filtered.csv', index_col="#SampleID")
meta_df_IHH = meta_df[meta_df["exposure_type"] == "IHH"]
meta_df_AIR = meta_df[meta_df["exposure_type"] == "Air"]
#metabolites_df = pd.read_csv('Metabolomics_OSA_data.csv', index_col="#featureID").T
microbes = load_table("haddad_6week_deblur_otus_unrare_hdf5.biom")
#microbes = load_table("haddad_6weeks_deblur_otus_rare2k_matched.biom")
metabolites = load_table("haddad_6weeks_allFeatures_pqn_matched.biom")
table=pd.read_table("allFeatures_dsfdr.txt", sep='\t', dtype=str, index_col="Unnamed: 0")
microbes_df = microbes.to_dataframe().T
metabolites_df = metabolites.to_dataframe()
metabolites_df = metabolites_df[table.index]

mic_samp=set(microbes_df.index)
met_samp=set(metabolites_df.index)
meta_IHH=set(meta_df_IHH.index)

matched_IHH = sorted(list(mic_samp & met_samp & meta_IHH))
microbes_df=microbes_df.loc[matched_IHH]
metabolites_df=metabolites_df.loc[matched_IHH]


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

model = VBayesMM(batch_size=50, latent_dim=3, err = "normal", errstd_v=0.01, 
                 temperature = 0.5, mu01 = 0, mu02 = 1, rho01=0, rho02=1, 
                 lambda01=0, lambda02=0.8, ssprior = "normal", errstd_epsilon=0.01,
                 #temperature = 0.5, mu01 = -0.5, mu02 = 0.5, rho01=-1, rho02=-1, 
                 #lambda01=0, lambda02=1, ssprior = "uniform", errstd_epsilon=0.01,
                 hard=False, threshold = 0.5)

config = tf.compat.v1.ConfigProto()

with tf.Graph().as_default(), tf.compat.v1.Session(config=config) as session:
    model(session,
          train_microbes_coo, train_metabolites_df.values,
          test_microbes_coo, test_metabolites_df.values)

    loss, cv, SMAPE = model.fit(epoch=1000)   
    
plt.plot(loss, "green")

plt.plot(cv, "red")

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



































































