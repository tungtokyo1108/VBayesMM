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
from tqdm import tqdm
import tensorflow as tf
from tensorflow.compat.v1.distributions import Multinomial, Normal
from tensorflow_probability import distributions as tfd
import datetime


#------------------------------------------------------------------------------
#****************************** Utils Functions *******************************
#------------------------------------------------------------------------------


def sigmoid(z):
    return 1. / (1 + tf.exp(-z))

def logit(z):
    return tf.math.log(z/(1. - z))

def gumbel_softmax(logits, U, temperature=0.5, hard=False, threshold=0.5, eps=1e-8):
    z = logits + tf.math.log(U + eps) - tf.math.log(1 - U + eps)
    #y = tf.math.sigmoid(z / temperature)
    y = tf.math.sigmoid(-z / temperature)
    if not hard:
        return y
    y_hard = tf.cast(y > threshold, tf.float32)
    
    # Set gradients w.r.t. y_hard as gradients w.r.t. y
    y_hard = tf.stop_gradient(y_hard - y) + y
    return y_hard

def sample_gumbel(shape, eps=1e-8):
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)

def gumbel_softmax_r(logits, temperature=0.5, hard=False):
    
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    #y = tf.nn.softmax(gumbel_softmax_sample / temperature)
    y = tf.math.sigmoid((-gumbel_softmax_sample) / temperature)
    #y = tf.math.sigmoid((gumbel_softmax_sample) / temperature)

    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
                         y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y

#------------------------------------------------------------------------------
#******************* Variational Bayesian Neural Network **********************
#************************* with Spike-and-Slab prior **************************
#------------------------------------------------------------------------------


class VBayesMM(object):
    
    """ Variational Bayesian microbiome multiomics 
    
    Parameters
    ----------

    trainX : sparse array in coo format for microbiome data
    trainY : np.array for metabolite data
    testX : sparse array in coo format for microbiome data
    testY : np.array for metabolite data
    
    """

    def __init__(self, u_mean=0, u_scale=1, v_mean=0, v_scale=1,
                 batch_size=50, latent_dim=3, err = "normal", errstd_v=0.01,
                 learning_rate=0.1, beta_1=0.8, beta_2=0.9,
                 temperature = 0.5, mu01 = -0.6, mu02 = 0.6, rho01=-6., rho02=-6., lambda01=0, lambda02=1, 
                 hard = False, threshold = 0.5, ssprior = "normal", errstd_epsilon=0.01, 
                 clipnorm=10., device_name='/cpu:0', save_path=None):
        
        
        p = latent_dim
        self.device_name = device_name
        if save_path is None:
            basename = "logdir"
            suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            save_path = "_".join([basename, suffix])

        self.p = p
        self.u_mean = u_mean
        self.u_scale = u_scale
        self.v_mean = v_mean
        self.v_scale = v_scale
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.err = err
        self.errstd_v = errstd_v

        # Parameter for optimization
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
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
        self.errstd_epsilon = errstd_epsilon
        
        self.clipnorm = clipnorm
        self.save_path = save_path

    def __call__(self, session, trainX, trainY, testX, testY):
        
        self.session = session
        self.nnz = len(trainX.data)
        self.num_samples = trainX.shape[0]
        self.d1 = trainX.shape[1]
        self.d2 = trainY.shape[1]
        self.cv_size = len(testX.data)
        
        self.rho_prior = tf.convert_to_tensor([np.log(np.exp(1.3) - 1)], dtype=tf.float32)
        
        total = (self.d1 + 1) * self.p
        a = np.log(total) + 0.1 * ((1 + 1) * np.log(self.d1) + np.log(np.sqrt(self.num_samples) * self.d1))
        lm = 1 / np.exp(a)
        self.phi_prior = tf.convert_to_tensor(lm, dtype=tf.float32)

        # keep the multinomial sampling on the cpu

        with tf.device('/cpu:0'):
            X_ph = tf.SparseTensor(
                indices=np.array([trainX.row,  trainX.col]).T,
                values=trainX.data,
                dense_shape=trainX.shape)
            Y_ph = tf.constant(trainY, dtype=tf.float32)

            X_holdout = tf.SparseTensor(
                indices=np.array([testX.row,  testX.col]).T,
                values=testX.data,
                dense_shape=testX.shape)
            Y_holdout = tf.constant(testY, dtype=tf.float32)

            total_count = tf.reduce_sum(Y_ph, axis=1)
            # Assuming X_ph.values is a tensor containing logits
            logits = tf.math.log(tf.reshape(X_ph.values, [1, -1]))
            # Ensure logits are of floating type
            logits = tf.cast(logits, dtype=tf.float32)
            batch_ids = tfd.Multinomial(total_count=self.batch_size, logits=logits)

            batch_ids = batch_ids.sample()
            batch_ids = tf.squeeze(batch_ids)
            X_samples = tf.gather(X_ph.indices, 0, axis=1)
            X_obs = tf.gather(X_ph.indices, 1, axis=1)
            sample_ids = tf.gather(X_samples, tf.cast(batch_ids, dtype=tf.int32))

            Y_batch = tf.gather(Y_ph, sample_ids)
            X_batch = tf.gather(X_obs, tf.cast(batch_ids, dtype=tf.int32))

        with tf.device(self.device_name):
            
            # Spike-and-slab variational prior
            
            if self.ssprior == "uniform":
                self.qUmain_mean_w_mu = tf.Variable(tf.random.uniform(shape=[self.d1, self.p], minval=self.mu01, maxval=self.mu02), name='qUmain_mean_w_mu', trainable=True)
                self.qUmain_mean_w_rho = tf.Variable(tf.random.uniform(shape=[self.d1, self.p], minval=self.rho01, maxval=self.rho02), name='qUmain_mean_w_rho', trainable=True)
                self.qUmain_mean_w_theta = tf.Variable(logit(tf.random.uniform(shape=[self.d1, self.p], minval=self.lambda01, maxval=self.lambda02)), name='qUmain_mean_w_theta', trainable=True)
                self.qUmain_mean_gamma = None
                
                self.qUbias_mean_w_mu = tf.Variable(tf.random.uniform(shape=[self.d1, 1], minval=self.mu01, maxval=self.mu02), name='qUbias_mean_w_mu', trainable=True)
                self.qUbias_mean_w_rho = tf.Variable(tf.random.uniform(shape=[self.d1, 1], minval=self.rho01, maxval=self.rho02), name='qUbias_mean_w_rho', trainable=True)
                self.qUbias_mean_w_theta = tf.Variable(logit(tf.random.uniform(shape=[self.d1, 1], minval=self.lambda01, maxval=self.lambda02)), name='qUbias_mean_w_theta', trainable=True)
                self.qUbias_mean_gamma = None
                
            else:
                self.qUmain_mean_w_mu = tf.Variable(tf.random.normal(shape=[self.d1, self.p], mean=self.mu01, stddev=self.mu02), name='qUmain_mean_w_mu', trainable=True)
                self.qUmain_mean_w_rho = tf.Variable(tf.random.normal(shape=[self.d1, self.p], mean=self.rho01, stddev=self.rho02), name='qUmain_mean_w_rho', trainable=True)
                self.qUmain_mean_w_theta = tf.Variable(logit(tf.random.uniform(shape=[self.d1, self.p], minval=self.lambda01, maxval=self.lambda02)), name='qUmain_mean_w_theta', trainable=True)
                self.qUmain_mean_gamma = None
                
                self.qUbias_mean_w_mu = tf.Variable(tf.random.normal(shape=[self.d1, 1], mean=self.mu01, stddev=self.mu02), name='qUbias_mean_w_mu', trainable=True)
                self.qUbias_mean_w_rho = tf.Variable(tf.random.normal(shape=[self.d1, 1], mean=self.rho01, stddev=self.rho02), name='qUbias_mean_w_rho', trainable=True)
                self.qUbias_mean_w_theta = tf.Variable(logit(tf.random.uniform(shape=[self.d1, 1], minval=self.lambda01, maxval=self.lambda02)), name='qUbias_mean_w_theta', trainable=True)
                self.qUbias_mean_gamma = None
            
            
            # Sample Spike-and-slab variational prior
            
            qUmain_mean_sigma_w = tf.math.softplus(self.qUmain_mean_w_rho, name="qUmain_mean_sigma_w")
            qUbias_mean_sigma_w = tf.math.softplus(self.qUbias_mean_w_rho, name="qUbias_mean_sigma_w")
            sigma_prior = tf.math.softplus(self.rho_prior, name="sigma_prior")
            
            qUmain_mean_u_w = tf.random.uniform(shape=self.qUmain_mean_w_theta.shape, minval=0, maxval=1)
            qUbias_mean_u_w = tf.random.uniform(shape=self.qUbias_mean_w_theta.shape, minval=0, maxval=1)
            #self.qUmain_mean_gamma = gumbel_softmax(self.qUmain_mean_w_theta, qUmain_mean_u_w, temperature=self.temp, hard=self.hard, threshold = self.threshold)
            #self.qUbias_mean_gamma = gumbel_softmax(self.qUbias_mean_w_theta, qUbias_mean_u_w, temperature=self.temp, hard=self.hard, threshold = self.threshold)
            
            self.qUmain_mean_gamma = gumbel_softmax_r(self.qUmain_mean_w_theta, temperature=self.temp, hard=self.hard)
            self.qUbias_mean_gamma = gumbel_softmax_r(self.qUbias_mean_w_theta, temperature=self.temp, hard=self.hard)
            
            qUmain_mean_epsilon_w = tf.random.normal(shape=self.qUmain_mean_w_mu.shape, mean=0, stddev=self.errstd_epsilon)
            qUbias_mean_epsilon_w = tf.random.normal(shape=self.qUbias_mean_w_mu.shape, mean=0, stddev=self.errstd_epsilon)
            
            # Variational distributions for V matrix
            
            self.qVmain_mean = tf.Variable(tf.random.normal([self.p, self.d2-1]), name='qVmain_mean', trainable=True)
            self.qVmain_std = tf.Variable(tf.random.normal([self.p, self.d2-1]), name='qVmain_std', trainable=True)
            self.qVbias_mean = tf.Variable(tf.random.normal([1, self.d2-1]), name='qVbias_mean', trainable=True)
            self.qVbias_std = tf.Variable(tf.random.normal([1, self.d2-1]), name='qVbias_std', trainable=True)
            
            if self.err == "uniform":
                self.qUmain  = self.qUmain_mean_gamma * (self.qUmain_mean_w_mu + qUmain_mean_epsilon_w * qUmain_mean_sigma_w)
                self.qUbias = self.qUbias_mean_gamma * (self.qUbias_mean_w_mu + qUbias_mean_epsilon_w * qUbias_mean_sigma_w)
                self.qVmain = self.qVmain_mean + (tf.exp(self.qVmain_std * 0.5) * tf.random.uniform(shape=self.qVmain_std.shape, minval=-0.1, maxval=0.1))
                self.qVbias = self.qVbias_mean + (tf.exp(self.qVbias_std * 0.5) * tf.random.uniform(shape=self.qVbias_std.shape, minval=-0.1, maxval=0.1))
            else:
                
                self.qUmain  = self.qUmain_mean_gamma * (self.qUmain_mean_w_mu + qUmain_mean_epsilon_w * qUmain_mean_sigma_w)
                self.qUbias = self.qUbias_mean_gamma * (self.qUbias_mean_w_mu + qUbias_mean_epsilon_w * qUbias_mean_sigma_w)
                self.qVmain = self.qVmain_mean + (tf.exp(self.qVmain_std * 0.5) * tf.random.normal(shape=self.qVmain_std.shape, mean=0, stddev=self.errstd_v))
                self.qVbias = self.qVbias_mean + (tf.exp(self.qVbias_std * 0.5) * tf.random.normal(shape=self.qVbias_std.shape, mean=0, stddev=self.errstd_v))


            qU = tf.concat(
                [tf.ones([self.d1, 1]), self.qUbias, self.qUmain], axis=1)
            qV = tf.concat(
                [self.qVbias, tf.ones([1, self.d2-1]), self.qVmain], axis=0)
            

            du = tf.gather(qU, X_batch, axis=0, name='du')
            dv = tf.concat([tf.zeros([du.shape[0], 1]),
                            du @ qV], axis=1, name='dv')

            tc = tf.gather(total_count, sample_ids)
            Y = tfd.Multinomial(total_count=tc, logits=dv, name='Y')
            num_samples = trainX.shape[0]
            norm = num_samples / self.batch_size
            logprob_y = tf.reduce_sum(Y.log_prob(Y_batch), name='logprob_y')
            
            # Calculate the KL divergence part of the ELBO
            
            qUmain_mean_w_phi = tf.math.sigmoid(self.qUmain_mean_w_theta, name="qUmain_mean_w_phi")
            qUbias_mean_w_phi = tf.math.sigmoid(self.qUbias_mean_w_theta, name="qUbias_mean_w_phi")
            
            kl_Umain_mean = qUmain_mean_w_phi * (tf.math.log(qUmain_mean_w_phi) - tf.math.log(self.phi_prior)) + \
                        (1 - qUmain_mean_w_phi) * (tf.math.log(1 - qUmain_mean_w_phi) - tf.math.log(1 - self.phi_prior)) + \
                        qUmain_mean_w_phi * (tf.math.log(sigma_prior) - tf.math.log(qUmain_mean_sigma_w) + 
                                                  0.5 * (qUmain_mean_sigma_w ** 2 + self.qUmain_mean_w_mu ** 2) / sigma_prior ** 2 - 0.5)
            kl_Umain_mean = tf.reduce_sum(kl_Umain_mean, name="kl_Umain_mean")
            
            kl_Ubias_mean = qUbias_mean_w_phi * (tf.math.log(qUbias_mean_w_phi) - tf.math.log(self.phi_prior)) + \
                        (1 - qUbias_mean_w_phi) * (tf.math.log(1 - qUbias_mean_w_phi) - tf.math.log(1 - self.phi_prior)) + \
                            qUbias_mean_w_phi * (tf.math.log(sigma_prior) - tf.math.log(qUbias_mean_sigma_w) + 
                                                 0.5 * (qUbias_mean_sigma_w ** 2 + self.qUbias_mean_w_mu ** 2) / sigma_prior ** 2 - 0.5)
            kl_Ubias_mean = tf.reduce_sum(kl_Ubias_mean, name="kl_Ubias_mean")

            kl_Vmain = 0.5 * tf.reduce_sum(1 + self.qVmain_std - tf.square(self.qVmain_mean) - tf.exp(self.qVmain_std), name='kl_Vmain')
            kl_Vbias = 0.5 * tf.reduce_sum(1 + self.qVbias_std - tf.square(self.qVbias_mean) - tf.exp(self.qVbias_std), name="kl_Vbias")
            kl_divergence = kl_Vmain + kl_Vbias + kl_Umain_mean + kl_Ubias_mean 
            
            self.log_loss = -((logprob_y * norm) + kl_divergence)

        # keep the multinomial sampling on the cpu
        with tf.device('/cpu:0'):
            # cross validation
            with tf.name_scope('accuracy'):
                
                logits = tf.math.log(tf.reshape(X_holdout.values, [1, -1]))
                
                # Ensure logits are of floating type
                logits = tf.cast(logits, dtype=tf.float32)
                cv_batch_ids = tfd.Multinomial(total_count=self.cv_size, logits=logits)
                cv_batch_ids = cv_batch_ids.sample()
                cv_batch_ids = tf.squeeze(cv_batch_ids)
                X_cv_samples = tf.gather(X_holdout.indices, 0, axis=1)
                X_cv = tf.gather(X_holdout.indices, 1, axis=1)
                cv_sample_ids = tf.gather(X_cv_samples, tf.cast(cv_batch_ids, dtype=tf.int32))

                Y_cvbatch = tf.gather(Y_holdout, cv_sample_ids)
                X_cvbatch = tf.gather(X_cv, tf.cast(cv_batch_ids, dtype=tf.int32))
                holdout_count = tf.reduce_sum(Y_cvbatch, axis=1)
                
                qU_cv = tf.concat(
                    [tf.ones([self.d1, 1]), self.qUbias, self.qUmain], axis=1)
                qV_cv = tf.concat(
                    [self.qVbias, tf.ones([1, self.d2-1]), self.qVmain], axis=0)
                
                cv_du = tf.gather(qU_cv, X_cvbatch, axis=0, name='cv_du')
                pred = tf.reshape(
                    holdout_count, [-1, 1]) * tf.nn.softmax(
                        tf.concat([tf.zeros([
                            cv_du.shape[0], 1]),
                                   cv_du @ qV_cv], axis=1, name='pred')
                    )

                self.cv = tf.reduce_mean(
                    tf.squeeze(tf.abs(pred - Y_cvbatch))
                )
                
                self.SMAPE = tf.reduce_mean(
                    tf.squeeze(tf.abs(pred - Y_cvbatch + 1e-8)/(tf.abs(Y_cvbatch)+tf.abs(pred) + 1e-8))
                )
                
        
        with tf.device(self.device_name):
            with tf.name_scope('optimize'):
                optimizer = tf.compat.v1.train.AdamOptimizer(
                    self.learning_rate, beta1=self.beta_1, beta2=self.beta_2)

                gradients, self.variables = zip(
                    *optimizer.compute_gradients(self.log_loss))
                self.gradients, _ = tf.clip_by_global_norm(
                    gradients, self.clipnorm)
                self.train = optimizer.apply_gradients(
                    zip(self.gradients, self.variables))

        tf.compat.v1.global_variables_initializer().run()

    def fit(self, epoch=10, summary_interval=1000, checkpoint_interval=3600,
            testX=None, testY=None):
        
        iterations = epoch * self.nnz // self.batch_size
        losses, cvs, SMAPE = [], [], []
        cv = None
        last_checkpoint_time = 0
        last_summary_time = 0
        #saver = tf.compat.v1.train.Saver()
        now = time.time()
        for i in tqdm(range(0, epoch)):
            
            res = self.session.run(
                [self.train, self.log_loss, self.cv, self.SMAPE,
                    self.qUmain, self.qUbias,
                    self.qVmain, self.qVbias,
                    self.qUmain_mean_w_mu, self.qUbias_mean_w_mu, 
                    self.qUmain_mean_gamma, self.qUmain_mean_w_theta, 
                    self.qUbias_mean_gamma, self.qUbias_mean_w_theta,
                    self.qVmain_mean, self.qVmain_std]
            )
            train_, loss, cv, R2, rU, rUb, rV, rVb, rU_m, rUb_m, rU_m_g, rU_m_w_th, rUb_m_g, rUb_m_w_th, rV_m, rV_std = res
            losses.append(loss)
            cvs.append(cv)
            SMAPE.append(R2)

        self.U = rU
        self.U_mean = rU_m
        self.U_mean_gamma = rU_m_g
        self.U_mean_w_theta = rU_m_w_th
        self.U_std_w_theta = rU_m_w_th
        self.V = rV
        self.V_mean = rV_m
        self.V_std = rV_std
        self.Ubias = rUb
        self.Ubias_mean = rUb_m
        self.Ubias_mean_gamma = rUb_m_g
        self.Ubias_mean_w_theta = rUb_m_w_th
        self.Vbias = rVb

        return losses, cvs, SMAPE
    
