import tensorflow as tf
import numpy as np
import pdb

class Model():
    def __init__(self, config):

    	self.config = config
        self.build_model()
        self.init_saver()

    def build_model(self):
    	self.X = tf.placeholder(tf.float32, shape = [None, 2])
    	self.Z = tf.placeholder(tf.float32, shape = [None, self.config.Z_dim])  
    	self.g_samples = self.build_generator(self.Z)
    	self.d_logits_real, _ = self.build_discriminator(self.X)
    	self.d_logits_fake, _ = self.build_discriminator(self.g_samples, reuse=True)

    	# Declare GAN losses
    	with tf.name_scope("loss"):
		    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		    							logits = self.d_logits_real,
		    							labels = tf.ones_like(self.d_logits_real)))
		    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		    							logits = self.d_logits_fake,
		    							labels = tf.zeros_like(self.d_logits_fake)))
		    self.d_loss = d_loss_real + d_loss_fake
		    self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		    							logits = self.d_logits_fake,
		    							labels = tf.ones_like(self.d_logits_fake)))

		    # get discriminator and generator parameters
		    theta_g = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
		    theta_d = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')

		    # declare trainers
		    self.d_solver = tf.train.AdamOptimizer(learning_rate = 1e-4, beta1 = 0.5).minimize(self.d_loss, var_list=theta_d)
		    self.g_solver = tf.train.AdamOptimizer(learning_rate = 1e-4, beta1 = 0.5).minimize(self.g_loss, var_list=theta_g)


    def build_generator(self, z):
    	with tf.variable_scope("generator"):
	    	l1 = tf.layers.dense(z, 32, activation = tf.nn.relu)
	    	l2 = tf.layers.dense(l1, 32, activation = tf.nn.relu)
	    	l3 = tf.layers.dense(l2, 32, activation = tf.nn.relu)
	    	logit = tf.layers.dense(l3, 2, activation = None)
        return logit

    def build_discriminator(self, x, reuse=False):
        """ I.e. Regular GAN discriminator """
        with tf.variable_scope("discriminator", reuse = reuse):
	        l1 = tf.layers.dense(x, 32, activation = tf.nn.relu)
	        l2 = tf.layers.dense(l1, 32, activation = tf.nn.relu)
	        l3 = tf.layers.dense(l2, 32, activation = tf.nn.relu)
	        logit = tf.layers.dense(l3, 1, activation = None)
	       	prob = tf.nn.sigmoid(logit)
        return logit, prob

    def init_saver(self):
        #here you initalize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

        pass