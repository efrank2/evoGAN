from tqdm import tqdm
import numpy as np
from utils.utils import sample_Z
import tensorflow as tf
import pdb, time

class Trainer():
    def __init__(self, sess, config, model, data):
        self.model = model
        # self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.sess.run(tf.global_variables_initializer())

    def train_epoch(self):
        loop = tqdm(range(self.config.dataset_n_samples/self.config.batch_size))
        losses = []
        for it in loop:
            loss = self.train_step()
            losses.append(loss)

        # Sample from generator
        samples = self.sess.run(self.model.g_samples, feed_dict={self.model.Z: sample_Z(5000, 32)})

        # Get average likelihood of GAN generated samples wrt real distribution
        likelihood = -self.data.gmm.score(samples)
        
        hist = np.histogram(self.data.gmm.predict(samples), bins=range(10))

        # Aim for uniform histogram (even distribution among gmm components)
        target_hist = np.ones(9)*samples.shape[0]/9.

        # Now find the L2 Distance between this and a uniform histogram
        diversity = np.linalg.norm(hist[0]-target_hist)/len(samples)

        return losses, samples, likelihood, diversity
        # cur_it = self.model.global_step_tensor.eval(self.sess)
        # summaries_dict = {}
        # summaries_dict['loss'] = loss
        # summaries_dict['acc'] = acc
        # self.logger.summarize(cur_it, summaries_dict=summaries_dict)

    def train_step(self):

        batch = next(self.data.next_batch(self.config.batch_size))

        _, d_loss_curr = self.sess.run([self.model.d_solver, self.model.d_loss], feed_dict={self.model.X: batch, self.model.Z: sample_Z(self.config.batch_size, self.config.Z_dim)})
        _, g_loss_curr = self.sess.run([self.model.g_solver, self.model.g_loss], feed_dict={self.model.X: batch, self.model.Z: sample_Z(self.config.batch_size, self.config.Z_dim)})
        
        return d_loss_curr, g_loss_curr