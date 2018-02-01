import tensorflow as tf
import numpy as np
import matplotlib
import pdb
import itertools
from scipy import linalg
import imageio
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import os
import pickle
import sklearn.datasets
import sklearn.metrics
from sklearn import mixture
from emd import emd
import scipy as sp

batch_size = 512
Z_dim      = 32

'''
Utility functions
'''

def save_as_gif(directory):
	from os import listdir
	from os.path import isfile, join
	filenames = [f for f in listdir(directory) if isfile(join(directory, f))]
	images = []
	for filename in filenames[1:]:
		images.append(imageio.imread(directory + '/' + filename))
	imageio.mimsave(directory + '/animation.gif', images)

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

def plot_results(X, Y_, means, covariances, index, title, directory='figs'):
    splot = plt.subplot(1, 1, 1)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim([-0.2,1.2])
    plt.ylim([-0.2,1.2])

    plt.title(title)
    plt.savefig(directory + '/%06d.png' % index, bbox_inches='tight')

def sample_Z(m, n):
    return np.random.normal(size=[m, n])

def make_grid_data(n_gauss=3, std=0.01):
	# Number of blobs along x and y
	nx, ny = (n_gauss, n_gauss)
	# Range 0 to 1
	x = np.linspace(0,1, nx)
	y = np.linspace(0,1, ny)
	xv, yv = np.meshgrid(x, y)
	xv, yv = list(np.reshape(xv, -1)), list(np.reshape(yv, -1))
	blob_centers = zip(xv, yv)

	dataset, _ = sklearn.datasets.make_blobs(n_samples=10000, centers=blob_centers, cluster_std=std)
	return dataset, blob_centers

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

'''
Model
'''

# Set up weights
D_width = 32
D_W1 = tf.Variable(xavier_init([2, D_width]))
D_b1 = tf.Variable(tf.zeros(shape=[D_width]))

D_W2 = tf.Variable(xavier_init([D_width, D_width]))
D_b2 = tf.Variable(tf.zeros(shape=[D_width]))

D_W3 = tf.Variable(xavier_init([D_width, D_width]))
D_b3 = tf.Variable(tf.zeros(shape=[D_width]))

D_W4 = tf.Variable(xavier_init([D_width, 1]))
D_b4 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3, D_W4, D_b4]

G_width = D_width

G_W1 = tf.Variable(xavier_init([Z_dim, G_width]))
G_b1 = tf.Variable(tf.zeros(shape=[G_width]))

G_W2 = tf.Variable(xavier_init([G_width, G_width]))
G_b2 = tf.Variable(tf.zeros(shape=[G_width]))

G_W3 = tf.Variable(xavier_init([G_width, G_width]))
G_b3 = tf.Variable(tf.zeros(shape=[G_width]))

G_W4 = tf.Variable(xavier_init([G_width, 2]))
G_b4 = tf.Variable(tf.zeros(shape=[2]))

theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3, G_W4, G_b4]

# Set up model
def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2) + G_h1
    G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3) + G_h2
    G_logit = tf.matmul(G_h3, G_W4) + G_b4
    return G_logit

def gan_discriminator(x_real, x_fake):
    """ I.e. Regular GAN discriminator """
    D_h1_real = tf.nn.relu(tf.matmul(x_real / 4, D_W1) + D_b1)
    D_h2_real = tf.nn.relu(tf.matmul(D_h1_real, D_W2) + D_b2) + D_h1_real
    D_h3_real = tf.nn.relu(tf.matmul(D_h2_real, D_W3) + D_b3) + D_h2_real
    D_logit_real = tf.matmul(D_h3_real, D_W4) + D_b4
    D_prob_real = tf.nn.sigmoid(D_logit_real)

    D_h1_fake = tf.nn.relu(tf.matmul(x_fake / 4, D_W1) + D_b1)
    D_h2_fake = tf.nn.relu(tf.matmul(D_h1_fake, D_W2) + D_b2) + D_h1_fake
    D_h3_fake = tf.nn.relu(tf.matmul(D_h2_fake, D_W3) + D_b3) + D_h2_fake
    D_logit_fake = tf.matmul(D_h3_fake, D_W4) + D_b4 
    D_prob_fake = tf.nn.sigmoid(D_logit_fake)
    return D_prob_real, D_logit_real, D_prob_fake, D_logit_fake

# Model placeholders
X = tf.placeholder(tf.float32, shape=[None, 2])
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])    
G_sample = generator(Z)

# Grab logits from model
D_prob_real, D_logit_real, D_prob_fake, D_logit_fake = gan_discriminator(X, G_sample) 

# Declare GAN losses
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

# Declare optimizers for G and D
D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(G_loss, var_list=theta_G)

# Grab the data
# data = read()
data, _ = make_grid_data(3)
data_size = len(data)
data = np.concatenate((data, data[:batch_size,:]), axis=0)

# Make and save vis of true distribution
save_fig_path = 'figs'
plt.figure(figsize=(5,5))
plt.plot(data[:1000,0], data[:1000,1], 'b.')
axes = plt.gca()
axes.set_xlim([-0.2,1.2])
axes.set_ylim([-0.2,1.2])
plt.title('True data distribution')
plt.savefig(save_fig_path + '/real.png', bbox_inches='tight')

num_runs = 20

all_emd = []

for i in range(num_runs):
    directory = save_fig_path + '/run%d' % i
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Start the session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Train
    np_samples = []
    plot_every = 1000
    dists = []
    for it in range(50000):
        start_idx = it*batch_size%data_size
        X_mb = data[start_idx:start_idx+batch_size, :]

        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(batch_size, Z_dim)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X: X_mb, Z: sample_Z(batch_size, Z_dim)})
          
        if (it+1) % plot_every == 0:
            samples = sess.run(G_sample, feed_dict={Z: sample_Z(1000, Z_dim)})
            dist = emd(np.random.permutation(samples)[:250], np.random.permutation(data)[:250])
            dists.append(dist)
            np_samples.append(samples)
            # Clear plot
            plt.clf()     
            # Fit GMM to output
            dpgmm = mixture.BayesianGaussianMixture(n_components=9,
            	covariance_type='full').fit(samples)
            plot_results(samples, dpgmm.predict(samples), dpgmm.means_, dpgmm.covariances_, it,
            	'Iter: {}, loss(D): {:2.2f}, loss(G):{:2.2f}'.format(it+1, D_loss_curr, G_loss_curr), directory)
    
    all_emd.append(dists)

    plt.clf()
    plt.plot(dists)
    plt.ylabel('Earth Mover Distance (Target, Samples)')
    plt.xlabel('Epochs')
    plt.title('Earth Mover Distance GAN Through Training')
    plt.savefig(directory + '/emd.png', bbox_inches='tight')
    plt.clf()
    save_as_gif(directory)

import pickle
pickle.dump(all_emd, open("earth_mover_distances.p", "wb"))