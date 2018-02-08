import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn import mixture

class DataGenerator():
    def __init__(self, config):
    	self.config = config
    	self.dataset, _, self.gmm = self.make_grid_dataset(config.dataset_dim)

    def make_grid_dataset(self, dim = 3):
	    # Number of blobs along x and y
	    nx, ny = (dim, dim)
	    # Range 0 to 1
	    x = np.linspace(0,1, nx)
	    y = np.linspace(0,1, ny)
	    xv, yv = np.meshgrid(x, y)
	    xv, yv = list(np.reshape(xv, -1)), list(np.reshape(yv, -1))
	    blob_centers = zip(xv, yv)
	    dataset, _ = sklearn.datasets.make_blobs(n_samples=self.config.dataset_n_samples, 
	    										 centers=blob_centers,
	    										 cluster_std=self.config.dataset_std)
	    gmm = sklearn.mixture.GaussianMixture(n_components=dim*dim).fit(dataset)

	    return dataset, blob_centers, gmm

    def next_batch(self, batch_size):
    	# Each batch is randomly sampled from 10,000 points in the true data distribution.
    	# Is that kosher?
        idx = np.random.choice(self.config.dataset_n_samples, batch_size)
        yield self.dataset[idx]