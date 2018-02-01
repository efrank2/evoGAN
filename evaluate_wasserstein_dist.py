import scipy as sp
import numpy as np
# import matplotlib.pylab as pl
# from mpl_toolkits.mplot3d import Axes3D  # noqa
# import ot
import time
from emd import emd

# def get_dist(n_samples):


# 	mu_s = np.array([0, 0])
# 	cov_s = np.array([[1, 0], [0, 1]])

# 	xs = ot.datasets.get_2D_samples_gauss(10000, mu_s, cov_s)

# 	mu_t = np.array([4, 4])
# 	cov_t = np.array([[1, 0], [0, 1]])

# 	P = sp.linalg.sqrtm(cov_t)

# 	xt = np.random.randn(10000, 2).dot(P) + mu_t

# 	xs2 = xs[:n_samples]
# 	xt2 = xt[:n_samples]

	
	
# 	C1 = sp.spatial.distance.cdist(xs2, xs2)
# 	C2 = sp.spatial.distance.cdist(xt2, xt2)

# 	C1 /= C1.max()
# 	C2 /= C2.max()

# 	p = ot.unif(n_samples)
# 	q = ot.unif(n_samples)

# 	gw_dist = ot.gromov_wasserstein2(C1, C2, p, q, 'square_loss', epsilon=5e-4)
# 	print('Gromov-Wasserstein distances between the distribution: ' + str(gw_dist))

def get_dist(n_samples):
	
	mu_X, mu_Y = 1.0, -1.0

	X = np.random.randn(n_samples,2) + mu_X
	Y = np.random.randn(n_samples,2) + mu_Y
	return emd(X,Y)

for n in [10, 20, 40, 80, 160, 250, 500, 750, 1000]:
	start = time.clock()
	dist = get_dist(n)
	print('Earth Mover Distance: ' + str(dist))
	end = time.clock()
	elapsed_time = end-start
	print "Number of Samples: %d, Time: %f" % (n, elapsed_time)