import tensorflow as tf 
import numpy as np
import pdb

x_ = tf.placeholder(tf.float32, [None, 32])
def cppn(embedding, n_hid_layer=3):

	outp = tf.nn.relu(tf.layers.batch_normalization(tf.layers.dense(inputs=embedding, units=32, activation=None)))
	for i in range(n_hid_layer):
		outp = tf.nn.relu(tf.layers.batch_normalization(tf.layers.dense(inputs=outp, units=32, activation=None)))
	outp = tf.layers.dense(inputs=outp, units=1, activation=tf.nn.tanh)
	return outp

# Emb dim = complexity of embedding
# Also add output num configurability
def random_embedding(layer_width, emb_dim=4):
	emb = tf.random_normal([layer_width*layer_width, emb_dim])
	return emb

def xlinear(inp, width, emb_dim=4):
	emb0 = random_embedding(width, emb_dim)
	cppn_out = cppn(emb0)
	weights = tf.reshape(cppn_out, (width, width))
	outp = tf.matmul(inp, weights)
	return outp

# xnet = xlinear(x_, 32)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# x_inp = np.random.normal(size=(20, 32))
# y_ = sess.run(xnet, feed_dict={x_:x_inp})
# pdb.set_trace()