import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym



class PPO(object):
	def __init__(
			self,
			A_DIM,
			S_DIM,
			A_LR = 0.001,
			C_LR = 0.001,
			A_UPDATE_STEPS = 16,
			C_UPDATE_STEPS = 16,
			
	):
		self.sess = tf.Session()

		self.S_DIM = S_DIM
		self.A_DIM = A_DIM
		self.A_LR = A_LR
		self.C_LR = C_LR
		self.A_UPDATE_STEPS = A_UPDATE_STEPS
		self.C_UPDATE_STEPS = C_UPDATE_STEPS


		self.tfstate = tf.placeholder(tf.float32, [None, self.S_DIM], 'state')
		self.tfaction = tf.placeholder(tf.int32, [None, 1], 'action')
		self.tfadvantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
		self.tfdc = tf.placeholder(tf.float32, [None, 1], 'discounted_reward')

		# define critic net
		with tf.variable_scope("critic"):
			# input state buffer output v
			w_init = tf.initializers.he_uniform()
			l1 = tf.layers.dense(self.tfstate, 24, tf.nn.relu, kernel_initializer = w_init, bias_initializer = w_init)
			l2 = tf.layers.dense(l1, 24, tf.nn.relu, kernel_initializer = w_init, bias_initializer = w_init)

			self.v = tf.layers.dense(l2, 1, kernel_initializer = w_init, bias_initializer = w_init)
			# calculate advantage
			self.advantage = self.tfdc - self.v
			# define critic loss
			self.closs = tf.reduce_mean(tf.square(self.advantage))
			# define optimizer
			self.ctrain_op = tf.train.AdamOptimizer(self.C_LR).minimize(self.closs)

		# actor
		self.pi, self.pi_para = self._build_anet('pi', trainable=True)
		self.oldpi, self.oldpi_para = self._build_anet('oldpi', trainable=False)

		with tf.variable_scope('update_oldpi'):
			self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.pi_para, self.oldpi_para)]
		a_indices = tf.stack([tf.range(tf.shape(self.tfaction)[0], dtype=tf.int32), tf.squeeze(self.tfaction, axis = 1)], axis=1)
		pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)   # shape=(None, )
		oldpi_prob = tf.gather_nd(params=self.oldpi, indices=a_indices)  # shape=(None, )

		with tf.variable_scope("actorloss"):
			with tf.variable_scope("surrogate"):
				ratio = pi_prob/oldpi_prob
				surr = ratio * self.tfadvantage
			self.aloss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1 - 0.2, 1 + 0.2)*self.tfadvantage))


		with tf.variable_scope("train_actor"):
			self.atrain_op = tf.train.AdamOptimizer(self.A_LR).minimize(self.aloss)

		tf.summary.FileWriter("log/", self.sess.graph)
		self.sess.run(tf.global_variables_initializer())
	
	def update_target(self):
		self.sess.run(self.update_oldpi_op)
	
	def update(self, s, a, r):
		
		advantage = self.sess.run(self.advantage, {self.tfstate:s, self.tfdc:r})

		#update critic
		for _ in range(self.C_UPDATE_STEPS):
			self.sess.run(self.ctrain_op, {self.tfstate:s, self.tfdc:r})

		#update actor
		for _ in range(self.A_UPDATE_STEPS):
			self.sess.run(self.atrain_op, {self.tfstate:s, self.tfadvantage:advantage, self.tfaction:a})

	def _build_anet(self, name, trainable):
		with tf.variable_scope(name):
			w_initializer = tf.initializers.he_uniform()
			l_a = tf.layers.dense(self.tfstate, 24, tf.nn.relu, kernel_initializer = w_initializer, bias_initializer = w_initializer, trainable=trainable)
			l_a2 = tf.layers.dense(l_a, 24, tf.nn.relu, kernel_initializer = w_initializer, bias_initializer = w_initializer,trainable=trainable)
			a_prob = tf.layers.dense(l_a2, self.A_DIM, tf.nn.softmax, kernel_initializer = w_initializer, bias_initializer = w_initializer, trainable=trainable)
		params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
		return a_prob, params


	def choose_action(self, s):  # run by a local
		prob_weights = self.sess.run(self.pi, feed_dict={self.tfstate: s[None, :]})
		action = np.random.choice(range(prob_weights.shape[1]),
									  p=prob_weights.ravel())  # select action w.r.t the actions prob
		return action
	def get_v(self, s):
		if s.ndim < 2: s = s[np.newaxis, :]
		return self.sess.run(self.v, {self.tfstate: s})[0, 0]

