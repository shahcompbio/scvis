import os
import numpy as np
import tensorflow as tf
from datetime import datetime

from matplotlib import pyplot as plt

from scvis.vae import GaussianVAE
from scvis.likelihood import log_likelihood_student
from scvis.tsne_helper import compute_transition_probability


# =============================================================================
class SCVIS(object):
    def __init__(self, architecture, hyperparameter):
        self.eps = 1e-20

        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()

        self.architecture, self.hyperparameter = architecture, hyperparameter
        self.regularizer_l2 = self.hyperparameter['regularizer_l2']
        self.n = self.hyperparameter['batch_size']
        self.perplexity = self.hyperparameter['perplexity']

        tf.set_random_seed(self.hyperparameter['seed'])

        # Place_holders
        self.batch_size = tf.placeholder(dtype=tf.int32)
        self.x = tf.placeholder(tf.float32, shape=[None, self.architecture['input_dimension']])
        self.z = tf.placeholder(tf.float32, shape=[None, self.architecture['latent_dimension']])

        self.p = tf.placeholder(tf.float32, shape=[None, None])
        self.iter = tf.placeholder(dtype=tf.float32)

        self.vae = GaussianVAE(self.x,
                               self.batch_size,
                               self.architecture['inference']['layer_size'],
                               self.architecture['latent_dimension'],
                               decoder_layer_size=self.architecture['model']['layer_size'])

        self.encoder_parameter = self.vae.encoder_parameter
        self.latent = dict()
        self.latent['mu'] = self.encoder_parameter.mu
        self.latent['sigma_square'] = self.encoder_parameter.sigma_square
        self.latent['sigma'] = tf.sqrt(self.latent['sigma_square'])

        self.decoder_parameter = self.vae.decoder_parameter
        self.dof = tf.Variable(tf.constant(1.0, shape=[self.architecture['input_dimension']]),
                               trainable=True, name='dof')
        self.dof = tf.clip_by_value(self.dof, 0.1, 100, name='dof')

        with tf.name_scope('ELBO'):
            self.weight = tf.clip_by_value(tf.reduce_sum(self.p, 0), 0.01, 2.0)

            self.log_likelihood = tf.reduce_mean(tf.multiply(
                log_likelihood_student(self.x,
                                       self.decoder_parameter.mu,
                                       self.decoder_parameter.sigma_square,
                                       self.dof),
                self.weight), name="log_likelihood")

            self.kl_divergence = \
                tf.reduce_mean(0.5 * tf.reduce_sum(self.latent['mu'] ** 2 +
                                                   self.latent['sigma_square'] -
                                                   tf.log(self.latent['sigma_square']) - 1,
                                                   reduction_indices=1))
            self.kl_divergence *= tf.maximum(0.1, self.architecture['input_dimension']/self.iter)
            self.elbo = self.log_likelihood - self.kl_divergence

        self.z_batch = self.vae.z

        with tf.name_scope('tsne'):
            self.kl_pq = self.tsne_repel() * tf.minimum(self.iter, self.architecture['input_dimension'])

        with tf.name_scope('objective'):
            self.obj = self.kl_pq + self.regularizer() - self.elbo

        # Optimization
        with tf.name_scope('optimizer'):
            learning_rate = self.hyperparameter['optimization']['learning_rate']

            if self.hyperparameter['optimization']['method'].lower() == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate)
            elif self.hyperparameter['optimization']['method'].lower() == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate,
                                                        beta2=0.999,
                                                        epsilon=0.0001)

            gradient_clipped = self.clip_gradient()

            self.train_op = self.optimizer.apply_gradients(gradient_clipped, name='minimize_cost')

        self.saver = tf.train.Saver()

    def clip_gradient(self, clip_value=3.0, clip_norm=10.0):
        trainable_variable = self.sess.graph.get_collection('trainable_variables')
        grad_and_var = self.optimizer.compute_gradients(self.obj, trainable_variable)

        grad_and_var = [(grad, var) for grad, var in grad_and_var if var is not None]
        grad, var = zip(*grad_and_var)
        grad, global_grad_norm = tf.clip_by_global_norm(grad, clip_norm=clip_norm)

        grad_clipped_and_var = [(tf.clip_by_value(grad[i], -clip_value*0.1, clip_value*0.1), var[i])
                                if 'encoder-sigma' in var[i].name
                                else (tf.clip_by_value(grad[i], -clip_value, clip_value), var[i])
                                for i in range(len(grad_and_var))]

        return grad_clipped_and_var

    def regularizer(self):
        penalty = [tf.nn.l2_loss(var) for var in
                   self.sess.graph.get_collection('trainable_variables')
                   if 'weight' in var.name]

        l2_regularizer = self.regularizer_l2 * tf.add_n(penalty)

        return l2_regularizer

    def tsne_repel(self):
        nu = tf.constant(self.architecture['latent_dimension'] - 1, dtype=tf.float32)

        sum_y = tf.reduce_sum(tf.square(self.z_batch), reduction_indices=1)
        num = -2.0 * tf.matmul(self.z_batch,
                               self.z_batch,
                               transpose_b=True) + tf.reshape(sum_y, [-1, 1]) + sum_y
        num = num / nu

        p = self.p + 0.1 / self.n
        p = p / tf.expand_dims(tf.reduce_sum(p, reduction_indices=1), 1)

        num = tf.pow(1.0 + num, -(nu + 1.0) / 2.0)
        attraction = tf.multiply(p, tf.log(num))
        attraction = -tf.reduce_sum(attraction)

        den = tf.reduce_sum(num, reduction_indices=1) - 1
        repellant = tf.reduce_sum(tf.log(den))

        return (repellant + attraction) / self.n

    def _train_batch(self, x, t):
        p = compute_transition_probability(x, perplexity=self.perplexity)

        feed_dict = {self.x: x,
                     self.p: p,
                     self.batch_size: x.shape[0],
                     self.iter: t}

        _, elbo, tsne_cost = self.sess.run([
            self.train_op,
            self.elbo,
            self.kl_pq],
            feed_dict=feed_dict)

        return elbo, tsne_cost

    def train(self, data, max_iter=1000, batch_size=None,
              pretrained_model=None, verbose=True, verbose_interval=50,
              show_plot=True, plot_dir='./img/'):

        max_iter = max_iter
        batch_size = batch_size or self.hyperparameter['batch_size']

        status = dict()
        status['elbo'] = np.zeros(max_iter)
        status['tsne_cost'] = np.zeros(max_iter)

        if pretrained_model is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.load_sess(pretrained_model)

        start = datetime.now()
        for iter_i in range(max_iter):
            x, y = data.next_batch(batch_size)

            status_batch = self._train_batch(x, iter_i+1)
            status['elbo'][iter_i] = status_batch[0]
            status['tsne_cost'][iter_i] = status_batch[1]

            if verbose and iter_i % verbose_interval == 0:
                print('Batch {}'.format(iter_i))
                print((
                    'elbo: {}\n'
                    'scaled_tsne_cost: {}\n').format(
                    status['elbo'][iter_i],
                    status['tsne_cost'][iter_i]))

                if show_plot:
                    z_mu, _ = self.encode(x)
                    plt.figure(figsize=(10, 7))

                    plt.scatter(z_mu[:, 0], z_mu[:, 1], c=y, s=5)

                    if not os.path.isdir(plot_dir):
                        os.mkdir(plot_dir)

                    name = os.path.join(plot_dir, 'embedding_iter_%05d.png' % iter_i)
                    plt.savefig(name)
                    plt.close()

        print('Time used for training: {}\n'.format(datetime.now() - start))

        return status

    def encode(self, x):
        var = self.vae.encoder()
        feed_dict = {self.x: x}

        return self.sess.run(var, feed_dict=feed_dict)

    def decode(self, z):
        var = self.vae.decoder(tf.cast(z, tf.float32))
        feed_dict = {self.z: z, self.batch_size: z.shape[0]}

        return self.sess.run(var, feed_dict=feed_dict)

    def encode_decode(self, x):
        var = [self.latent['mu'],
               self.latent['sigma_square'],
               self.decoder_parameter.mu,
               self.decoder_parameter.sigma_square]

        feed_dict = {self.x: x, self.batch_size: x.shape[0]}

        return self.sess.run(var, feed_dict=feed_dict)

    def save_sess(self, model_name):
        self.saver.save(self.sess, model_name)

    def load_sess(self, model_name):
        self.saver.restore(self.sess, model_name)

    def get_log_likelihood(self, x, dof=None):

        dof = dof or self.dof
        log_likelihood = log_likelihood_student(
            self.x,
            self.decoder_parameter.mu,
            self.decoder_parameter.sigma_square,
            dof
        )
        num_samples = 5

        feed_dict = {self.x: x, self.batch_size: x.shape[0]}
        log_likelihood_value = 0

        for i in range(num_samples):
            log_likelihood_value += self.sess.run(log_likelihood, feed_dict=feed_dict)

        log_likelihood_value /= np.float32(num_samples)

        return log_likelihood_value

    def get_elbo(self, x):
        log_likelihood = log_likelihood_student(
            self.x,
            self.decoder_parameter.mu,
            self.decoder_parameter.sigma_square,
            self.dof
        )
        kl_divergence = tf.reduce_mean(0.5 * tf.reduce_sum(self.latent['mu'] ** 2 +
                                                           self.latent['sigma_square'] -
                                                           tf.log(self.latent['sigma_square']) - 1,
                                       reduction_indices=1))

        feed_dict = {self.x: x, self.batch_size: x.shape[0]}

        return self.sess.run(log_likelihood - kl_divergence, feed_dict=feed_dict)
