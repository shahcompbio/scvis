import tensorflow as tf
from scvis.tf_helper import weight_xavier_relu, bias_variable, shape
from collections import namedtuple

LAYER_SIZE = [128, 64, 32]
OUTPUT_DIM = 2
KEEP_PROB = 1.0
EPS = 1e-6
MAX_SIGMA_SQUARE = 1e10

LocationScale = namedtuple('LocationScale', ['mu', 'sigma_square'])


# =============================================================================
class MLP(object):
    def __init__(self, input_data, input_size, layer_size, output_dim,
                 activate_op=tf.nn.elu,
                 init_w_op=weight_xavier_relu,
                 init_b_op=bias_variable):
        self.input_data = input_data
        self.input_dim = shape(input_data)[1]
        self.input_size = input_size

        self.layer_size = layer_size
        self.output_dim = output_dim

        self.activate, self.init_w, self.init_b = \
            activate_op, init_w_op, init_b_op

        with tf.name_scope('encoder-net'):
            self.weights = [self.init_w([self.input_dim, layer_size[0]])]
            self.biases = [self.init_b([layer_size[0]])]

            self.hidden_layer_out = \
                tf.matmul(self.input_data, self.weights[-1]) + self.biases[-1]
            self.hidden_layer_out = self.activate(self.hidden_layer_out)

            for in_dim, out_dim in zip(layer_size, layer_size[1:]):
                self.weights.append(self.init_w([in_dim, out_dim]))
                self.biases.append(self.init_b([out_dim]))
                self.hidden_layer_out = self.activate(
                    tf.matmul(self.hidden_layer_out, self.weights[-1]) +
                    self.biases[-1])


class GaussianVAE(MLP):
    def __init__(self, input_data, input_size,
                 layer_size=LAYER_SIZE,
                 output_dim=OUTPUT_DIM,
                 decoder_layer_size=LAYER_SIZE[::-1]):
        super(self.__class__, self).__init__(input_data, input_size,
                                             layer_size, output_dim)

        self.num_encoder_layer = len(self.layer_size)

        with tf.name_scope('encoder-mu'):
            self.bias_mu = self.init_b([self.output_dim])
            self.weights_mu = self.init_w([self.layer_size[-1], self.output_dim])

        with tf.name_scope('encoder-sigma'):
            self.bias_sigma_square = self.init_b([self.output_dim])
            self.weights_sigma_square = self.init_w([self.layer_size[-1], self.output_dim])

        with tf.name_scope('encoder-parameter'):
            self.encoder_parameter = self.encoder()

        with tf.name_scope('sample'):
            self.ep = tf.random_normal(
                [self.input_size, self.output_dim],
                mean=0, stddev=1, name='epsilon_univariate_norm')

            self.z = tf.add(self.encoder_parameter.mu,
                            tf.sqrt(self.encoder_parameter.sigma_square) * self.ep,
                            name='latent_z')

        self.decoder_layer_size = decoder_layer_size
        self.num_decoder_layer = len(self.decoder_layer_size)

        with tf.name_scope('decoder'):
            self.weights.append(self.init_w([self.output_dim, self.decoder_layer_size[0]]))
            self.biases.append(self.init_b([self.decoder_layer_size[0]]))

            self.decoder_hidden_layer_out = self.activate(
                tf.matmul(self.z, self.weights[-1]) +
                self.biases[-1])

            for in_dim, out_dim in \
                    zip(self.decoder_layer_size, self.decoder_layer_size[1:]):
                self.weights.append(self.init_w([in_dim, out_dim]))
                self.biases.append(self.init_b([out_dim]))

                self.decoder_hidden_layer_out = self.activate(
                    tf.matmul(self.decoder_hidden_layer_out, self.weights[-1]) +
                    self.biases[-1])

            self.decoder_bias_mu = self.init_b([self.input_dim])
            self.decoder_weights_mu = \
                self.init_w([self.decoder_layer_size[-1],
                             self.input_dim])

            self.decoder_bias_sigma_square = self.init_b([self.input_dim])
            self.decoder_weights_sigma_square = \
                self.init_w([self.decoder_layer_size[-1],
                             self.input_dim])

            mu = tf.add(tf.matmul(self.decoder_hidden_layer_out,
                                  self.decoder_weights_mu),
                        self.decoder_bias_mu)
            sigma_square = tf.add(tf.matmul(self.decoder_hidden_layer_out,
                                            self.decoder_weights_sigma_square),
                                  self.decoder_bias_sigma_square)

            self.decoder_parameter = \
                LocationScale(mu, tf.clip_by_value(tf.nn.softplus(sigma_square),
                                                   EPS, MAX_SIGMA_SQUARE))

    def decoder(self, z):
        hidden_layer_out = self.activate(
            tf.matmul(z, self.weights[self.num_encoder_layer]) +
            self.biases[self.num_encoder_layer]
        )

        for layer in range(self.num_encoder_layer+1,
                           self.num_encoder_layer + self.num_decoder_layer):
            hidden_layer_out = self.activate(
                tf.matmul(hidden_layer_out, self.weights[layer]) +
                self.biases[layer])

        mu = tf.add(tf.matmul(hidden_layer_out, self.decoder_weights_mu),
                    self.decoder_bias_mu)
        sigma_square = tf.add(tf.matmul(hidden_layer_out,
                                        self.decoder_weights_sigma_square),
                              self.decoder_bias_sigma_square)

        return LocationScale(mu, tf.clip_by_value(tf.nn.softplus(sigma_square),
                                                  EPS, MAX_SIGMA_SQUARE))

    def encoder(self, prob=0.9):
        weights_mu = tf.nn.dropout(self.weights_mu, prob)
        mu = tf.add(tf.matmul(self.hidden_layer_out, weights_mu),
                    self.bias_mu)
        sigma_square = tf.add(tf.matmul(self.hidden_layer_out,
                                        self.weights_sigma_square),
                              self.bias_sigma_square)

        return LocationScale(mu,
                             tf.clip_by_value(tf.nn.softplus(sigma_square),
                                              EPS, MAX_SIGMA_SQUARE))
