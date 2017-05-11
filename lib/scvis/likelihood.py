import tensorflow as tf
import numpy as np

EPS = 1e-20


def log_likelihood_gaussian(x, mu, sigma_square):
    return tf.reduce_sum(-0.5 * tf.log(2.0 * np.pi) - 0.5 * tf.log(sigma_square) -
                         (x - mu) ** 2 / (2.0 * sigma_square), 1)


def log_likelihood_student(x, mu, sigma_square, df=2.0):
    sigma = tf.sqrt(sigma_square)

    dist = tf.contrib.distributions.StudentT(df=df,
                                             loc=mu,
                                             scale=sigma)
    return tf.reduce_sum(dist.log_prob(x), reduction_indices=1)
