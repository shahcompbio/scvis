import sys
import numpy as np

MAX_VAL = np.log(sys.float_info.max) / 2.0

np.random.seed(0)


def compute_transition_probability(x, perplexity=30.0,
                                   tol=1e-4, max_iter=50, verbose=False):
    # x should be properly scaled so the distances are not either too small or too large

    if verbose:
        print('tSNE: searching for sigma ...')

    (n, d) = x.shape
    sum_x = np.sum(np.square(x), 1)

    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    p = np.zeros((n, n))

    # Parameterized by precision
    beta = np.ones((n, 1))
    entropy = np.log(perplexity) / np.log(2)

    # Binary search for sigma_i
    idx = range(n)
    for i in range(n):
        idx_i = list(idx[:i]) + list(idx[i+1:n])

        beta_min = -np.inf
        beta_max = np.inf

        # Remove d_ii
        dist_i = dist[i, idx_i]
        h_i, p_i = compute_entropy(dist_i, beta[i])
        h_diff = h_i - entropy

        iter_i = 0
        while np.abs(h_diff) > tol and iter_i < max_iter:
            if h_diff > 0:
                beta_min = beta[i].copy()
                if np.isfinite(beta_max):
                    beta[i] = (beta[i] + beta_max) / 2.0
                else:
                    beta[i] *= 2.0
            else:
                beta_max = beta[i].copy()
                if np.isfinite(beta_min):
                    beta[i] = (beta[i] + beta_min) / 2.0
                else:
                    beta[i] /= 2.0

            h_i, p_i = compute_entropy(dist_i, beta[i])
            h_diff = h_i - entropy

            iter_i += 1

        p[i, idx_i] = p_i

    if verbose:
        print('Min of sigma square: {}'.format(np.min(1 / beta)))
        print('Max of sigma square: {}'.format(np.max(1 / beta)))
        print('Mean of sigma square: {}'.format(np.mean(1 / beta)))

    return p


def compute_entropy(dist=np.array([]), beta=1.0):
    p = -dist * beta
    shift = MAX_VAL - max(p)
    p = np.exp(p + shift)
    sum_p = np.sum(p)

    h = np.log(sum_p) - shift + beta * np.sum(np.multiply(dist, p)) / sum_p

    return h, p / sum_p
