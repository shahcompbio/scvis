import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
import yaml
import pandas as pd

from scvis.model import SCVIS
from scvis import plot
from scvis import data

import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except ImportError:
    import pickle

CURR_PATH = os.path.dirname(os.path.abspath(__file__))


def train(args):

    x, y, architecture, hyperparameter, train_data, model, normalizer, out_dir, name = \
        _init_model(args, 'train')
    iter_per_epoch = round(x.shape[0] / hyperparameter['batch_size'])

    max_iter = int(iter_per_epoch * hyperparameter['max_epoch'])

    if max_iter < 3000:
        max_iter = 3000
    elif max_iter > 30000:
        max_iter = np.max([30000, iter_per_epoch * 2])

    name += '_iter_' + str(max_iter)
    res = model.train(data=train_data,
                      batch_size=hyperparameter['batch_size'],
                      verbose=args.verbose,
                      verbose_interval=args.verbose_interval,
                      show_plot=args.show_plot,
                      plot_dir=os.path.join(out_dir, (name+"_intermediate_result")),
                      max_iter=max_iter,
                      pretrained_model=args.pretrained_model_file)
    model.set_normalizer(normalizer)

    # Save the trained model
    out_dir = args.out_dir
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    model_dir = os.path.join(out_dir, "model")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    model_name = name + ".ckpt"
    model_name = os.path.join(model_dir, model_name)
    model.save_sess(model_name)

    # The objective function trace plot
    elbo = res['elbo']
    tsne_cost = res['tsne_cost']

    iteration = len(elbo)
    avg_elbo = elbo - tsne_cost
    for i in range(iteration)[1:]:
        avg_elbo[i] = (elbo[i] - tsne_cost[i]) / i + \
                   avg_elbo[i-1] * (i-1) / i

    plot.plot_trace([range(iteration)] * 3,
                    [avg_elbo, elbo, tsne_cost],
                    ['avg_cost', 'elbo', 'tsne_cost'])

    fig_file = name + '_obj.png'
    fig_file = os.path.join(out_dir, fig_file)
    plt.savefig(fig_file)

    obj_file = name + '_obj.tsv'
    obj_file = os.path.join(out_dir, obj_file)
    res = pd.DataFrame(np.column_stack((elbo, tsne_cost)),
                       columns=['elbo', 'tsne_cost'])
    res.to_csv(obj_file, sep='\t', index=True, header=True)

    # Save the mapping results
    _save_result(x, y, model, out_dir, name)

    return()


def map(args):
    x, y, architecture, hyperparameter, train_data, model, _, out_dir, name = \
        _init_model(args, 'map')

    name = "_".join([name, "map"])
    _save_result(x, y, model, out_dir, name)

    return()


def _init_model(args, mode):
    x = pd.read_csv(args.data_matrix_file, sep='\t').values

    config = {}
    config_file = CURR_PATH + '/config/model_config.yaml'
    config_file = args.config_file or config_file
    try:
        config_file_yaml = open(config_file, 'r')
        config = yaml.load(config_file_yaml)
        config_file_yaml.close()
    except yaml.YAMLError as exc:
        print('Error in the configuration file: {}'.format(exc))

    architecture = config['architecture']
    architecture.update({'input_dimension': x.shape[1]})

    hyperparameter = config['hyperparameter']
    if hyperparameter['batch_size'] > x.shape[0]:
        hyperparameter.update({'batch_size': x.shape[0]})

    model = SCVIS(architecture, hyperparameter)
    normalizer = 1.0
    if args.pretrained_model_file is not None:
        model.load_sess(args.pretrained_model_file)
        normalizer = model.get_normalizer()

    if mode == 'train':
        if args.normalize is not None:
            normalizer = float(args.normalize)
        else:
            normalizer = np.max(np.abs(x))
    else:
        if args.normalize is not None:
            normalizer = float(args.normalize)

    x /= normalizer

    y = None
    if args.data_label_file is not None:
        label = pd.read_csv(args.data_label_file, sep='\t').values
        label = pd.Categorical(label[:, 0])
        y = label.codes

    # fixed random seed
    np.random.seed(0)
    train_data = data.DataSet(x, y)

    out_dir = args.out_dir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    name = '_'.join(['perplexity', str(hyperparameter['perplexity']),
                     'regularizer', str(hyperparameter['regularizer_l2']),
                     'batch_size', str(hyperparameter['batch_size']),
                     'learning_rate', str(hyperparameter['optimization']['learning_rate']),
                     'latent_dimension', str(architecture['latent_dimension']),
                     'activation', str(architecture['activation']),
                     'seed', str(hyperparameter['seed'])])

    return x, y, architecture, hyperparameter, train_data, model, normalizer, out_dir, name


def _save_result(x, y, model, out_dir, name):
    z_mu, _ = model.encode(x)
    plt.figure(figsize=(12, 8))
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c=y, s=10)

    if y is not None:
        plt.colorbar()

    fig_name = name + '.png'
    fig_name = os.path.join(out_dir, fig_name)
    plt.savefig(fig_name)

    z_mu = pd.DataFrame(z_mu, columns=['z_coordinate_'+str(i) for i in range(z_mu.shape[1])])
    map_name = name + '.tsv'
    map_name = os.path.join(out_dir, map_name)
    z_mu.to_csv(map_name, sep='\t', index=True, header=True)

    ##
    log_likelihood = model.get_log_likelihood(x)
    plt.figure(figsize=(12, 8))
    plt.scatter(z_mu.iloc[:, 0], z_mu.iloc[:, 1], c=log_likelihood, s=10)
    plt.colorbar()

    fig_name = name + '_log_likelihood' + '.png'
    fig_name = os.path.join(out_dir, fig_name)
    plt.savefig(fig_name)

    log_likelihood = pd.DataFrame(log_likelihood, columns=['log_likelihood'])
    map_name = name + '_log_likelihood' + '.tsv'
    map_name = os.path.join(out_dir, map_name)
    log_likelihood.to_csv(map_name, sep='\t', index=True, header=True)
