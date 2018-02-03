scvis is a python package for dimension reduction of high-dimensional biological data, especially single-cell RNA-sequencing (scRNA-seq) data.


# License

scvis is free for academic/non-profit use.


# Versions

## 0.1.0


# Installation

To install scvis, please make sure that you have the necessary libraries (below) installed.
After that scvis can be installed from terminal: 

```shell
# In terminal
python setup.py install
```

Dependencies:

  * tensorflow >= 1.1
  * PyYAML >= 3.11
  * matplotlib >= 1.5.1
  * numpy >= 1.11.1
  * pandas >= 0.19.1


# How to use

After installing scvis, you can use the `scvis` command.

## 1, the `train` function

The `train` function can be used to learn a probabilistic parametric mapping (the exact directories of the input files should change based on their actual positions in the computer system):

```shell
# In terminal
scvis train --data_matrix_file ./data/bipolar_pca100.tsv \
    --out_dir ./output/bipolar \
    --data_label_file ./data/bipolar_label.tsv \
    --verbose \
    --verbose_interval 50
```

  * `--data_matrix_file`: a high-dimensional data matrix with the first row as the column names, in the tab delimited format. Each row represents a data point, e.g., the expression profile of a cell.
  * `--out_dir` (optional): path for output files
  * `--data_label_file` (optional): a one column file (with column header) provides the corresponding cluster information for each data point, just used for coloring scatter plots
  * `--verbose` (optional): the program will print progress information to the screen if this flag is set 
  * `--verbose_interval` (optional): the mini-bach interval to show running information

A trained model is saved in the folder `./output/bipolar/model/`

In addition to the model file, the low-dimensional embedding and the log-likelihoods are also written to two files in `./output/bipolar`,
and are shown as two scatter plots colored by the given label information and the log-likelihoods (the log-likelihood files are names as `*_log_likelihood.tsv` and `*_log_likelihood.png`).

The different components of the objective function are also saved to a file (`*_obj.tsv`) and shown in a graph (`*_obj.png`).
If you want to plot intermediate embeddings during optimizations, you can set the flag: `--show_plot`

By default, the `data_matrix_file` is normalized by the maximum absolute value. If you want to provide a positive float number for normalization, you can set (`--normalize your_number`).

Another important parameter is `--config_file`, which allows you to set various parameters. If you want to use your own config file, you can pass it as a parameter with flag: `--config_file`. The default config file is in `scvis/config/model_config.yaml`,  and you can use this file as a template to set parameters.  
```shell
# In terminal
scvis train --data_matrix_file ./data/bipolar_pca100.tsv \
    --out_dir ./output/bipolar \
    --data_label_file ./data/bipolar_label.tsv \
    --verbose \
    --verbose_interval 50 \ 
    --config_file model_config.yaml
```


## 2, the `map` function
After learning a probabilistic parametric mapping, the `map` function can be used to add new data to an existing embedding:

```shell
# In terminal
scvis map --data_matrix_file ./data/retina_pca100_bipolar.tsv \
    --out_dir ./output/retina \
    --pretrained_model_file ./output/bipolar/model/xxx.ckpt
```

  * `--data_matrix_file`: a high-dimensional data matrix with the first row as the column names, in tab delimited format
  * `--out_dir` (optional): path for output files
  * `--pretrained_model_file`: a pre-trained scvis model by calling the `scvis train`, where `xxx` should be replaced by the checkpoint file prefix in the model folder.

As for calling the `train` command, this command will also output the likelihood files and the low-dimensional embedding files, but without the model files and the objective function trace file and plots.

The data matrix files for calling both `train` and `map` should be normalized similarly, i.e., the parameters used to normalize the training data should be used to normalize the test data. This is the default setting. You can also pass a positive float number to normalize your data: `--normalize your_number`. 

For `map`, you can also pass the config file as a parameter with flag: `--config_file`. Notice that the `config_file` for `train` and `map` should be the same. 






