"""
This script contains the functionality to generate 2D embedding for scRNA-seq data
The latent features generated by DAWN can also be used for the embedding
Author: Suhas Srinivasan
Date Created: 12/20/2018
Python Version: 2.7
"""

import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.preprocessing import scale
import sys
import timeit


def load_data(filename, delimiter=',', skip_rows=0, dtype=float, read_mode='r', transpose=False):
    """
    Method to load large datasets efficiently (better than using Numpy)
    :param filename:
    :param delimiter:
    :param skip_rows:
    :param dtype:
    :param read_mode:
    :param transpose:
    :return data:
    """
    if filename.find("latent_features.csv") > -1:
        skip_rows = 1

    def iter_func():
        with open(filename, read_mode) as infile:
            for _ in range(skip_rows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        load_data.rowlength = len(line)

    data = numpy.fromiter(iter_func(), dtype=dtype)
    data = data.reshape(-1, load_data.rowlength)
    # Required for deep learning
    if transpose:
        data = data.transpose()
    return data


def calc_embedding(input_path):
    """
    Method to run t-SNE for generating 2D coordinates
    :param input_path:
    :return xy:
    """
    data_set = load_data(input_path)
    if debug_stmts:
        print 'Dataset shape - ', str(data_set.shape)
        print 'Max. expression value - ', numpy.max(data_set)
        print 'Min. expression value - ', numpy.min(data_set)

    temp = numpy.where(~data_set.any(axis=0))[0]
    data_set = numpy.delete(data_set, temp, axis=1)
    data_set = scale(data_set)
    if debug_stmts:
        print 'Dataset shape - ', str(data_set.shape)
        print 'Max. expression value - ', numpy.max(data_set)
        print 'Min. expression value - ', numpy.min(data_set)

    print 't-SNE starting...'

    # Perplexity of 3-5 is sufficient for typical cluster distributions in scRNA-seq data
    perplex = 5
    n_comps = 2
    coord_file = input_path.rsplit('.', 1)[0] + '-2d_coord.csv'
    header = 'X,Y'
    start_time = timeit.default_timer()
    tsne = manifold.TSNE(n_components=n_comps, init='pca', perplexity=perplex, random_state=100)
    xy = tsne.fit_transform(data_set)
    end_time = timeit.default_timer()
    training_time = (end_time - start_time)
    print 'Total training time - %.2fm' % (training_time / 60.0)

    # Save coordinates so that plot style can be changed without running t-SNE again
    print 'Saving coordinates...'
    numpy.savetxt(coord_file, xy, fmt='%f', delimiter=',', header=header, comments='')
    return xy


def plot_embedding(input_path, xyz, annotate=False):
    """
    Method to plot the embedding, contains stylistic attributes
    :param input_path:
    :param xyz:
    :param annotate:
    :return:
    """
    mpl.rcParams['font.family'] = 'arial'
    mpl.rcParams['font.size'] = 5
    mpl.rcParams['axes.linewidth'] = 0.5
    labels = range(1, xyz.shape[0]+1)
    fig = plt.figure(figsize=(3, 3), dpi=200, frameon=False)
    area = 3
    ax = fig.gca()
    ax.scatter(xyz[:, 0], xyz[:, 1], s=area, alpha=0.85, linewidth=0)
    if annotate:
        for i, txt in enumerate(labels):
            ax.text(xyz[i, 0], xyz[i, 1], txt)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plot_path = input_path.rsplit('.', 1)[0] + '-2d_viz.tif'
    fig.tight_layout()
    fig.savefig(plot_path, format="tif")


if __name__ == '__main__':
    debug_stmts = False
    if len(sys.argv) > 1:
        given_path = sys.argv[1]
        print 'Visualizer started'
        print 'Given file - ', given_path
        xyz_coord = calc_embedding(given_path)
        plot_embedding(given_path, xyz_coord)
        print 'Visualizer completed'
    else:
        print 'Please provide the dataset file path'
