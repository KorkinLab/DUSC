"""
This script contains the functionality for DAWN
DAWN uses deep learning to generate a latent representation of scRNA-seq data
Includes source code from Theano
Author: Suhas Srinivasan
Date Created: 12/20/2018
Python Version: 2.7
"""

import numpy
from sklearn.decomposition import PCA
import sys
import timeit
import theano
import theano.tensor as tensor
from theano.tensor.shared_randomstreams import RandomStreams

debug_stmts = False


class DaE(object):
    """
    Denoising Autoencoder (DAE) class
    """
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=None,
        n_hidden=None,
        weights=None,
        bias_hid=None,
        bias_vis=None
    ):
        """
        Initialize the class
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for standalone DAE

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type weights: theano.tensor.TensorType
        :param weights: Theano variable pointing to a set of weights that can be
                  shared. If DAE is standalone set this to None.

        :type bias_hid: theano.tensor.TensorType
        :param bias_hid: Theano variable pointing to a set of bias values for
                     hidden units that can be shared. If DAE is standalone set this to None.

        :type bias_vis: theano.tensor.TensorType
        :param bias_vis: Theano variable pointing to a set of bias values for
                     visible units that can be shared. If DAE is standalone set this to None.
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : weights' was written as `weights_prime` and bias' as `bias_prime`
        if not weights:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            weights = theano.shared(value=initial_W, name='weights', borrow=True)

        if not bias_vis:
            bias_vis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bias_hid:
            bias_hid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='bias',
                borrow=True
            )

        self.weights = weights
        # bias corresponds to the bias of the hidden units
        self.bias = bias_hid
        # bias_prime corresponds to the bias of the visible units
        self.bias_prime = bias_vis
        # Tied weights, therefore weights_prime is weights transpose
        self.weights_prime = self.weights.T
        self.theano_rng = theano_rng
        # If no input is given, generate a variable representing the input
        if input is None:
            # Use a matrix because a minibatch of several examples is expected, each example being a row
            self.x = tensor.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.weights, self.bias, self.bias_prime]

    def get_corrupted_input(self, input, corruption_level):
        """
        Method to corrupt entries of the input by zeroing out
        randomly selected subsets of size 'corruption_level'
        :param input:
        :param corruption_level:
        :return corrupt_input:
        """
        if corruption_level > 0.0:
            corrupt_input = self.theano_rng.binomial(size=input.shape, n=1,
                                            p=1 - corruption_level,
                                            dtype=theano.config.floatX) * input
        else:
            corrupt_input = input

        return corrupt_input

    def get_hidden_values(self, input):
        """
        Computes the values of the hidden layer
        :param input:
        :return hidden_val:
        """
        hidden_val = tensor.nnet.sigmoid(tensor.dot(input, self.weights) + self.bias)
        return hidden_val

    def get_reconstructed_input(self, hidden):
        """
        Computes the reconstructed input given the values of the hidden layer
        :param hidden:
        :return recon_input:
        """
        recon_input = tensor.nnet.sigmoid(tensor.dot(hidden, self.weights_prime) + self.bias_prime)
        return recon_input

    def get_cost_updates(self, corruption_level, learning_rate):
        """
        This function computes the cost and the updates for one trainng step of the DAE
        :param corruption_level:
        :param learning_rate:
        :return cost, updates:
        """
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # Sum over the size of a datapoint.
        # When using minibatches, L will be a vector, with one entry per example in minibatch
        L = - tensor.sum(self.x * tensor.log(z) + (1 - self.x) * tensor.log(1 - z), axis=1)
        # L is now a vector, where each element is the cross-entropy cost of the reconstruction of the
        # corresponding example of the minibatch.
        # Compute the average of all these to get the cost of the minibatch
        cost = tensor.mean(L)

        # Compute the gradients of the cost of the DAE with respect to its parameters
        gparams = tensor.grad(cost, self.params)
        # Generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        return(cost, updates)


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


def estimate_neurons(sc_data):
    """
    Method to estimate no. of neurons required in the hidden layer
    :param sc_data:
    :return neurons:
    """
    print 'Estimating hidden neurons...'
    # Keep 95% of variance
    analysis = PCA(0.95, svd_solver='full')
    x_fit = analysis.fit_transform(sc_data)
    # print x_fit.shape
    neurons = analysis.n_components_
    print 'No. of hidden neurons estimated - ', neurons
    return neurons


def preprocess_data(input_path):
    """
    Method to preprocess data (Clean, Normalize and Binarize)
    :param input_path:
    :return norm_data:
    :return norm_data, neurons
    """
    print 'Begin preprocessing...'
    data_set = load_data(input_path)
    if debug_stmts:
        print 'Dataset shape - ', str(data_set.shape)
        print 'Max. expression value - ', numpy.max(data_set)
        print 'Min. expression value - ', numpy.min(data_set)

    temp = numpy.where(~data_set.any(axis=0))[0]
    # print temp.shape
    temp_data = numpy.delete(data_set, temp, axis=1)
    # print temp_data.shape
    neurons = estimate_neurons(temp_data)
    # Normalize data
    norm_data = (temp_data - temp_data.min(0)) / temp_data.ptp(0)

    # Convert to 32-bit precision for deep learning
    norm_data = numpy.float32(norm_data)
    if debug_stmts:
        print 'Cleaned dataset shape - ', str(data_set.shape)
        print 'Max. normalized value - ', numpy.max(norm_data)
        print 'Min. normalized value - ', numpy.min(norm_data)

    # Save binary file (helpful during execution/multiple runs for large datasets)
    print 'Saving preprocessed dataset...'
    output_path = input_path.rsplit('.', 1)[0] + '-pre_procs.bin'
    save_file = open(output_path, 'wb')
    numpy.save(save_file, norm_data)
    save_file.close()
    print 'Preprocessing completed'
    return(norm_data, neurons)


def deep_learning(input_path, dataset, hidden_units):
    """
    Method to perform feature learning using the Denoising Autoencoder (DAE)
    :param input_path:
    :param dataset:
    :param hidden_units:
    :return:
    """
    # Stable parameters for the DAE
    # Changes to these parameters affects feautre learning
    training_epochs = 250
    learning_rate = 0.05
    batch_size = 20
    corruption_level = 0.10

    # Store the data in shared variables
    # When using GPU, this helps to copy data to GPU memory (faster than copying a minibatch everytime)
    dataset = dataset.transpose()
    train_data = theano.shared(numpy.asarray(dataset, dtype=theano.config.floatX), borrow=True)
    visible_units = dataset.shape[1]
    train_batches = train_data.get_value(borrow=True).shape[0] // batch_size
    print 'Initializing model...'
    print 'Training epochs - ', training_epochs
    print 'Learning rate - ', learning_rate
    print 'Batch size - ', batch_size
    print 'Training batches - ', train_batches
    print 'Visible units - ', visible_units
    print 'Hidden units - ', hidden_units
    print 'Corruption level - ', corruption_level

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    index = tensor.lscalar()  # Index to a minibatch
    x = tensor.matrix('x')  # Initialize input

    # Build the model
    da = DaE(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=visible_units,
        n_hidden=hidden_units
    )

    cost, updates = da.get_cost_updates(
        corruption_level=corruption_level,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_data[index * batch_size: (index + 1) * batch_size]
        }
    )

    # Train the model
    print 'Training model...'
    start_time = timeit.default_timer()
    for epoch in range(training_epochs):
        c = []
        for batch_index in range(train_batches):
            c.append(train_da(batch_index))
        print 'Epoch %d, cost - %f' % (epoch, numpy.mean(c))

    end_time = timeit.default_timer()
    training_time = (end_time - start_time)
    print 'Training finished'
    print 'Total training time - %.2fm' % (training_time / 60.0)
    print 'Saving latent features...'
    weights = numpy.asarray(da.weights.get_value(borrow=True).T)
    output_file = input_path.rsplit('.', 1)[0] + '-latent_features.csv'
    header = ''
    length = hidden_units
    list = range(1, length+1)
    for i in list:
        if i < length:
            header = header + 'F' + str(i) + ','
        else:
            header = header + 'F' + str(i)
    numpy.savetxt(output_file, weights.transpose(), fmt='%f', delimiter=',', header=header, comments='')


if __name__ == '__main__':
    debug_stmts = False
    if len(sys.argv) > 1:
        given_path = sys.argv[1]
        print 'DAWN started'
        print 'Given file - ', given_path
        pro_data, neurons = preprocess_data(given_path)
        deep_learning(given_path, pro_data, neurons)
        print 'DAWN completed'
    else:
        print 'Please provide the dataset file path'


